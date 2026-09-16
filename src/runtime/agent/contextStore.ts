import type {
  AudioSource,
  ContextDocument,
  MeetingPerspective,
  SessionKind,
  TranscriptSegment,
} from "../../app/types";
import { SessionFactStore } from "./sessionFacts";

// Segments stay resident far longer than the live window: the live prompt only
// ever sees `windowMs` of them, but long-range memory (language/stack anchors and
// callback recall) needs the older turns to still be around. A 30-minute
// transcript is well under a megabyte, so this costs nothing.
const MAX_TRANSCRIPT_AGE_MS = 1_800_000;

// What the coach itself has already told the candidate. The transcript only
// holds what was *said out loud*; the candidate routinely answers with "嗯，好
// 的" instead of reading the card, so without this the coach has no memory of
// its own previous answer and answers every follow-up from scratch — the
// single biggest reason a follow-up ("那第二种呢?") degrades into a generic
// answer, or repeats advice the candidate already gave.
const MAX_COACH_TURNS = 3;
const MAX_COACH_TURN_ANSWER_CHARS = 400;
const MAX_COACH_TURNS_CHARS = 3_000;

export type CoachTurn = {
  /**
   * The question this answer responded to. For `coach` turns this is the
   * interviewer's question as heard; for `manual` turns it is whatever the user
   * typed/asked via the manual-ask path.
   */
  question: string;
  /** What the coach answered. Truncated on ingest. */
  answer: string;
  atMs: number;
  /**
   * Who asked. Matters because the two paths mean different things: a `coach`
   * turn is "the interviewer asked X and we answered", while a `manual` turn is
   * "the user asked X and we answered". Presenting them identically would make
   * the model believe the interviewer said things the user actually typed.
   */
  origin: "coach" | "manual";
};

export type ContextSnapshot = {
  documents: ContextDocument[];
  recentTranscript: TranscriptSegment[];
  durableTranscript: TranscriptSegment[];
  /** Coach answers already given this session, oldest first. */
  previousCoachTurns: CoachTurn[];
  /** Compact "what this session is about" line, or "" when nothing is known. */
  anchors: string;
  latestSegment: TranscriptSegment | null;
  perspective: MeetingPerspective;
  sessionKind: SessionKind;
  audioSource: AudioSource;
  goal: string;
  sessionId: string | null;
};

export class ContextStore {
  private documents: ContextDocument[] = [];
  private perspective: MeetingPerspective = "candidate";
  private sessionKind: SessionKind = "remote";
  private audioSource: AudioSource = "system";
  private goal = "";
  private sessionId: string | null = null;
  private segments: TranscriptSegment[] = [];
  private coachTurns: CoachTurn[] = [];
  private facts = new SessionFactStore();
  private factIngested = new Set<string>();
  private lastFactMs = 0;

  clear() {
    this.segments = [];
    this.coachTurns = [];
    this.facts.reset();
    this.factIngested.clear();
    this.lastFactMs = 0;
  }

  /**
   * Record an answer the coach just gave. Called on the completion path, so the
   * next follow-up can build on it instead of restarting from zero.
   */
  pushCoachTurn(turn: CoachTurn) {
    const answer = (turn.answer || "").trim();
    if (!answer) return;
    const next: CoachTurn[] = [
      ...this.coachTurns,
      {
        question: (turn.question || "").trim(),
        answer:
          answer.length > MAX_COACH_TURN_ANSWER_CHARS
            ? `${answer.slice(0, MAX_COACH_TURN_ANSWER_CHARS)}…`
            : answer,
        atMs: turn.atMs,
        origin: turn.origin === "manual" ? "manual" : "coach",
      },
    ];
    this.coachTurns = next.slice(-MAX_COACH_TURNS);

    // Drop the oldest turns until the block fits the budget. One pathological
    // answer must not push the whole prompt past the model's comfort zone.
    while (this.coachTurns.length > 1 && this.coachTurnChars() > MAX_COACH_TURNS_CHARS) {
      this.coachTurns = this.coachTurns.slice(1);
    }
    if (this.coachTurns.length === 1 && this.coachTurnChars() > MAX_COACH_TURNS_CHARS) {
      const only = this.coachTurns[0];
      this.coachTurns = [
        { ...only, answer: `${only.answer.slice(0, MAX_COACH_TURNS_CHARS)}…` },
      ];
    }
  }

  /**
   * Read-only view for the manual-ask path, which lives outside the runtime and
   * therefore cannot go through `snapshot()`. Manual ask sends its history to
   * Rust as `turns`; without this it would only ever see its own turns and
   * would answer a follow-up with no knowledge of what the coach already said.
   */
  coachTurnSnapshot(): CoachTurn[] {
    return this.coachTurns;
  }

  private coachTurnChars() {
    return this.coachTurns.reduce((sum, turn) => sum + turn.question.length + turn.answer.length, 0);
  }

  setDocuments(documents: ContextDocument[]) {
    this.documents = documents;
  }

  setPerspective(perspective: MeetingPerspective) {
    this.perspective = perspective;
  }

  setSessionConfig(config: { kind: SessionKind; audioSource: AudioSource; goal: string }) {
    this.sessionKind = config.kind;
    this.audioSource = config.audioSource;
    this.goal = config.goal.trim();
  }

  setSessionId(sessionId: string | null) {
    this.sessionId = sessionId;
  }

  pushTranscript(segment: TranscriptSegment) {
    this.segments = [...this.segments, segment].sort((left, right) => left.endMs - right.endMs);
    this.evictOldSegments(MAX_TRANSCRIPT_AGE_MS);
    this.primeFacts(segment);
  }

  /**
   * Extract long-range facts from a segment that has not been pushed yet.
   * Speculative prefetch runs *before* the segment reaches the store but still
   * needs the session anchors, otherwise its answer is generated with zero
   * memory of what this interview is about — the classic "asked Java, answered
   * JavaScript" failure. Idempotent: re-ingesting a segment is a no-op.
   */
  primeFacts(segment: TranscriptSegment) {
    if (this.factIngested.has(segment.id)) return;
    this.factIngested.add(segment.id);
    this.lastFactMs = Math.max(this.lastFactMs, segment.endMs ?? 0);
    this.facts.ingest([segment]);
  }

  /**
   * Compact "what this session is about" line, or "" when nothing is known.
   * The clock comes from the segments themselves, never `Date.now()` — segment
   * timestamps are not guaranteed to share an epoch with the wall clock, and a
   * mismatched "now" would silently decay every fact to zero salience.
   */
  anchors(): string {
    const latest = this.segments[this.segments.length - 1];
    const nowMs = Math.max(latest?.endMs ?? 0, this.lastFactMs);
    return nowMs > 0 ? this.facts.anchors(nowMs) : "";
  }

  snapshot(windowMs: number): ContextSnapshot {
    const latest = this.segments[this.segments.length - 1] ?? null;
    if (!latest) {
      return {
        documents: this.documents,
        latestSegment: null,
        perspective: this.perspective,
        sessionKind: this.sessionKind,
        audioSource: this.audioSource,
        goal: this.goal,
        recentTranscript: [],
        durableTranscript: [],
        previousCoachTurns: this.coachTurns,
        anchors: "",
        sessionId: this.sessionId,
      };
    }

    return {
      documents: this.documents,
      latestSegment: latest,
      perspective: this.perspective,
      sessionKind: this.sessionKind,
      audioSource: this.audioSource,
      goal: this.goal,
      recentTranscript: this.segments.filter(
        (segment) => latest.endMs - segment.endMs <= windowMs
      ),
      durableTranscript: this.segments,
      previousCoachTurns: this.coachTurns,
      anchors: this.facts.anchors(latest.endMs),
      sessionId: this.sessionId,
    };
  }

  private evictOldSegments(maxAgeMs: number) {
    const latest = this.segments[this.segments.length - 1];
    if (!latest) return;

    this.segments = this.segments.filter(
      (segment) => latest.endMs - segment.endMs <= maxAgeMs
    );
  }
}
