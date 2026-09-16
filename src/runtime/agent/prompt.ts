import type { CoachTurn, ContextSnapshot } from "./contextStore";
import type { WakeEvent } from "./wake";
import { summarizeContextDocuments } from "../../app/contextDocuments";
import { debugLog } from "../../app/platform";
import { isCallbackQuestion, recallEarlierTurns } from "./sessionFacts";

export type AgentPrompt = {
  wake: WakeEvent;
  snapshot: ContextSnapshot;
  text: string;
};

const COACH_TURN_RULE =
  "If the latest question is a follow-up to something in the previous coach answers (它 / 这个 / 那个 / 第二种 / 刚才 / 为什么, or it simply continues the same topic), build on that answer and go one level deeper: never repeat it, never contradict it, and never restart from scratch. If the interviewer moved to a new topic, ignore the previous answers completely.";

// Interviewer-only variant. Same, plus an escape hatch: the candidate's voice was
// never transcribed, so the remembered answer may not match what they actually
// said. Without this the model defends its own suggestion against the
// interviewer's wording — the one source that IS reliable.
const COACH_TURN_RULE_INTERVIEWER_ONLY =
  COACH_TURN_RULE +
  " But if the interviewer's own words describe a different approach from your previous answer, the interviewer wins: answer about THEIR approach and drop yours, because the candidate may have said something entirely different.";

export function buildAgentPrompt(wake: WakeEvent, snapshot: ContextSnapshot): AgentPrompt {
  const recent = snapshot.recentTranscript
    .map((segment) => segment.text.trim())
    .filter(Boolean)
    .join("\n");

  const documents = summarizeContextDocuments(snapshot.documents);

  // Long-range memory. Both pieces are pure string work (no LLM, no I/O) and stay
  // bounded, so they add no latency: the anchor line is a few dozen characters and
  // the recall block only appears when the question is a bare callback.
  const question = wake.evidence[0] ?? "";
  const anchors = snapshot.anchors;
  const earlierContext = isCallbackQuestion(question)
    ? recallEarlierTurns(
        question,
        snapshot.durableTranscript,
        snapshot.recentTranscript[0]?.endMs ?? Number.POSITIVE_INFINITY
      )
    : "";
  // The coach's own previous answers. The transcript alone is not enough: it
  // only holds what was actually spoken, and candidates routinely answer with
  // "嗯，好的" instead of reading the card — so on the follow-up the model would
  // otherwise have no idea what it already told them.
  const previousCoachBlock = formatPreviousCoachTurns(
    snapshot.previousCoachTurns,
    snapshot.audioSource === "system"
  );

  // Observability: this is the only way to tell "anchors weren't extracted" apart
  // from "anchors were extracted but the model ignored them". The island windows
  // are contentProtected so screenshots are useless for this.
  const anchorLines = anchors
    ? anchors
        .split("\n")
        .filter((line) => line.startsWith("- "))
        .map((line) => line.slice(2).trim())
    : [];
  debugLog(
    `[session-memory] wake=${wake.kind} anchors=${anchorLines.length}[${anchorLines.join(" | ")}] earlier=${earlierContext ? "hit" : "miss"} coachTurns=${snapshot.previousCoachTurns.length} durable=${snapshot.durableTranscript.length} recent=${snapshot.recentTranscript.length}`
  );

  const roleInstruction = snapshot.perspective === "interviewer"
    ? "The user is the interviewer. Help them fairly evaluate the candidate, ask sharper follow-up questions, and connect the discussion to the candidate's resume. Do not trick or embarrass the candidate."
    : "The user is a job candidate in a live interview. Act as a silent, real-time interview coach who feeds them the actual answer they can say out loud. Stay in the candidate's first-person voice. Never speak as a meeting negotiator or business advisor.";

  return {
    wake,
    snapshot,
    text: [
      `Wake kind: ${wake.kind}`,
      `Wake reason: ${wake.reason}`,
      `Session kind: ${snapshot.sessionKind}`,
      `Audio source: ${snapshot.audioSource}`,
      `User perspective: ${snapshot.perspective}`,
      "",
      roleInstruction,
      "",
      // Anchors go BEFORE the transcript so the model reads the session's
      // language/stack as established fact, not as one more line of dialogue.
      ...(anchors
        ? [
            anchors,
            // "language" here means PROGRAMMING language — spelled out because a
            // bare "answer in that language" gets misread as "switch the spoken
            // language", which would derail a Chinese-language interview.
            "The anchors above are established facts about this session. A listed `language` is the PROGRAMMING language the interview is conducted in: answer technical questions in that language and its ecosystem, and never substitute another one (a Java question must never be answered with JavaScript, TypeScript, or Python). The SPOKEN language of your answer must still match the transcript.",
            "",
          ]
        : []),
      "Uploaded reference material:",
      documents || "(none)",
      "",
      "Recent transcript:",
      recent || "(none)",
      "",
      ...(earlierContext ? [earlierContext, ""] : []),
      ...(previousCoachBlock ? [previousCoachBlock, ""] : []),
      "You are the right-side realtime interview coach. Give the candidate a complete answer they can deliver out loud.",
      ...(previousCoachBlock
        ? [
            // Without this guard the block becomes a trap: the model happily
            // continues its own previous answer instead of answering the new
            // question. Say explicitly when to build on it and when to drop it.
            snapshot.audioSource === "system"
              ? COACH_TURN_RULE_INTERVIEWER_ONLY
              : COACH_TURN_RULE,
          ]
        : []),
      // 总-分 (conclusion-first). A flat paragraph is technically complete but
      // sounds like rambling when spoken aloud — the candidate's first breath
      // must land the point, then the unpacking follows.
      "For a candidate, answer the interviewer's latest question using 总-分 structure: open with the conclusion in one or two sentences that make sense on their own, then unpack it into 2-4 short speakable points that each add something new. Never flatten a conceptual, technical, or comparison question into one continuous paragraph. Include concrete numbers, named things, or a short code outline when the question is technical. Behavioural questions (\"tell me about a time...\") may stay a short narrative instead.",
      "For an interviewer, use the resume plus external company/product context to suggest fair follow-up questions, evidence checks, and evaluation angles. Do not design trick questions.",
      "Never say you detected a question. Never produce a meeting summary or negotiate like a business advisor.",
      "Do not pad with throat-clearing; get straight to the substance.",
    ].join("\n"),
  };
}

/**
 * Render the coach's own previous answers as a labelled block. Plain text, no
 * LLM and no I/O — a few hundred characters that are already in memory.
 */
function formatPreviousCoachTurns(turns: CoachTurn[], interviewerOnly: boolean): string {
  if (!turns || turns.length === 0) return "";

  const body = turns
    .map((turn) => {
      const question = turn.question.trim();
      // A manual ask is the *user* asking, not the interviewer. Labelling it the
      // same as a coach turn would make the model believe the interviewer said
      // it, and it would then happily "build on" a question that was never asked.
      const tag = turn.origin === "manual" ? "(asked by the user, not the interviewer)" : "";
      const head = question ? `Q: ${question}${tag ? ` ${tag}` : ""}\n` : "";
      return `${head}A: ${turn.answer.trim()}`;
    })
    .join("\n\n");

  // With only the interviewer transcribed the candidate's voice never reaches the
  // transcript, so this block is an *assumption* about what was said, not a record
  // of it. Say so explicitly: otherwise the model treats its own suggestion as
  // established fact and, measured on 2026-09-11, answers the interviewer's
  // Sentinel question with the Redis design it suggested itself.
  const note = interviewerOnly
    ? "what you already told the candidate earlier in this session, oldest first; the latest question may be a follow-up to one of these. NOTE: only the interviewer is being transcribed, so you CANNOT know whether the candidate actually said any of this. If the interviewer's own wording implies a different approach than yours, the interviewer is right — follow them."
    : "what you already told the candidate earlier in this session, oldest first; the latest question may be a follow-up to one of these";

  return [
    `<previous_coach_answers note="${note}">`,
    body,
    "</previous_coach_answers>",
  ].join("\n");
}
