import type { AssistantSuggestion, CoachToolTrace } from "../../app/types";
import { ContextStore } from "./contextStore";
import { buildAgentPrompt } from "./prompt";
import type { AgentTransport } from "./transport";
import type { WakeEvent } from "./wake";

const AGENT_CONTEXT_WINDOW_MS = 120_000;
const STT_WAKE_COOLDOWN_MS = 10_000;
const RECENT_EVIDENCE_TTL_MS = 60_000;
/// Last-resort watchdog for one coach run.
///
/// The transport has its own per-request timeout, and Rust now caps every LLM
/// call — but neither helps if a promise simply never settles (a prefetch
/// promise parked in state, a transport that swallows a rejection). Without
/// this, `inFlight` stays true forever, every later question is dropped as
/// "already running", and the panel looks dead with no way to recover.
///
/// Must stay ABOVE the transport budget (85s), otherwise it fires first and the
/// user gets this generic "watchdog" failure instead of the real reason. It is
/// a deadlock breaker, not a latency policy.
const AGENT_RUN_HARD_TIMEOUT_MS = 100_000;

export type AgentRuntimeCallbacks = {
  onDelta?: (delta: string, wake: WakeEvent) => void;
  onRetry?: (attempt: number, reason: string, wake: WakeEvent) => void;
  onWakeStart?: (wake: WakeEvent) => void;
  onWakeSkipped?: (wake: WakeEvent, reason: string) => void;
  onMessage: (suggestion: AssistantSuggestion, wake: WakeEvent) => void;
  onError: (message: string, wake: WakeEvent) => void;
  /** Fired by `cancel()`. The runtime has already released itself; the UI must
   * drop any "thinking" affordance it is still showing. */
  onCancelled?: (reason: string) => void;
  onToolEnd?: (name: string, isError: boolean, wake: WakeEvent) => void;
  onToolStart?: (name: string, wake: WakeEvent) => void;
  onToolTrace?: (trace: CoachToolTrace, wake: WakeEvent) => void;
};

export class AgentRuntime {
  private callbacks: AgentRuntimeCallbacks;
  private inFlight = false;
  private queue: WakeEvent[] = [];
  private lastCoachMessageAtMs = 0;
  private pendingSttWake: WakeEvent | null = null;
  private pendingSttTimer: number | null = null;
  private recentEvidence: Array<{ text: string; handledAtMs: number }> = [];
  private interactionEpoch = 0;
  private manualAskActive = false;
  /** Distinguishes the current `drain()` from an abandoned one. `inFlight`
   * alone can't: a cancelled run may still be awaiting a promise that never
   * settles, and its `finally` would otherwise clear the flag out from under
   * the run that replaced it. */
  private drainGeneration = 0;

  constructor(
    private context: ContextStore,
    private transport: AgentTransport,
    callbacks: AgentRuntimeCallbacks,
    /** Overridable so tests can exercise the watchdog without waiting 30s. */
    private runTimeoutMs: number = AGENT_RUN_HARD_TIMEOUT_MS
  ) {
    this.callbacks = callbacks;
  }

  setCallbacks(callbacks: AgentRuntimeCallbacks) {
    this.callbacks = callbacks;
  }

  beginManualAsk() {
    this.manualAskActive = true;
    this.interactionEpoch += 1;
    this.queue = [];
    this.pendingSttWake = null;
    if (this.pendingSttTimer !== null) {
      window.clearTimeout(this.pendingSttTimer);
      this.pendingSttTimer = null;
    }
  }

  finishManualAsk() {
    this.manualAskActive = false;
  }

  /**
   * Abandons everything in flight and forgets the current conversation.
   *
   * A Tauri invoke has no cancellation token, so the outstanding request keeps
   * running — but it is invalidated two ways: the epoch bump makes its result
   * ineligible for delivery, and the generation bump stops its `drain()` from
   * touching `inFlight` when it eventually settles. `inFlight` is released
   * *here*, not then, because "the user clicked away" must free the runtime
   * immediately; waiting for a stalled request (up to `runTimeoutMs`) would
   * leave it busy long after the user moved on.
   */
  cancel(reason: string) {
    this.interactionEpoch += 1;
    this.drainGeneration += 1;
    this.queue = [];
    this.pendingSttWake = null;
    if (this.pendingSttTimer !== null) {
      window.clearTimeout(this.pendingSttTimer);
      this.pendingSttTimer = null;
    }
    // A new conversation must not inherit the previous one's cooldown or its
    // "already answered this" memory.
    this.recentEvidence = [];
    this.lastCoachMessageAtMs = 0;
    this.manualAskActive = false;
    this.inFlight = false;
    this.callbacks.onCancelled?.(reason);
  }

  wake(event: WakeEvent) {
    if (this.manualAskActive) {
      this.callbacks.onWakeSkipped?.(event, "manual_ask_active");
      return;
    }

    if (isTranscriptWake(event) && (this.inFlight || this.queue.length > 0)) {
      this.pendingSttWake = event;
      this.callbacks.onWakeSkipped?.(event, "stt_coalesced_while_in_flight");
      return;
    }

    const skipReason = this.getSkipReason(event);
    if (skipReason) {
      // A cooldown hit is a timing accident, not a judgement that the question
      // is unworthy. Dropping it silently is indistinguishable from "the coach
      // never answered", so park transcript wakes and replay them once the
      // cooldown expires. `stt_duplicate` is deliberately NOT parked: that one
      // means we already answered this exact question.
      if (isTranscriptWake(event) && skipReason === "stt_cooldown") {
        this.pendingSttWake = event;
        this.schedulePendingSttWake();
      }
      this.callbacks.onWakeSkipped?.(event, skipReason);
      return;
    }

    this.queue.push(event);
    this.queue.sort((left, right) => right.priority - left.priority);
    void this.drain();
  }

  private async drain() {
    if (this.inFlight) return;
    this.inFlight = true;
    const generation = ++this.drainGeneration;

    try {
      while (generation === this.drainGeneration && this.queue.length > 0) {
        const wake = this.queue.shift()!;
        const epoch = this.interactionEpoch;
        this.callbacks.onWakeStart?.(wake);

        try {
          const snapshot = this.context.snapshot(AGENT_CONTEXT_WINDOW_MS);
          const prompt = buildAgentPrompt(wake, snapshot);
          const suggestion = await withHardTimeout(
            this.transport.complete(prompt, {
              onDelta: (delta) => {
                if (epoch === this.interactionEpoch) this.callbacks.onDelta?.(delta, wake);
              },
              onRetry: (attempt, reason) => {
                if (epoch === this.interactionEpoch) this.callbacks.onRetry?.(attempt, reason, wake);
              },
              onToolEnd: (name, isError) => {
                if (epoch === this.interactionEpoch) this.callbacks.onToolEnd?.(name, isError, wake);
              },
              onToolStart: (name) => {
                if (epoch === this.interactionEpoch) this.callbacks.onToolStart?.(name, wake);
              },
              onToolTrace: (trace) => {
                if (epoch === this.interactionEpoch) this.callbacks.onToolTrace?.(trace, wake);
              },
            }),
            this.runTimeoutMs
          );
          if (generation !== this.drainGeneration) {
            this.callbacks.onWakeSkipped?.(wake, "cancelled");
            continue;
          }
          if (epoch !== this.interactionEpoch) {
            this.callbacks.onWakeSkipped?.(wake, "superseded_by_manual_ask");
            continue;
          }
          // Remember what we just told the candidate *before* delivering it, so
          // a follow-up arriving a few seconds later already sees it.
          this.context.pushCoachTurn({
            question: wake.evidence[0] ?? "",
            answer: suggestion.answer,
            atMs: Date.now(),
            origin: "coach",
          });
          this.markHandled(wake);
          this.callbacks.onMessage(suggestion, wake);
        } catch (error) {
          if (generation !== this.drainGeneration) {
            this.callbacks.onWakeSkipped?.(wake, "cancelled");
            continue;
          }
          if (epoch !== this.interactionEpoch) {
            this.callbacks.onWakeSkipped?.(wake, "superseded_by_manual_ask");
            continue;
          }
          this.callbacks.onError(error instanceof Error ? error.message : String(error), wake);
        }
      }
    } finally {
      // Only the live drain owns `inFlight`; an abandoned one must not clear it
      // out from under the run that replaced it.
      if (generation === this.drainGeneration) {
        this.inFlight = false;
        if (!this.manualAskActive) this.schedulePendingSttWake();
      }
    }
  }

  private getSkipReason(event: WakeEvent) {
    if (event.kind === "enter" || event.kind === "session_start") {
      return null;
    }

    const now = Date.now();
    this.recentEvidence = this.recentEvidence.filter(
      (item) => now - item.handledAtMs <= RECENT_EVIDENCE_TTL_MS
    );

    if (this.lastCoachMessageAtMs && now - this.lastCoachMessageAtMs < STT_WAKE_COOLDOWN_MS) {
      return "stt_cooldown";
    }

    const evidence = normalizeEvidence(event.evidence[0] ?? "");
    if (evidence && this.recentEvidence.some((item) => isSimilarEvidence(item.text, evidence))) {
      return "stt_duplicate";
    }

    return null;
  }

  private markHandled(event: WakeEvent) {
    this.lastCoachMessageAtMs = Date.now();
    const evidence = normalizeEvidence(event.evidence[0] ?? "");
    if (evidence) {
      this.recentEvidence.push({ text: evidence, handledAtMs: this.lastCoachMessageAtMs });
    }
  }

  private schedulePendingSttWake() {
    if (!this.pendingSttWake || this.inFlight || this.queue.length > 0 || this.pendingSttTimer !== null) {
      return;
    }

    const delayMs = Math.max(
      0,
      STT_WAKE_COOLDOWN_MS - (Date.now() - this.lastCoachMessageAtMs)
    );

    this.pendingSttTimer = window.setTimeout(() => {
      this.pendingSttTimer = null;
      const pending = this.pendingSttWake;
      this.pendingSttWake = null;
      if (!pending) return;

      const skipReason = this.getSkipReason(pending);
      if (skipReason) {
        this.callbacks.onWakeSkipped?.(pending, skipReason);
        return;
      }

      this.queue.push(pending);
      void this.drain();
    }, delayMs);
  }
}

function isTranscriptWake(event: WakeEvent) {
  return event.kind === "stt_question" || event.kind === "stt_signal";
}

function normalizeEvidence(text: string) {
  return text.toLowerCase().replace(/\s+/g, "").replace(/[，。！？!?.,;；:："'“”‘’]/g, "");
}

function isSimilarEvidence(left: string, right: string) {
  if (!left || !right) return false;
  if (left === right) return true;
  if (left.length >= 8 && right.length >= 8 && (left.includes(right) || right.includes(left))) {
    return true;
  }
  return false;
}

function withHardTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const guard = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () =>
        reject(
          new Error(
            `本轮生成超过 ${Math.round(timeoutMs / 1000)} 秒仍未完成，已放弃；可以直接再问一次。`
          )
        ),
      timeoutMs
    );
  });

  return Promise.race([promise, guard]).finally(() => {
    if (timer !== null) clearTimeout(timer);
  });
}
