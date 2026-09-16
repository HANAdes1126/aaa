import { emit, listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useRef } from "react";
import {
  AgentRuntime,
  CoachEventJournal,
  ContextStore,
  createEnterWake,
  createSessionStartWake,
  createPiCoachTransport,
  detectSttWake,
  type AgentRuntimeCallbacks,
  type PrefetchProvider,
  type WakeEvent,
} from "../runtime/agent";
import { debugLog, isTauriRuntime } from "./platform";
import { formatSuggestionText } from "./coachMessageFormat";
import { matchPrefetchCache, speculativeSimilarity } from "./speculativePrefetch";
import {
  COACH_ANSWERED_EVENT,
  COACH_ANSWER_BRIDGE_CHARS,
  SESSION_MEMORY_CLEARED_EVENT,
  SPECULATIVE_REUSE_THRESHOLD,
  VOICE_ASK_ANSWERED_EVENT,
  VOICE_ASK_ANSWER_BRIDGE_CHARS,
} from "./constants";
import type {
  AudioSource,
  CoachMessage,
  CoachToolTrace,
  CoachTrigger,
  SessionKind,
  TranscriptSegment,
} from "./types";
import type { MeetlyState } from "./useMeetlyState";

export function useAgentRuntime(ctx: MeetlyState) {
  const contextRef = useRef<ContextStore | null>(null);
  const journalRef = useRef<CoachEventJournal | null>(null);
  const runtimeRef = useRef<AgentRuntime | null>(null);
  const currentSessionIdRef = useRef<string | null>(null);
  const sessionStartEventIdRef = useRef<string | null>(null);
  const manualAskEventIdsRef = useRef(new Map<string, string>());

  if (!contextRef.current) {
    contextRef.current = new ContextStore();
  }

  /**
   * Wipes coach memory and tells the voice-overlay window to drop its mirrored
   * copy. Always go through here instead of calling `contextRef.current.clear()`
   * directly: the overlay is a separate webview holding its own copy of the
   * coach answers, so a bare clear() leaves the previous interview alive there
   * and the next Fn-held question gets answered with stale history.
   */
  const resetCoachMemory = useCallback((sessionId: string | null) => {
    contextRef.current?.clear();
    contextRef.current?.setSessionId(sessionId);
    if (!isTauriRuntime()) return;
    void emit(SESSION_MEMORY_CLEARED_EVENT, {}).catch(() => undefined);
  }, []);

  if (!journalRef.current) {
    journalRef.current = new CoachEventJournal();
  }

  if (!runtimeRef.current) {
    runtimeRef.current = new AgentRuntime(
      contextRef.current,
      createPiCoachTransport({ prefetch: buildPrefetchProvider(ctx) }),
      buildCallbacks(ctx, journalRef.current)
    );
  }

  runtimeRef.current.setCallbacks(buildCallbacks(ctx, journalRef.current));

  useEffect(() => {
    const sessionId = ctx.interviewSession?.id ?? null;
    if (currentSessionIdRef.current === sessionId) {
      return;
    }

    currentSessionIdRef.current = sessionId;
    resetCoachMemory(sessionId);
    setCoachActivity(ctx, null);
    debugLog(`[agent] context reset session=${sessionId ?? "none"}`);
  }, [ctx.interviewSession?.id]);

  useEffect(() => {
    contextRef.current?.setPerspective(ctx.meetingPerspective);
  }, [ctx.meetingPerspective]);

  useEffect(() => {
    contextRef.current?.setDocuments(ctx.contextDocuments);
  }, [ctx.contextDocuments]);

  useEffect(() => {
    contextRef.current?.setSessionConfig({
      kind: ctx.sessionKind,
      audioSource: ctx.audioSource,
      goal: ctx.meetingGoal,
    });
  }, [ctx.audioSource, ctx.meetingGoal, ctx.sessionKind]);

  const primeSessionFacts = useCallback((segment: TranscriptSegment) => {
    contextRef.current?.primeFacts(segment);
  }, []);

  const sessionAnchors = useCallback(() => contextRef.current?.anchors() ?? "", []);

  const pushTranscriptFinal = useCallback((segment: TranscriptSegment) => {
    contextRef.current?.pushTranscript(segment);

    const sessionId = currentSessionIdRef.current;
    const observed = sessionId
      ? journalRef.current?.appendEvent({
          sessionId,
          type: "transcript.finalized",
          source: segment.source ?? "system",
          segmentId: segment.id,
          speaker: toJournalSpeaker(segment.speaker),
          evidencePreview: segment.text,
          details: {
            startMs: segment.startMs,
            endMs: segment.endMs,
          },
        })
      : null;

    const detectedWake = detectSttWake(segment, ctx.sessionKind, ctx.meetingPerspective);
    const wake = detectedWake
      ? {
          ...detectedWake,
          sessionId: sessionId ?? undefined,
          evidenceEventIds: observed ? [observed.id] : [],
        }
      : null;
    if (!wake) {
      debugLog(`[agent] stt wake skipped segment=${segment.id}`);
      return;
    }

    debugLog(`[agent] stt wake segment=${segment.id} reason=${wake.reason}`);
    runtimeRef.current?.wake(wake);
  }, [ctx.sessionKind, ctx.meetingPerspective]);

  /**
   * Aborts any coach run in flight. Used by "new conversation": the request
   * itself keeps running in the background (a Tauri invoke can't be revoked),
   * but its result is discarded and the runtime is free immediately.
   */
  const cancelCoach = useCallback((reason: string) => {
    runtimeRef.current?.cancel(reason);
  }, []);

  const wakeEnter = useCallback(() => {
    const sessionId = currentSessionIdRef.current;
    const wake = { ...createEnterWake(), sessionId: sessionId ?? undefined };
    debugLog(`[agent] enter wake reason=${wake.reason}`);
    runtimeRef.current?.wake(wake);
  }, []);

  const wakeSessionStart = useCallback((sessionId: string, hasDocuments: boolean) => {
    if (currentSessionIdRef.current !== sessionId) {
      currentSessionIdRef.current = sessionId;
      resetCoachMemory(sessionId);
    }
    // With no uploaded documents there is nothing for the coach to work from yet
    // — the interviewer hasn't spoken. Waking here only produced a placeholder
    // card ("I don't see a question in the transcript yet") right at the opening.
    if (!hasDocuments) {
      debugLog(`[agent] session wake skipped session=${sessionId} reason=no_context_yet`);
      return;
    }

    const wake = createSessionStartWake(hasDocuments);
    wake.sessionId = sessionId;
    wake.evidenceEventIds = sessionStartEventIdRef.current ? [sessionStartEventIdRef.current] : [];
    debugLog(`[agent] session wake session=${sessionId} reason=${wake.reason}`);
    runtimeRef.current?.wake(wake);
  }, []);

  const recordSessionStarted = useCallback((input: {
    sessionId: string;
    sessionKind: SessionKind;
    audioSource: AudioSource;
    hasDocuments: boolean;
  }) => {
    currentSessionIdRef.current = input.sessionId;
    resetCoachMemory(input.sessionId);
    journalRef.current?.clear();
    const event = journalRef.current?.appendEvent({
      sessionId: input.sessionId,
      type: "session.started",
      source: "runtime",
      details: {
        sessionKind: input.sessionKind,
        audioSource: input.audioSource,
        hasDocuments: input.hasDocuments,
      },
    });
    sessionStartEventIdRef.current = event?.id ?? null;
  }, []);

  const recordCaptureStarted = useCallback((sessionId: string, source: AudioSource) => {
    journalRef.current?.appendEvent({
      sessionId,
      type: "audio.capture.started",
      source,
      details: { source },
    });
  }, []);

  const recordCaptureFailed = useCallback((sessionId: string, source: AudioSource) => {
    journalRef.current?.appendEvent({
      sessionId,
      type: "audio.capture.failed",
      source,
      details: { source, reason: "capture_start_failed" },
    });
  }, []);

  const recordSessionEnded = useCallback((sessionId: string) => {
    journalRef.current?.appendEvent({
      sessionId,
      type: "session.ended",
      source: "runtime",
    });
  }, []);

  const recordManualAskStarted = useCallback((askId: string) => {
    runtimeRef.current?.beginManualAsk();
    setCoachActivity(ctx, null);
    ctx.setCoachDraft(null);
    ctx.setIsCoachThinking(false);
    ctx.coachToolTracesRef.current = [];
    ctx.coachInFlightRef.current = false;
    const sessionId = currentSessionIdRef.current;
    if (!sessionId) return;
    const event = journalRef.current?.appendEvent({
      sessionId,
      type: "user.manual_ask",
      source: "ui",
    });
    if (!event) return;
    manualAskEventIdsRef.current.set(askId, event.id);
    journalRef.current?.appendTransition({
      sessionId,
      status: "running",
      reason: "user_manual_ask",
      wakeId: askId,
      runId: askId,
      eventIds: [event.id],
    });
  }, []);

  const recordManualAskFinished = useCallback((
    askId: string,
    status: "spoken" | "failed",
    reason: string
  ) => {
    runtimeRef.current?.finishManualAsk();
    const sessionId = currentSessionIdRef.current;
    const eventId = manualAskEventIdsRef.current.get(askId);
    if (!sessionId || !eventId) return;
    journalRef.current?.appendTransition({
      sessionId,
      status,
      reason,
      wakeId: askId,
      runId: askId,
      eventIds: [eventId],
    });
    manualAskEventIdsRef.current.delete(askId);
  }, []);

  /**
   * Folds a manual-ask answer into coach memory.
   *
   * The manual path already carries its own `turns` to Rust, but the coach reads
   * a different store — until now the two memories never met. Ask one question
   * by hand, then let the interviewer follow up on it, and the coach answered as
   * if nothing had ever been said about the topic.
   */
  const recordManualAnswer = useCallback((question: string, answer: string) => {
    const text = (answer || "").trim();
    if (!text) return;
    contextRef.current?.pushCoachTurn({
      question,
      answer: text,
      atMs: Date.now(),
      origin: "manual",
    });
    debugLog(`[agent] manual answer folded into coach memory chars=${text.length}`);
  }, []);

  /**
   * The reverse direction: everything the coach (or an earlier manual ask) has
   * already answered, for the manual-ask path to send along as `turns`. Without
   * it, pressing Fn after a coach card starts a brand new conversation.
   */
  const coachMemoryForAsk = useCallback(() => contextRef.current?.coachTurnSnapshot() ?? [], []);

  // Voice Ask runs inside the voice-overlay window — a separate webview with its
  // own ContextStore instance, so it can't share memory by reference. Bridge the
  // answers back over the event bus instead. One-directional on purpose: the
  // overlay keeps its own conversation state and does not need the coach's.
  useEffect(() => {
    if (!isTauriRuntime()) return;

    let disposed = false;
    let unlisten: (() => void) | null = null;

    listen<{ question?: string; answer?: string }>(VOICE_ASK_ANSWERED_EVENT, (event) => {
      const payload = event.payload;
      const answer = (payload?.answer ?? "").trim();
      if (!answer) return;
      contextRef.current?.pushCoachTurn({
        question: payload?.question ?? "",
        answer: answer.slice(0, VOICE_ASK_ANSWER_BRIDGE_CHARS),
        atMs: Date.now(),
        origin: "manual",
      });
      debugLog(`[agent] voice-ask answer bridged chars=${answer.length}`);
    })
      .then((fn) => {
        if (disposed) {
          fn();
        } else {
          unlisten = fn;
        }
      })
      .catch(() => undefined);

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  return {
    cancelCoach,
    coachMemoryForAsk,
    recordManualAnswer,
    primeSessionFacts,
    pushTranscriptFinal,
    recordCaptureFailed,
    recordCaptureStarted,
    recordManualAskFinished,
    recordManualAskStarted,
    recordSessionEnded,
    recordSessionStarted,
    sessionAnchors,
    wakeEnter,
    wakeSessionStart,
  };
}

function buildPrefetchProvider(ctx: MeetlyState): PrefetchProvider {
  return {
    lookup(questionText: string) {
      const cache = ctx.prefetchCacheRef.current;
      if (!cache) return null;

      const suggestion = matchPrefetchCache(cache, questionText);
      if (!suggestion) return null;

      // 命中即消费，避免同一份缓存被后续不同问题误复用。
      ctx.prefetchCacheRef.current = null;
      ctx.setPrefetchStatus("idle");
      debugLog(
        `[prefetch] cache hit candidate=${cache.candidateId} chars=${suggestion.answer.length}`
      );
      return suggestion;
    },
    inflight(questionText: string) {
      const inFlight = ctx.prefetchInFlightRef.current;
      if (!inFlight) return null;
      if (speculativeSimilarity(inFlight.questionText, questionText) < SPECULATIVE_REUSE_THRESHOLD) {
        return null;
      }
      debugLog(`[prefetch] reusing in-flight candidate=${inFlight.candidateId}`);
      return inFlight.promise;
    },
  };
}

function buildCallbacks(ctx: MeetlyState, journal: CoachEventJournal): AgentRuntimeCallbacks {
  return {
    onDelta: (delta, wake) => {
      ctx.setCoachDraft((current) => {
        const draft = current ?? buildCoachMessage(wake, "");
        return {
          ...draft,
          text: `${draft.text}${delta}`,
        };
      });
    },
    onWakeStart: (wake) => {
      appendWakeTransition(journal, wake, "running", wake.reason);
      ctx.coachToolTracesRef.current = [];
      const draft = buildCoachMessage(wake, "");
      ctx.coachInFlightRef.current = true;
      ctx.setIsCoachThinking(true);
      ctx.setCoachDraft(draft);
      setCoachActivity(ctx, {
        phase: "thinking",
        label: "思考中",
        detail: wake.evidence[0] ? `参考：${wake.evidence[0].slice(0, 80)}` : undefined,
      });
      debugLog(`[agent] coach start wake=${wake.kind} reason=${wake.reason}`);
    },
    onWakeSkipped: (wake, reason) => {
      appendWakeTransition(journal, wake, "ignored", reason);
      debugLog(`[agent] wake skipped wake=${wake.kind} reason=${reason}`);
    },
    onCancelled: (reason) => {
      // The runtime already dropped the run; clear everything the UI was
      // showing for it so the panel is usable again immediately.
      ctx.setCoachDraft(null);
      ctx.setIsCoachThinking(false);
      ctx.coachInFlightRef.current = false;
      ctx.coachToolTracesRef.current = [];
      setCoachActivity(ctx, null);
      debugLog(`[agent] coach cancelled reason=${reason}`);
    },
    onRetry: (attempt, reason, wake) => {
      appendWakeTransition(journal, wake, "running", reason, { attempt });
      ctx.setCoachDraft(buildCoachMessage(wake, ""));
      setCoachActivity(ctx, {
        phase: "thinking",
        label: "重新尝试",
        detail: "上一请求超过 10 秒，已丢弃并重发",
      });
      debugLog(`[agent] coach retry wake=${wake.kind} attempt=${attempt} reason=${reason}`);
    },
    onMessage: (suggestion, wake) => {
      appendWakeTransition(journal, wake, "spoken", "message_committed", {
        answerChars: suggestion.answer.length,
      });
      const message = buildCoachMessage(
        wake,
        formatSuggestionText(suggestion),
        ctx.coachToolTracesRef.current
      );
      const next = appendInQuestionOrder(ctx.coachMessagesRef.current, message);
      ctx.coachMessagesRef.current = next;
      ctx.setCoachMessages(next);
      ctx.setCoachDraft(null);
      ctx.coachToolTracesRef.current = [];
      ctx.setIsCoachThinking(false);
      ctx.coachInFlightRef.current = false;
      setCoachActivity(ctx, {
        phase: "speaking",
        label: "说话中",
      }, 1_200);
      debugLog(`[agent] coach message wake=${wake.kind} chars=${suggestion.answer.length}`);
      // Publish to the voice-overlay window, which is a separate webview and
      // therefore blind to this store. It sends these along as history, so a
      // Fn-held follow-up continues the interview instead of restarting it.
      void emit(COACH_ANSWERED_EVENT, {
        question: wake.evidence[0] ?? "",
        answer: suggestion.answer.slice(0, COACH_ANSWER_BRIDGE_CHARS),
      }).catch(() => undefined);
    },
    onError: (message, wake) => {
      appendWakeTransition(journal, wake, "failed", "run_failed");
      const errorMessage = buildCoachMessage(wake, `教练生成失败：${message}`, ctx.coachToolTracesRef.current);
      const next = appendInQuestionOrder(ctx.coachMessagesRef.current, errorMessage);
      ctx.coachMessagesRef.current = next;
      ctx.setCoachMessages(next);
      ctx.setCoachDraft(null);
      ctx.coachToolTracesRef.current = [];
      ctx.setIsCoachThinking(false);
      ctx.coachInFlightRef.current = false;
      setCoachActivity(ctx, null);
      debugLog(`[agent] error wake=${wake.kind} message=${message}`);
    },
    onToolEnd: (name, isError) => {
      setCoachActivity(ctx, {
        phase: "tool",
        label: isError ? `${toolLabel(name)}失败` : `${toolLabel(name)}完成`,
      }, 1_200);
      debugLog(`[agent] coach tool end name=${name} error=${isError}`);
    },
    onToolStart: (name) => {
      setCoachActivity(ctx, {
        phase: "tool",
        label: toolLabel(name),
      });
      debugLog(`[agent] coach tool start name=${name}`);
    },
    onToolTrace: (trace) => {
      ctx.coachToolTracesRef.current = upsertToolTrace(ctx.coachToolTracesRef.current, trace);
      ctx.setCoachDraft((current) => {
        if (!current) {
          return current;
        }
        return {
          ...current,
          toolTraces: ctx.coachToolTracesRef.current,
        };
      });
      debugLog(`[agent] coach tool trace name=${trace.name} status=${trace.status}`);
    },
  };
}

function appendWakeTransition(
  journal: CoachEventJournal,
  wake: WakeEvent,
  status: "ignored" | "running" | "spoken" | "failed",
  reason: string,
  details?: Record<string, string | number | boolean | null>
) {
  if (!wake.sessionId) return;
  journal.appendTransition({
    sessionId: wake.sessionId,
    status,
    reason,
    wakeId: wake.id,
    runId: wake.id,
    eventIds: wake.evidenceEventIds,
    details,
  });
}

function toJournalSpeaker(speaker: TranscriptSegment["speaker"]) {
  if (speaker === "user") return "user" as const;
  if (speaker === "interviewer") return "other" as const;
  return "unknown" as const;
}

function setCoachActivity(
  ctx: MeetlyState,
  activity: MeetlyState["coachActivity"],
  clearAfterMs?: number
) {
  if (ctx.coachActivityClearTimerRef.current !== null) {
    window.clearTimeout(ctx.coachActivityClearTimerRef.current);
    ctx.coachActivityClearTimerRef.current = null;
  }

  ctx.setCoachActivity(activity);

  if (activity && clearAfterMs) {
    ctx.coachActivityClearTimerRef.current = window.setTimeout(() => {
      ctx.setCoachActivity(null);
      ctx.coachActivityClearTimerRef.current = null;
    }, clearAfterMs);
  }
}

function buildCoachMessage(wake: WakeEvent, text: string, toolTraces: CoachToolTrace[] = []): CoachMessage {
  return {
    id: wake.id,
    // When the question was asked, not when the answer finished. The workspace
    // timeline merges these cards with manual asks by `createdAt`, so dating a
    // card by its completion time sorts a slow answer after questions that came
    // later — the coach looks like it is answering backwards.
    createdAt: wake.createdAtMs,
    trigger: toCoachTrigger(wake),
    text,
    contextPreview: wake.evidence.join("\n").slice(0, 260),
    toolTraces,
  };
}

/// Adds a finished card and keeps the panel in the order the questions were
/// asked. Answers normally arrive in that order already, but a question can be
/// held back (cooldown, retry) and land after a later one.
function appendInQuestionOrder(current: CoachMessage[], message: CoachMessage): CoachMessage[] {
  return [...current, message]
    .sort((left, right) => left.createdAt - right.createdAt)
    .slice(-8);
}

function toCoachTrigger(wake: WakeEvent): CoachTrigger {
  if (wake.kind === "enter") return "agent_enter";
  if (wake.kind === "session_start") return "session_started";
  if (wake.kind === "stt_signal" && wake.reason === "external_context_needed") return "research_signal";
  if (wake.kind === "stt_signal") return "context_signal";
  return "agent_stt_question";
}

function toolLabel(name: string) {
  if (name === "read_file") return "读取资料";
  if (name === "web_search" || name === "web_fetch") return "获取网页信息";
  return "使用工具";
}

function upsertToolTrace(current: CoachToolTrace[], trace: CoachToolTrace) {
  const index = current.findIndex((item) => item.id === trace.id);
  if (index < 0) {
    return [...current, trace];
  }

  const next = [...current];
  next[index] = {
    ...next[index],
    ...trace,
    createdAt: next[index].createdAt,
  };
  return next;
}

export type AgentRuntimeActions = ReturnType<typeof useAgentRuntime>;
