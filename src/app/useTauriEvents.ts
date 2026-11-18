import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { isTauriRuntime } from "./platform";
import type {
  AssistantDeltaPayload,
  AssistantErrorPayload,
  AssistantSuggestion,
  AgentToolTraceEvent,
  AudioLevelChanged,
  CoachToolTrace,
  PartialTranscript,
  TranscriptError,
  TranscriptSegment,
} from "./types";
import type { MeetlyState } from "./useMeetlyState";
import type { AutoAssistActions } from "./useAutoAssist";
import type { SessionActions } from "./useSessionActions";
import type { AgentRuntimeActions } from "./useAgentRuntime";

export function useTauriEvents(
  ctx: MeetlyState,
  autoAssist: AutoAssistActions,
  session: SessionActions,
  agent: AgentRuntimeActions
) {
  // `useMeetlyState` hands back a freshly built object on every render, and
  // `audio_level_changed` re-renders this tree dozens of times a second while a
  // meeting is running. Naming `ctx` (or `autoAssist` / `session` / `agent`) as
  // an effect dependency therefore tore every listener down and re-registered
  // it on each render — and `listen()` only finishes registering asynchronously.
  // Any event that landed inside that gap was dropped, which is why an utterance
  // sometimes produced no transcript at all and saying the exact same thing
  // again worked: the second attempt happened to miss the gap.
  //
  // So the listeners below register exactly once and read the current values
  // through this ref. Keep the dependency arrays empty; putting a render-scoped
  // object back in reopens the hole.
  const latest = useRef({ ctx, autoAssist, session, agent });
  latest.current = { ctx, autoAssist, session, agent };

  useEffect(() => {
    if (!isTauriRuntime()) return;

    let disposed = false;
    let unlisten: (() => void) | undefined;

    void listen<boolean>("island_visibility_changed", (event) => {
      if (!disposed) latest.current.ctx.setIsHidden(!event.payload);
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlisten = nextUnlisten;
    });

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;

    let disposed = false;
    let unlisten: (() => void) | undefined;

    const levels = { system: 0, microphone: 0 };
    void listen<AudioLevelChanged>("audio_level_changed", (event) => {
      if (disposed) return;
      const source = event.payload.source ?? "system";
      levels[source] = event.payload.level;
      latest.current.ctx.setAudioLevel(Math.max(levels.system, levels.microphone));
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlisten = nextUnlisten;
    });

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;

    let disposed = false;
    let unlistenFinal: (() => void) | undefined;
    let unlistenError: (() => void) | undefined;
    let unlistenStarted: (() => void) | undefined;
    let unlistenPartial: (() => void) | undefined;

    void listen<TranscriptSegment>("transcript_final", (event) => {
      if (disposed) return;
      const { ctx, autoAssist } = latest.current;
      ctx.setPendingTranscriptCount((current) => Math.max(0, current - 1));
      ctx.setPartialTranscript(null);
      autoAssist.addTranscriptSegment(event.payload);
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlistenFinal = nextUnlisten;
    });

    void listen<PartialTranscript>("transcript_partial", (event) => {
      if (disposed) return;
      latest.current.ctx.setPartialTranscript(event.payload);
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlistenPartial = nextUnlisten;
    });

    void listen<TranscriptError>("transcript_error", (event) => {
      if (disposed) return;
      const { ctx } = latest.current;
      ctx.setPendingTranscriptCount((current) => Math.max(0, current - 1));
      ctx.setPartialTranscript(null);
      ctx.setTranscriptError(event.payload.message);
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlistenError = nextUnlisten;
    });

    void listen<string>("transcript_started", () => {
      if (disposed) return;
      latest.current.ctx.setPendingTranscriptCount((current) => current + 1);
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlistenStarted = nextUnlisten;
    });

    return () => {
      disposed = true;
      unlistenFinal?.();
      unlistenError?.();
      unlistenStarted?.();
      unlistenPartial?.();
    };
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;

    let disposed = false;
    let unlisten: (() => void) | undefined;

    void listen<AgentToolTraceEvent>("agent_tool_trace", (event) => {
      if (disposed) return;
      const { ctx } = latest.current;
      const payload = event.payload;
      const trace: CoachToolTrace = {
        id: payload.traceId,
        name: payload.name,
        label: payload.label,
        status: payload.status,
        query: payload.query,
        content: payload.content,
        createdAt: payload.createdAt,
        completedAt: payload.completedAt,
      };

      const chatTurnIndex = ctx.agentChatTurnsRef.current.findIndex(
        (turn) => turn.id === payload.runId
      );
      if (chatTurnIndex >= 0) {
        const next = [...ctx.agentChatTurnsRef.current];
        const turn = next[chatTurnIndex];
        next[chatTurnIndex] = {
          ...turn,
          toolTraces: upsertToolTrace(turn.toolTraces, trace),
        };
        ctx.agentChatTurnsRef.current = next;
        ctx.setAgentChatTurns(next);
        return;
      }

      ctx.setCoachDraft((current) => {
        if (!current || current.id !== payload.runId) {
          return current;
        }
        const toolTraces = upsertToolTrace(current.toolTraces, trace);
        ctx.coachToolTracesRef.current = toolTraces;
        return { ...current, toolTraces };
      });
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlisten = nextUnlisten;
    });

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;

    let disposed = false;
    let unlistenDelta: (() => void) | undefined;
    let unlistenDone: (() => void) | undefined;
    let unlistenError: (() => void) | undefined;

    void listen<AssistantDeltaPayload>("assistant_delta", (event) => {
      if (!disposed) latest.current.ctx.setAssistantDraft((current) => current + event.payload.text);
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlistenDelta = nextUnlisten;
    });

    void listen<AssistantSuggestion>("assistant_done", (event) => {
      if (disposed) return;
      const { ctx, agent, session } = latest.current;
      ctx.setAssistantSuggestion(event.payload);
      ctx.setAssistantDraft("");
      ctx.setAssistantError(null);
      ctx.setIsAsking(false);
      const askId = ctx.pendingAskIdRef.current;
      if (askId) {
        agent.recordManualAskFinished(askId, "spoken", "message_committed");
        session.updateInterviewSession((current) => ({
          ...current,
          status: current.endedAt ? "idle" : "listening",
          asks: current.asks.map((ask) =>
            ask.id === askId ? { ...ask, answer: event.payload.answer, error: null } : ask
          ),
        }));
        ctx.pendingAskIdRef.current = null;
      }
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlistenDone = nextUnlisten;
    });

    void listen<AssistantErrorPayload>("assistant_error", (event) => {
      if (disposed) return;
      const { ctx, agent, session } = latest.current;
      ctx.setAssistantError(event.payload.message);
      ctx.setAssistantDraft("");
      ctx.setIsAsking(false);
      const askId = ctx.pendingAskIdRef.current;
      if (askId) {
        agent.recordManualAskFinished(askId, "failed", "run_failed");
        session.updateInterviewSession((current) => ({
          ...current,
          status: current.endedAt ? "idle" : "listening",
          asks: current.asks.map((ask) => ask.id === askId ? { ...ask, error: event.payload.message } : ask),
        }));
        ctx.pendingAskIdRef.current = null;
      }
    }).then((nextUnlisten) => {
      if (disposed) nextUnlisten();
      else unlistenError = nextUnlisten;
    });

    return () => {
      disposed = true;
      unlistenDelta?.();
      unlistenDone?.();
      unlistenError?.();
    };
  }, []);
}

function upsertToolTrace(current: CoachToolTrace[], trace: CoachToolTrace) {
  const index = current.findIndex((item) => item.id === trace.id);
  if (index < 0) return [...current, trace];

  const next = [...current];
  next[index] = {
    ...next[index],
    ...trace,
    createdAt: next[index].createdAt,
  };
  return next;
}
