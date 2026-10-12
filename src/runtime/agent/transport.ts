import { invoke } from "@tauri-apps/api/core";
import { resolveCoachMode } from "../../app/speculativePrefetch";
import type { AssistantSuggestion, CoachToolTrace } from "../../app/types";
import type { AgentPrompt } from "./prompt";

/**
 * There is deliberately no timeout on a coach request.
 *
 * There used to be one: an 85s per-request cap here, raced against a 100s
 * watchdog in `AgentRuntime`. Because the frontend held the tightest budget in
 * the stack, *it* — not the Rust limits — was what actually decided when a long
 * answer died. A reasoning model that takes minutes is doing its job, not
 * stalling; cutting it off mid-answer throws away exactly the part the user was
 * waiting for, and the one-shot retry it triggered doubled the wait instead of
 * shortening it.
 *
 * A run that really hangs is abandoned by the user starting a new conversation,
 * which invalidates the in-flight result through the epoch/generation guards in
 * `AgentRuntime` rather than through a clock.
 */
export type AgentTransport = {
  complete(prompt: AgentPrompt, callbacks?: AgentTransportCallbacks): Promise<AssistantSuggestion>;
};

export type AgentTransportCallbacks = {
  onDelta?: (delta: string) => void;
  onRetry?: (attempt: number, reason: string) => void;
  onToolEnd?: (name: string, isError: boolean) => void;
  onToolStart?: (name: string) => void;
  onToolTrace?: (trace: CoachToolTrace) => void;
};

/**
 * 推测性预取缓存读取入口：给定最终问题文本，返回可直接复用的建议（无则 null）。
 * 由 `useAgentRuntime` 注入，内部负责 TTL 与相似度/极性守卫，命中后消费缓存。
 */
export type PrefetchProvider = {
  lookup(questionText: string): AssistantSuggestion | null;
  inflight(questionText: string): Promise<AssistantSuggestion | null> | null;
};

export function createPiCoachTransport(opts?: { prefetch?: PrefetchProvider }): AgentTransport {
  const prefetch = opts?.prefetch;

  return {
    async complete(prompt, callbacks) {
      // 先尝试复用推测性预取：已完成的结果直接返回；仍在途的 await 同一份
      // 生成（避免另发一版重复请求），均未命中才走全新生成。
      const evidence = prompt.wake.evidence[0] ?? "";
      if (prefetch && evidence) {
        const cached = prefetch.lookup(evidence);
        if (cached) {
          return cached;
        }

        const inflight = prefetch.inflight(evidence);
        if (inflight) {
          const awaited = await inflight;
          if (awaited) {
            return awaited;
          }
        }
      }

      const mode = resolveCoachMode(prompt.snapshot.sessionKind, prompt.snapshot.perspective);
      const suggestion = await invoke<AssistantSuggestion>("complete_assistant_with_question", {
        mode,
        question: prompt.text,
        runId: prompt.wake.id,
      });

      if (!suggestion.answer.trim()) {
        throw new Error("场边教练返回了空内容。");
      }

      return suggestion;
    },
  };
}
