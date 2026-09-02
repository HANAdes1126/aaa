import { invoke } from "@tauri-apps/api/core";
import { resolveCoachMode } from "../../app/speculativePrefetch";
import type { AssistantSuggestion, CoachToolTrace } from "../../app/types";
import type { AgentPrompt } from "./prompt";

const COACH_REQUEST_TIMEOUT_MS = 10_000;

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
      const suggestion = await runWithOneTimeoutRetry(
        () => invoke<AssistantSuggestion>("complete_assistant_with_question", {
          mode,
          question: prompt.text,
          runId: prompt.wake.id,
        }),
        COACH_REQUEST_TIMEOUT_MS,
        () => callbacks?.onRetry?.(2, "request_timeout")
      );

      if (!suggestion.answer.trim()) {
        throw new Error("场边教练返回了空内容。");
      }

      return suggestion;
    },
  };
}

export async function runWithOneTimeoutRetry<T>(
  operation: () => Promise<T>,
  timeoutMs: number,
  onRetry?: () => void
): Promise<T> {
  try {
    return await withTimeout(operation(), timeoutMs);
  } catch (error) {
    if (!(error instanceof CoachRequestTimeoutError)) throw error;
    onRetry?.();
    try {
      return await withTimeout(operation(), timeoutMs);
    } catch (retryError) {
      if (retryError instanceof CoachRequestTimeoutError) {
        throw new Error(`场边教练连续两次超过 ${Math.round(timeoutMs / 1000)} 秒，已结束本轮请求。`);
      }
      throw retryError;
    }
  }
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number) {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  const timeout = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => reject(new CoachRequestTimeoutError()), timeoutMs);
  });

  return Promise.race([promise, timeout]).finally(() => {
    if (timeoutId !== null) clearTimeout(timeoutId);
  });
}

class CoachRequestTimeoutError extends Error {
  constructor() {
    super("COACH_REQUEST_TIMEOUT");
    this.name = "CoachRequestTimeoutError";
  }
}
