import { useCallback } from "react";
import {
  AUTO_ASSIST_CACHE_TTL_MS,
  AUTO_ASSIST_DEDUPE_WINDOW_MS,
  AUTO_ASSIST_HINT_COOLDOWN_MS,
  AUTO_ASSIST_PREFETCH_CONFIDENCE,
  AUTO_ASSIST_PREFETCH_ENABLED,
  FULL_SESSION_SEGMENT_LIMIT,
} from "./constants";
import {
  detectQuestionCandidate,
  isLikelyDuplicateTranscript,
  transcriptSimilarity,
} from "./interviewLogic";
import { createId, debugLog, safeInvoke } from "./platform";
import { resolveCoachMode, withPrefetchTimeout } from "./speculativePrefetch";
import type { AssistantSuggestion, PrefetchCache, TranscriptSegment } from "./types";
import type { AgentRuntimeActions } from "./useAgentRuntime";
import type { MeetlyState } from "./useMeetlyState";
import type { SessionActions } from "./useSessionActions";

// 问题级去重阈值，与 coachWakePolicy 保持一致。
const QUESTION_DEDUPE_SIMILARITY = 0.82;

export function useAutoAssist(ctx: MeetlyState, session: SessionActions, agent: AgentRuntimeActions) {
  const maybePrefetch = useCallback(
    (segment: TranscriptSegment, priorHistory: TranscriptSegment[]) => {
      if (!AUTO_ASSIST_PREFETCH_ENABLED) return;
      // 只对面试官（系统音频）的问题做推测性预取，不预取用户自己的话。
      if (segment.speaker !== "interviewer") return;

      const candidate = detectQuestionCandidate(segment, priorHistory);
      if (!candidate || candidate.confidence < AUTO_ASSIST_PREFETCH_CONFIDENCE) return;

      const now = Date.now();
      if (ctx.prefetchInFlightRef.current) return;
      if (now - ctx.prefetchLastAtRef.current < AUTO_ASSIST_HINT_COOLDOWN_MS) return;

      const duplicate = ctx.recentQuestionCandidatesRef.current.some(
        (item) =>
          now - item.createdAt <= AUTO_ASSIST_DEDUPE_WINDOW_MS &&
          transcriptSimilarity(item.text, candidate.text) >= QUESTION_DEDUPE_SIMILARITY
      );
      if (duplicate) return;

      const runId = createId("prefetch");
      ctx.prefetchLastAtRef.current = now;
      ctx.recentQuestionCandidatesRef.current = [
        ...ctx.recentQuestionCandidatesRef.current,
        candidate,
      ].slice(-20);
      ctx.setPrefetchStatus("prefetching");

      const mode = resolveCoachMode(ctx.sessionKind, ctx.meetingPerspective);
      // Prefetch fires before the segment reaches the ContextStore, but its
      // answer is what the coach ends up showing — so it must carry the same
      // session anchors as the main path, or a Java interview can get a
      // JavaScript answer. The anchors are prepended to the question text: no
      // extra request, no extra round trip, no latency.
      const anchors = agent.sessionAnchors();
      const question = anchors ? `${anchors}\n\n${candidate.text}` : candidate.text;
      debugLog(
        `[prefetch] anchors=${anchors ? anchors.split("\n").length - 1 : 0} candidate=${candidate.id}`
      );
      const promise = withPrefetchTimeout(
        safeInvoke<AssistantSuggestion>("complete_assistant_with_question", {
          mode,
          question,
          runId,
        })
      )
        .then((suggestion) => {
          const stillCurrent = ctx.prefetchInFlightRef.current?.candidateId === candidate.id;
          if (stillCurrent) {
            ctx.prefetchInFlightRef.current = null;
          }
          if (!suggestion) {
            if (stillCurrent) ctx.setPrefetchStatus("idle");
            return null;
          }

          const cache: PrefetchCache = {
            candidateId: candidate.id,
            questionText: candidate.text,
            suggestion,
            createdAt: now,
            expiresAt: now + AUTO_ASSIST_CACHE_TTL_MS,
            contextPreview: candidate.text.slice(0, 120),
          };
          ctx.prefetchCacheRef.current = cache;
          ctx.setPrefetchStatus("ready");
          debugLog(`[prefetch] ready candidate=${candidate.id} chars=${suggestion.answer.length}`);
          return suggestion;
        })
        .catch((error) => {
          if (ctx.prefetchInFlightRef.current?.candidateId === candidate.id) {
            ctx.prefetchInFlightRef.current = null;
            ctx.setPrefetchStatus("error");
          }
          debugLog(
            `[prefetch] error candidate=${candidate.id} message=${error instanceof Error ? error.message : String(error)}`
          );
          return null;
        });

      ctx.prefetchInFlightRef.current = {
        candidateId: candidate.id,
        questionText: candidate.text,
        confidence: candidate.confidence,
        startedAt: now,
        promise,
      };
      debugLog(
        `[prefetch] start candidate=${candidate.id} confidence=${candidate.confidence.toFixed(2)} chars=${candidate.text.length}`
      );
    },
    [agent, ctx]
  );

  const addTranscriptSegment = useCallback(
    (segment: TranscriptSegment) => {
      const normalizedSegment: TranscriptSegment = {
        ...segment,
        source: segment.source ?? "system",
        speaker: segment.speaker ?? (segment.source === "microphone" ? "user" : "interviewer"),
      };

      const priorHistory = ctx.transcriptHistoryRef.current;
      if (isLikelyDuplicateTranscript(normalizedSegment, priorHistory)) {
        debugLog(
          `[mic] transcript duplicate suppressed id=${normalizedSegment.id} text=${normalizedSegment.text.slice(0, 160).replace(/\n/g, " ")}`
        );
        return;
      }

      const next = [...priorHistory, normalizedSegment]
        .sort((left, right) => left.endMs - right.endMs)
        .slice(-FULL_SESSION_SEGMENT_LIMIT);

      ctx.transcriptHistoryRef.current = next;
      ctx.setLatestTranscript(next[next.length - 1] ?? null);
      ctx.setTranscriptError(null);
      ctx.setTranscriptHistory(next.slice(-20));
      session.updateInterviewSession((current) => ({ ...current, transcript: next }));
      // 先把本段的长程事实抽取进记忆（纯 CPU），再触发推测性预取——
      // 这样预取请求也能带上本场面试的语言/技术栈锚点，最后才进教练唤醒。
      // 教练 transport 会优先复用已就绪/在途的预取结果。
      agent.primeSessionFacts(normalizedSegment);
      maybePrefetch(normalizedSegment, priorHistory);
      agent.pushTranscriptFinal(normalizedSegment);
    },
    [agent, ctx, maybePrefetch, session]
  );

  return {
    addTranscriptSegment,
  };
}

export type AutoAssistActions = ReturnType<typeof useAutoAssist>;
