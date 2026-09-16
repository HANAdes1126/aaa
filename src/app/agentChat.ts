import type { CoachTurn } from "../runtime/agent/contextStore";
import type { AgentChatTurn, AssistantSuggestion, ContextDocument, TranscriptSegment } from "./types";

export function resolveAgentChatMessage(message?: string) {
  return message?.trim() || "需要帮助";
}

export function buildAgentChatContext(input: {
  documents: ContextDocument[];
  goal: string;
  transcript: TranscriptSegment[];
}) {
  const transcript = input.transcript
    .slice(-80)
    .map((segment) => `${segment.speaker === "user" ? "我" : "对方"}：${segment.text.trim()}`)
    .filter((line) => !line.endsWith("："))
    .join("\n");
  const documents = input.documents.map((document) => document.name).join("、");

  return [
    input.goal.trim() ? `会议目标：${input.goal.trim()}` : null,
    documents ? `可用资料：${documents}` : null,
    transcript ? `会议转录：\n${transcript}` : null,
  ].filter(Boolean).join("\n\n").slice(-12_000);
}

/**
 * History for the manual-ask path, merged with what the proactive coach already
 * answered.
 *
 * The two live in separate stores, so before this merge an interviewer question
 * that followed a coach card was answered with no knowledge of it — identical to
 * the failure the coach itself had until coach turns were persisted.
 *
 * Only `coach`-origin turns are merged in: a manual ask is already present in
 * `turns`, and folding it in a second time would send Rust the same exchange
 * twice, which reads as the assistant repeating itself.
 */
export function buildAgentChatHistory(
  turns: AgentChatTurn[],
  coachTurns: CoachTurn[] = [],
  limit = 6
) {
  const fromChat = turns
    .filter((turn) => turn.suggestion)
    .map((turn) => ({
      atMs: turn.createdAt,
      question: turn.question,
      suggestion: turn.suggestion as AssistantSuggestion,
    }));

  // Coach turns are stored as plain text, but Rust serialises whatever sits in
  // `suggestion` straight into the assistant message, so rebuild the shape.
  const fromCoach = coachTurns
    .filter((turn) => turn.origin === "coach" && turn.answer.trim())
    .map((turn) => ({
      atMs: turn.atMs,
      question: turn.question,
      suggestion: {
        answer: turn.answer,
        bullets: [] as string[],
        clarifyingQuestion: null,
        kind: "knowledge",
      } satisfies AssistantSuggestion,
    }));

  const seen = new Set<string>();
  return [...fromChat, ...fromCoach]
    .sort((left, right) => left.atMs - right.atMs)
    .filter((entry) => {
      // A coach turn and a chat turn can describe the same exchange when the
      // user re-asks something the coach just answered.
      const key = `${entry.question.slice(0, 60)}|${entry.suggestion.answer.slice(0, 60)}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(-limit)
    .map((entry) => ({
      question: entry.question,
      suggestion: entry.suggestion,
    }));
}
