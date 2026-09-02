import {
  SPECULATIVE_FLIP_CAP,
  SPECULATIVE_PREFETCH_TIMEOUT_MS,
  SPECULATIVE_REUSE_THRESHOLD,
} from "./constants";
import { transcriptSimilarity } from "./interviewLogic";
import type {
  AssistantMode,
  AssistantSuggestion,
  MeetingPerspective,
  PrefetchCache,
  SessionKind,
} from "./types";

/**
 * 推测性预取的纯逻辑层（参照 Natively Cluely 的 `speculativeSimilarity.ts`）。
 *
 * 场景：面试官说问题时，用转写段提前触发一次生成并缓存；当教练在同一问题
 * 上被唤醒时，若相似度足够高且语义未翻转，直接复用（省掉一轮 LLM 往返）。
 */

/**
 * 与教练 transport 保持一致的模式解析。历史原因：会话类型只有
 * remote / in_person（会议），其余分支保留以兼容未来 interview 会话。
 */
export function resolveCoachMode(
  sessionKind: SessionKind,
  perspective: MeetingPerspective
): AssistantMode {
  if (sessionKind === "remote" || sessionKind === "in_person") {
    return "meeting";
  }
  return perspective === "interviewer" ? "interviewer" : "interview";
}

/**
 * 相似度 + 语义极性守卫。底层用现有 `transcriptSimilarity`（n-gram Jaccard），
 * 但当两侧出现否定/反义词翻转时，把分数压制到复用阈值以下。
 */
export function speculativeSimilarity(left: string, right: string): number {
  const base = transcriptSimilarity(left, right);
  if (base <= 0) return 0;
  if (hasPolarityFlip(left, right)) {
    return Math.min(base, SPECULATIVE_FLIP_CAP);
  }
  return base;
}

/**
 * 判断预取缓存是否可复用于给定的最终问题。
 */
export function matchPrefetchCache(
  cache: PrefetchCache,
  questionText: string,
  now = Date.now()
): AssistantSuggestion | null {
  if (now > cache.expiresAt) return null;
  const similarity = speculativeSimilarity(cache.questionText, questionText);
  if (similarity < SPECULATIVE_REUSE_THRESHOLD) return null;
  return cache.suggestion;
}

/**
 * 给预取请求加兜底超时：超时后 resolve 为 null（而非 reject），
 * 让教练 transport 能在 null 时平滑回退到全新生成。
 */
export function withPrefetchTimeout<T>(
  promise: Promise<T>,
  timeoutMs = SPECULATIVE_PREFETCH_TIMEOUT_MS
): Promise<T | null> {
  return new Promise((resolve) => {
    const timer = window.setTimeout(() => resolve(null), timeoutMs);
    promise.then(
      (value) => {
        window.clearTimeout(timer);
        resolve(value);
      },
      () => {
        window.clearTimeout(timer);
        resolve(null);
      }
    );
  });
}

/**
 * 检测两侧语义极性是否翻转：其一含否定标记而另一不含，或出现反义词替换。
 * 只在高基础相似度下被调用，宁可保守（多判翻转 → 重新生成）也不可复用错答案。
 */
export function hasPolarityFlip(left: string, right: string): boolean {
  const l = left.toLowerCase();
  const r = right.toLowerCase();

  if (hasNegation(l) !== hasNegation(r)) {
    return true;
  }

  return ANTONYM_PAIRS.some(([a, b]) => {
    const aInL = l.includes(a);
    const aInR = r.includes(a);
    const bInL = l.includes(b);
    const bInR = r.includes(b);
    return (aInL && bInR) || (bInL && aInR);
  });
}

// “X 不 X / 有 X 有”类疑问形式虽含否定字面，但不是否定语义，需先排除。
const QUESTION_NEGATION_OVERRIDES = [
  "能不能",
  "可不可以",
  "是不是",
  "有没有",
  "会不会",
  "要不要",
];

const NEGATION_MARKERS = [
  "不是",
  "不能",
  "不会",
  "不用",
  "不要",
  "没有",
  "无需",
  "无须",
  "从未",
  "从不",
  "别做",
  "无法",
  // 常见「了解/认同/支持/熟悉」等动词的否定前缀，捕获“你了解X吗 → 你不了解X吗”类翻转。
  "不了解",
  "不确定",
  "不支持",
  "不认同",
  "不同意",
  "不建议",
  "不推荐",
  "不合适",
  "不熟悉",
  "不使用",
  "不采用",
  "不觉得",
  "不认为",
  "没了解",
  "没做过",
  "没接触",
  "没听说",
  "没用过",
  " no",
  "not",
  "don't",
  "doesn't",
  "didn't",
  "cannot",
  "can't",
  "won't",
  "never",
  "isn't",
  "aren't",
  "wasn't",
  "weren't",
  "without",
];

function hasNegation(text: string): boolean {
  if (QUESTION_NEGATION_OVERRIDES.some((marker) => text.includes(marker))) {
    return false;
  }
  return NEGATION_MARKERS.some((marker) => text.includes(marker));
}

// 保守的成对反义词：只收录语义明确相反的常见面试/会议表述，避免过度触发。
const ANTONYM_PAIRS: Array<[string, string]> = [
  ["优点", "缺点"],
  ["优势", "劣势"],
  ["最好", "最差"],
  ["最好", "最坏"],
  ["好处", "坏处"],
  ["增加", "减少"],
  ["提高", "降低"],
  ["上升", "下降"],
  ["好处", "代价"],
  ["benefit", "drawback"],
  ["advantage", "disadvantage"],
  ["best", "worst"],
  ["good", "bad"],
  ["increase", "decrease"],
  ["pro", "con"],
  ["gain", "loss"],
];
