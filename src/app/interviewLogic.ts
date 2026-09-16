import {
  AUTO_ASSIST_MIN_CONFIDENCE,
  FULL_SESSION_SEGMENT_LIMIT,
  TRANSCRIPT_DEDUPE_WINDOW_MS,
} from "./constants";
import { summarizeContextDocuments } from "./contextDocuments";
import { debugLog } from "./platform";
import type {
  AskContext,
  CoachTrigger,
  ContextDocument,
  LatestQuestionCandidate,
  MeetingPerspective,
  QuestionCandidate,
  QuestionKind,
  TranscriptSegment,
} from "./types";

export function normalizeTranscriptText(text: string) {
  return text
    .toLowerCase()
    .replace(/[\s，。！？,.!?;；:：'"“”‘’、（）()[\]{}<>《》-]+/g, "")
    .trim();
}

export function isLikelyDuplicateTranscript(candidate: TranscriptSegment, existing: TranscriptSegment[]) {
  const normalized = normalizeTranscriptText(candidate.text);
  if (normalized.length < 2) {
    return false;
  }

  return existing.slice(-8).some((segment) => {
    // Compare on transcript timestamps, never Date.now(): the segment clock
    // and the wall clock are not the same origin, and mixing them made the
    // session-anchor decay silently wrong once already.
    if (candidate.startMs - segment.endMs > TRANSCRIPT_DEDUPE_WINDOW_MS) {
      return false;
    }
    if (segment.source && candidate.source && segment.source !== candidate.source) {
      return false;
    }
    const previous = normalizeTranscriptText(segment.text);
    if (!previous) {
      return false;
    }

    if (previous === normalized) {
      return true;
    }

    const shorter = previous.length < normalized.length ? previous : normalized;
    const longer = previous.length < normalized.length ? normalized : previous;
    return shorter.length >= 8 && longer.includes(shorter);
  });
}

export function detectLatestQuestion(segments: TranscriptSegment[]): LatestQuestionCandidate | null {
  for (const segment of [...segments].reverse()) {
    const text = segment.text.trim();
    if (!text) {
      continue;
    }

    const confidence = questionConfidence(text);
    if (confidence > 0) {
      return {
        text,
        confidence,
        reason: confidence >= 0.85 ? "explicit_question_signal" : "interview_prompt_signal",
      };
    }
  }

  const latest = segments[segments.length - 1]?.text.trim();
  if (!latest) {
    return null;
  }

  return {
    text: latest,
    confidence: 0.35,
    reason: "fallback_latest_transcript",
  };
}

export function questionConfidence(text: string) {
  const normalized = text.trim().toLowerCase();
  if (!normalized) {
    return 0;
  }
  if (isSetupTranscript(text)) {
    return 0;
  }

  const hasExplicitQuestion =
    /[?？]\s*$/.test(normalized) ||
    /(吗|呢|么|嘛|什么|为什么|怎么|怎样|如何|哪|多少|能不能|可不可以|有没有)/.test(normalized);
  if (hasExplicitQuestion) {
    return 0.9;
  }

  // "请描述 X 的底层结构" is just as explicit a question as "X 是什么？" — it
  // only looks weaker because Chinese doesn't need a particle for it. Scoring it
  // below the explicit tier silently disabled speculative prefetch (0.88 gate).
  if (hasInterviewRequest(text)) {
    return 0.9;
  }

  const interviewPrompt =
    hasInterviewPrompt(text) || hasTopicPrompt(text);

  return interviewPrompt ? 0.75 : 0;
}

export function detectQuestionCandidate(
  segment: TranscriptSegment,
  recentTranscript: TranscriptSegment[]
): QuestionCandidate | null {
  const text = segment.text.trim();
  const normalized = normalizeTranscriptText(text);
  const isShortFollowUp =
    /^(why|how|what|怎么说|为什么|什么意思|然后呢|还有呢)$/i.test(text.trim()) ||
    /^(为什么|怎么说|什么意思)$/.test(normalized);

  if (isFillerTranscript(text) || isSetupTranscript(text)) {
    debugLog(`[auto] candidate ignored reason=filler segment=${segment.id} text=${text.slice(0, 80)}`);
    return null;
  }

  if (normalized.length < 4 && !(isShortFollowUp && recentTranscript.length > 1)) {
    debugLog(`[auto] candidate ignored reason=too_short segment=${segment.id} text=${text.slice(0, 80)}`);
    return null;
  }

  const kind = classifyQuestionKind(text);
  const baseConfidence = questionConfidence(text);
  const hasPrompt = hasInterviewPrompt(text) || hasTopicPrompt(text);
  const explicitQuestion =
    /[?？]\s*$/.test(text) || /(吗|呢|么|什么|为什么|怎么|如何|哪|多少|能不能|可不可以|有没有)/.test(text);
  const confidenceBoost =
    kind === "system_design" || kind === "technical" ? 0.04 : kind === "behavioral" ? 0.02 : 0;
  const confidence = Math.min(
    0.96,
    Math.max(baseConfidence, explicitQuestion ? 0.9 : 0, hasPrompt ? 0.78 : 0, isShortFollowUp ? 0.72 : 0) +
      confidenceBoost
  );

  if (confidence < AUTO_ASSIST_MIN_CONFIDENCE) {
    debugLog(
      `[auto] candidate ignored reason=low_confidence confidence=${confidence.toFixed(2)} segment=${segment.id} text=${text.slice(0, 100)}`
    );
    return null;
  }

  return {
    id: `question-${segment.id}`,
    segmentId: segment.id,
    text,
    confidence,
    reason: explicitQuestion ? "explicit_question" : isShortFollowUp ? "short_follow_up" : hasPrompt ? "interview_prompt" : "question_signal",
    kind,
    createdAt: Date.now(),
  };
}

export function detectQuestionCandidateWithContext(
  segment: TranscriptSegment,
  recentTranscript: TranscriptSegment[]
): QuestionCandidate | null {
  const direct = detectQuestionCandidate(segment, recentTranscript);
  if (direct) {
    return direct;
  }

  const window = recentTranscript
    .filter((item) => segment.endMs - item.endMs <= 8_000)
    .slice(-4);
  if (window.length < 2) {
    return null;
  }

  const combinedText = joinTranscriptFragments(window.map((item) => item.text));
  if (normalizeTranscriptText(combinedText) === normalizeTranscriptText(segment.text)) {
    return null;
  }

  const syntheticSegment: TranscriptSegment = {
    ...segment,
    text: combinedText,
  };
  const candidate = detectQuestionCandidate(syntheticSegment, recentTranscript);
  if (!candidate) {
    return null;
  }

  return {
    ...candidate,
    id: `question-combined-${segment.id}`,
    text: combinedText,
    reason: `combined_${candidate.reason}`,
  };
}

export function questionKindLabel(kind: QuestionKind) {
  if (kind === "technical") return "技术问题";
  if (kind === "behavioral") return "行为问题";
  if (kind === "system_design") return "系统设计题";
  if (kind === "product") return "产品问题";
  return "检测到问题";
}

export function transcriptSimilarity(left: string, right: string) {
  const normalizedLeft = normalizeTranscriptText(left);
  const normalizedRight = normalizeTranscriptText(right);
  if (!normalizedLeft || !normalizedRight) return 0;
  if (normalizedLeft === normalizedRight) return 1;

  const shorter = normalizedLeft.length < normalizedRight.length ? normalizedLeft : normalizedRight;
  const longer = normalizedLeft.length < normalizedRight.length ? normalizedRight : normalizedLeft;
  if (shorter.length >= 8 && longer.includes(shorter)) return 0.92;

  const leftGrams = new Set(toCharGrams(normalizedLeft));
  const rightGrams = new Set(toCharGrams(normalizedRight));
  if (leftGrams.size === 0 || rightGrams.size === 0) return 0;

  let intersection = 0;
  for (const gram of leftGrams) {
    if (rightGrams.has(gram)) intersection += 1;
  }

  return intersection / (leftGrams.size + rightGrams.size - intersection);
}

export function buildInterviewAskContext(
  segments: TranscriptSegment[],
  perspective: MeetingPerspective = "candidate",
  documents: ContextDocument[] = []
): AskContext | null {
  const ordered = [...segments].sort((left, right) => left.endMs - right.endMs);
  const newestEndMs = ordered[ordered.length - 1]?.endMs ?? 0;
  const recentSegments = ordered.filter((segment) => newestEndMs - segment.endMs <= 120_000);
  if (recentSegments.length === 0) return null;

  const fullSegments = ordered.slice(-FULL_SESSION_SEGMENT_LIMIT);
  const recentTranscript = formatTranscriptBlock(recentSegments);
  const fullTranscript = formatTranscriptBlock(fullSegments);
  const latestSegment = recentSegments[recentSegments.length - 1];
  const documentContext = summarizeContextDocuments(documents);
  const latest: LatestQuestionCandidate = {
    text: "模型根据完整转写上下文自行判断",
    confidence: 1,
    reason: "model_decides_from_transcript",
  };
  const userMessage =
    perspective === "interviewer"
      ? [
          "The user is conducting a live interview. The app has been continuously transcribing meeting audio.",
          "The user pressed Enter to ask for interviewer-side help right now.",
          "Infer the candidate's latest answer, ambiguity, weakness, or signal from the transcript.",
          "Suggest the next interviewer move: a follow-up question, clarification prompt, or evaluation probe.",
          "The goal is to fairly evaluate the candidate's evidence and fit, not to trick or embarrass them.",
          "Ignore microphone checks, greetings, filler words, and old topics unless the recent transcript makes them relevant.",
          documentContext ? `Candidate/reference documents:\n${documentContext}` : "Candidate/reference documents: (none provided)",
          "",
          `Recent transcript:\n${recentTranscript}`,
          "",
          `Full session transcript:\n${fullTranscript}`,
        ].join("\n")
      : [
          "The user is in a live interview. The app has been continuously transcribing meeting audio.",
          "The user pressed Enter to ask for help right now.",
          "Do not rely on any app-extracted question. Infer the current interviewer question, unresolved prompt, or best next response directly from the transcript.",
          "Ignore microphone checks, greetings, filler words, and old questions unless the recent transcript makes them relevant.",
          "Prioritize the most recent meaningful interviewer prompt and the user's current answer state. If the user is already answering, help continue, correct, or tighten that answer.",
          documentContext ? `User resume/background documents:\n${documentContext}` : "User resume/background documents: (none provided)",
          "",
          `Recent transcript:\n${recentTranscript}`,
          "",
          `Full session transcript:\n${fullTranscript}`,
        ].join("\n");

  return {
    latest,
    recentSegments,
    fullSegments,
    userMessage,
    preview: `model_decides perspective=${perspective} latest_segment=${latestSegment.text.slice(0, 120)} recent_tail=${recentTranscript.slice(-220)}`,
  };
}

export function formatTranscriptBlock(segments: TranscriptSegment[]) {
  return segments.map((segment) => `[${Math.round(segment.endMs / 1000)}s] ${segment.text}`).join("\n");
}

export function buildPiCoachPrompt({
  trigger,
  transcript,
  candidate,
  latestAnswer,
  wakeSignal,
  wakeReason,
}: {
  trigger: CoachTrigger;
  transcript: TranscriptSegment[];
  candidate?: QuestionCandidate;
  latestAnswer?: string;
  wakeSignal?: string;
  wakeReason?: string;
}) {
  const recentSegments = transcript.slice(-10);

  return [
    `Trigger: ${trigger}`,
    wakeSignal ? `Wake signal: ${wakeSignal}` : "Wake signal: direct",
    wakeReason ? `Wake reason: ${wakeReason}` : "Wake reason: direct",
    candidate
      ? `Detected question: ${candidate.text}\nQuestion kind: ${candidate.kind}\nConfidence: ${candidate.confidence.toFixed(2)}`
      : "Detected question: none",
    latestAnswer ? `Latest answer shown to user:\n${latestAnswer.slice(0, 700)}` : "Latest answer shown to user: none",
    "",
    `Recent transcript:\n${formatTranscriptBlock(recentSegments)}`,
  ].join("\n");
}

export function coachTriggerLabel(trigger: CoachTrigger) {
  if (trigger === "agent_enter") return "主动唤醒";
  if (trigger === "agent_stt_question") return "问题救场";
  if (trigger === "session_started") return "开场";
  if (trigger === "research_signal") return "资料检索";
  if (trigger === "context_signal") return "上下文提醒";
  if (trigger === "question_detected") return "问题救场";
  if (trigger === "manual_ask_done") return "回答补位";
  if (trigger === "heartbeat") return "主动提醒";
  return "场边提示";
}

function isFillerTranscript(text: string) {
  const normalized = normalizeTranscriptText(text);
  if (!normalized) return true;
  return /^(嗯+|呃+|啊+|哦+|好+|好的|可以|ok|okay|right|yeah|yes|no|mm|mmm|uh|um|thanks|thankyou)$/.test(normalized);
}

export function isSetupTranscript(text: string) {
  const normalized = normalizeTranscriptText(text);
  return /^(你好)?(能)?听(得)?(到|见)吗(说一下|测试一下)?$/.test(normalized) ||
    /^(hello|hi)(canyouhearme|test)?$/i.test(normalized) ||
    // Mic / connection checks in a remote interview. These end in 吗 so they
    // look like questions, but answering them wastes a wake and pushes a junk
    // card into the overlay at the worst possible moment (the opening minute).
    /(听得到|听得见|能听见|能听到|有声音|看得到|看得见|能看到|屏幕共享|共享屏幕|可以开始|我们开始|开始了吗|准备好了吗)/.test(normalized) ||
    /\b(can you hear|do you hear|are you there|can you see|screen share|shall we start|are you ready)\b/i.test(normalized);
}

// --- imperative request detection -------------------------------------------
// The single biggest gap in question detection: Chinese interviewers phrase
// questions as *requests*, not as 吗/呢 questions. "请描述哈希映射的底层结构。"
// has no question mark and none of the usual particles, so it scored 0 on both
// the prefetch path and the wake path and the user silently got nothing.
//
// Two shapes cover nearly all of them:
//   1. Polite opener + verb  — 请描述 / 麻烦解释 / 能否介绍 / 你来讲 / 跟我讲
//   2. Verb + 一下            — 描述一下 / 讲一下 / 总结一下 / 手写一下

const REQUEST_VERBS =
  "描述|解释|说明|介绍|讲解|讲讲|讲|说说|说|谈谈|谈|分析|概括|总结|对比|比较|列举|举例|实现|手写|写|设计|推导|计算|评价|聊聊|聊";

const REQUEST_OPENERS = [
  "请", "麻烦你", "麻烦", "能否", "可否", "能不能", "可不可以", "能不能",
  "跟我", "你来", "你给", "你", "帮我", "想请你", "希望你能",
];

export const CHINESE_REQUEST_MARKERS: string[] = REQUEST_OPENERS.flatMap((opener) =>
  ["描述", "解释", "说明", "介绍", "讲", "说", "谈", "分析", "概括", "总结", "对比", "比较", "列举", "举例", "实现", "手写", "写", "设计", "推导", "计算", "评价", "聊"].map(
    (verb) => `${opener}${verb}`
  )
).concat(
  ["简单", "详细", "具体", "大概", "再"].flatMap((adverb) =>
    ["描述", "解释", "说明", "介绍", "讲", "说"].map((verb) => `请${adverb}${verb}`)
  ),
  // "X一下" imperative shape.
  ["描述", "解释", "说明", "介绍", "讲解", "讲", "说", "谈", "分析", "概括", "总结", "对比", "比较", "列举", "举例", "实现", "手写", "写", "设计", "推导", "计算", "评价", "聊"].flatMap(
    (verb) => [`${verb}一下`, `${verb}一${verb}`, `${verb}${verb}看`]
  )
);

/** Verb-of-interrogation at the start of an utterance: "实现一个 LRU 缓存". */
const CHINESE_IMPERATIVE_HEAD_RE = new RegExp(`^(?:${REQUEST_VERBS})`);

const ENGLISH_REQUEST_RE =
  /\b(please\s+(describe|explain|walk me through|tell me about|implement|write|design|compare|summarize)|describe|explain|implement|walk me through)\b/i;

/**
 * True when the text is an interviewer *asking* rather than stating. This is the
 * shared predicate for both the prefetch candidate path and the STT wake path —
 * keeping one definition is what stops the two from drifting apart again.
 */
export function hasInterviewRequest(text: string): boolean {
  if (!text) return false;
  if (CHINESE_REQUEST_MARKERS.some((marker) => text.includes(marker))) return true;
  if (CHINESE_IMPERATIVE_HEAD_RE.test(text.trim())) return true;
  return ENGLISH_REQUEST_RE.test(text);
}

/** Nouns that, at the end of a phrase, mark it as "explain X's <noun>". */
const TOPIC_NOUN_TAIL =
  "原理|机制|关系|区别|联系|作用|流程|实现|底层|源码|架构|抽象|调度|协调|切片|优先级|" +
  "结构|特点|特性|优点|缺点|场景|过程|步骤|方案|算法|复杂度|意义|方式|方法|策略|条件|时机|" +
  "情况|原因|差异|细节|思路|做法|瓶颈|优化|注意点|难点|好处|代价|优势|劣势|行为|本质|定义|" +
  "概念|分类|类型|种类|组成|构成|设计";

const TOPIC_NOUN_TAIL_RE = new RegExp(`[\\da-z\\u4e00-\\u9fa5]{2,}(?:的)?(?:${TOPIC_NOUN_TAIL})$`, "i");

function hasInterviewPrompt(text: string) {
  return /\b(tell me about|walk me through|talk me through|explain|describe|how would you|what do you think|what is your|why should|can you|could you|have you|design a|time complexity|space complexity|tradeoff|trade-off)\b/i.test(
    text
  ) ||
    /(讲一下|说一下|介绍一下|解释一下|聊聊|谈谈|你怎么看|你怎么理解|你会怎么|设计一个|系统设计|复杂度|权衡|取舍)/.test(text) ||
    hasInterviewRequest(text);
}

function hasTopicPrompt(text: string) {
  const normalized = normalizeTranscriptText(text);
  if (!normalized) {
    return false;
  }
  // An explicit request outranks the declarative-answer heuristic: "请描述 X 的
  // 底层结构" is a question no matter how its clauses are shaped.
  if (hasInterviewRequest(text)) {
    return true;
  }
  if (isLikelyDeclarativeAnswer(normalized)) {
    return false;
  }

  return TOPIC_NOUN_TAIL_RE.test(normalized) ||
    /(是什么关系|做了什么|干什么|怎么实现|如何实现|底层代码)/.test(normalized);
}

function isLikelyDeclarativeAnswer(normalized: string) {
  return (/是/.test(normalized) && !/(是什么|是不是|为什么|怎么|如何)/.test(normalized)) ||
    /(主要是|特点是|可以|负责|通过|基于|实现了|用来|从而|这个系统|它其实|他其实)/.test(normalized);
}

function joinTranscriptFragments(fragments: string[]) {
  return fragments
    .map((fragment) => fragment.trim())
    .filter(Boolean)
    .join("")
    .replace(/。(?=[\u4e00-\u9fa5A-Za-z0-9])/g, "")
    .replace(/[,.，]\s*$/g, "")
    .trim();
}

function classifyQuestionKind(text: string): QuestionKind {
  if (/(系统设计|架构|高并发|扩展|缓存|队列|分布式|design a system|system design|scalab|cache|queue|distributed)/i.test(text)) {
    return "system_design";
  }
  if (/(复杂度|算法|数据结构|react|vue|rust|typescript|javascript|数据库|索引|事务|api|性能|time complexity|space complexity|algorithm|database|index|transaction)/i.test(text)) {
    return "technical";
  }
  if (/(经历|项目|冲突|失败|优点|缺点|压力|合作|leadership|conflict|failure|strength|weakness|tell me about yourself|behavioral)/i.test(text)) {
    return "behavioral";
  }
  if (/(产品|用户|指标|增长|留存|转化|需求|商业|customer|user|metric|growth|retention|conversion|product)/i.test(text)) {
    return "product";
  }
  return "general";
}

function toCharGrams(text: string) {
  if (text.length <= 2) return [text];
  return Array.from({ length: text.length - 1 }, (_, index) => text.slice(index, index + 2));
}
