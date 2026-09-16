// Session-scoped, deterministic conversation memory.
//
// Modeled on the pattern used by long-session interview copilots: keep the live
// prompt tiny, but retain a *structured* view of what the conversation is about
// so a question asked at minute 40 still inherits the language/stack established
// at minute 1.
//
// Hard constraints this module must satisfy (latency is the whole point):
//   • NO LLM, NO network, NO I/O. Regex + array scans only.
//   • Bounded output. The anchor line is a few dozen characters; the long-range
//     recall block is capped at MAX_RECALL_CHARS.
//   • Fail-safe. When nothing is confidently known, return "" — a missing hint
//     is strictly better than a wrong one (a wrong language anchor is exactly
//     how a Java interview ends up answered in JavaScript).
//
// Everything here is pure and synchronous, so it runs on the wake path without
// adding a single millisecond of network latency.

import type { TranscriptSegment } from "../../app/types";

export type FactKind = "language" | "framework" | "storage" | "topic";

export interface SessionFact {
  kind: FactKind;
  value: string;
  /** When it was last mentioned (segment endMs). */
  endMs: number;
  /** How many times it has been mentioned; boosts salience. */
  hits: number;
}

/** Half-life in ms for salience decay, by kind. */
const HALF_LIFE_MS: Record<FactKind, number> = {
  // A language established at the start of an interview stays relevant for the
  // whole session — this is the anchor that prevents cross-language answers.
  language: 3_600_000,
  framework: 1_800_000,
  storage: 1_800_000,
  topic: 900_000,
};

/** Below this salience a fact is too stale to be worth stating. */
const MIN_SALIENCE = 0.12;
const MAX_ANCHORS_PER_KIND = 2;
const MAX_RECALL_CHARS = 320;
const MAX_RECALL_SEGMENTS = 2;
const MAX_FACTS = 120;

// --- vocabularies -----------------------------------------------------------
// Deliberately covers backend/Java stacks first: an interview copilot whose
// dictionary only knows frontend tech will mislabel a Java interview. Matching
// is case-insensitive; Chinese terms match by substring (no word boundaries).

const LANGUAGE_PATTERNS: Array<{ value: string; re: RegExp }> = [
  { value: "JavaScript", re: /\b(javascript|\bjs\b)\b/i },
  // `(?!\s?script)` is load-bearing: without it "JavaScript" also matches "Java"
  // and a frontend answer gets produced for a backend interview.
  { value: "Java", re: /\bjava\b(?!\s?script)/i },
  { value: "TypeScript", re: /\btypescript\b|\bts\b/i },
  { value: "Python", re: /\bpython\b|\bpy\b/i },
  { value: "Go", re: /\bgolang\b|\bgo\s?语言\b/i },
  { value: "Rust", re: /\brust\b/i },
  { value: "C++", re: /c\+\+/i },
  { value: "C#", re: /\bc#\b|\.net/i },
  { value: "Kotlin", re: /\bkotlin\b/i },
  { value: "Swift", re: /\bswift\b/i },
  { value: "PHP", re: /\bphp\b/i },
  { value: "Ruby", re: /\bruby\b/i },
  { value: "Scala", re: /\bscala\b/i },
  { value: "SQL", re: /\bsql\b/i },
];

const FRAMEWORK_PATTERNS: Array<{ value: string; re: RegExp }> = [
  { value: "Spring Boot", re: /\bspring\s?boot\b|springboot/i },
  { value: "Spring", re: /\bspring\b(?!\s?boot)/i },
  { value: "MyBatis", re: /\bmybatis\b/i },
  { value: "Hibernate", re: /\bhibernate\b|\bjpa\b/i },
  { value: "Dubbo", re: /\bdubbo\b/i },
  { value: "Netty", re: /\bnetty\b/i },
  { value: "Tomcat", re: /\btomcat\b/i },
  { value: "Maven", re: /\bmaven\b/i },
  { value: "Gradle", re: /\bgradle\b/i },
  { value: "React", re: /\breact\b(?!\s?native)/i },
  { value: "Vue", re: /\bvue\b/i },
  { value: "Angular", re: /\bangular\b/i },
  { value: "Node.js", re: /\bnode\.?js\b|\bnode\b/i },
  { value: "Django", re: /\bdjango\b/i },
  { value: "Flask", re: /\bflask\b/i },
  { value: "FastAPI", re: /\bfastapi\b/i },
  { value: "gRPC", re: /\bgrpc\b/i },
];

const STORAGE_PATTERNS: Array<{ value: string; re: RegExp }> = [
  { value: "MySQL", re: /\bmysql\b/i },
  { value: "PostgreSQL", re: /\bpostgres(ql)?\b/i },
  { value: "Redis", re: /\bredis\b/i },
  { value: "Kafka", re: /\bkafka\b/i },
  { value: "RocketMQ", re: /\brocketmq\b/i },
  { value: "MongoDB", re: /\bmongo(db)?\b/i },
  { value: "Elasticsearch", re: /\belasticsearch\b|\bes\b/i },
  { value: "ZooKeeper", re: /\bzo+keeper\b/i },
  { value: "Docker", re: /\bdocker\b/i },
  { value: "Kubernetes", re: /\bkubernetes\b|\bk8s\b/i },
  { value: "Nginx", re: /\bnginx\b/i },
];

const TOPIC_TERMS = [
  // Java / JVM
  "多线程", "并发", "线程池", "锁", "同步", "volatile", "synchronized", "锁升级",
  "垃圾回收", "垃圾收集", "gc", "jvm", "内存模型", "类加载", "双亲委派", "反射",
  "注解", "泛型", "异常", "集合", "hashmap", "concurrenthashmap", "string",
  "spring循环依赖", "ioc", "aop", "事务", "索引", "慢查询", "分库分表",
  // distributed
  "分布式", "一致性", "cap", "限流", "熔断", "降级", "幂等", "消息队列",
  "微服务", "注册中心", "负载均衡", "缓存穿透", "缓存雪崩", "分布式锁",
  "分布式事务", "raft", "paxos",
  // algorithms
  "动态规划", "贪心", "回溯", "分治", "递归", "双指针", "滑动窗口", "前缀和",
  "二叉树", "链表", "哈希表", "哈希", "堆", "栈", "队列", "图", "拓扑排序",
  "bfs", "dfs", "最短路径", "排序", "二分", "时间复杂度", "空间复杂度",
];

/** Extract the facts a single transcript segment establishes. Pure + sync. */
export function extractSessionFacts(text: string, endMs: number): SessionFact[] {
  const raw = (text || "").trim();
  if (!raw) return [];

  const out: SessionFact[] = [];
  const push = (kind: FactKind, value: string) => {
    if (out.some((f) => f.kind === kind && f.value === value)) return;
    out.push({ kind, value, endMs, hits: 1 });
  };

  for (const p of LANGUAGE_PATTERNS) if (p.re.test(raw)) push("language", p.value);
  for (const p of FRAMEWORK_PATTERNS) if (p.re.test(raw)) push("framework", p.value);
  for (const p of STORAGE_PATTERNS) if (p.re.test(raw)) push("storage", p.value);

  const lower = raw.toLowerCase();
  for (const term of TOPIC_TERMS) {
    const needle = /^[a-z0-9+#.]+$/i.test(term) ? term : term.toLowerCase();
    if (lower.includes(needle)) push("topic", term);
  }

  return out;
}

/** Decay + repetition-boosted salience, mirroring a recency-weighted memory. */
export function factSalience(fact: SessionFact, nowMs: number): number {
  const age = Math.max(0, nowMs - fact.endMs);
  const decay = Math.pow(0.5, age / HALF_LIFE_MS[fact.kind]);
  return Math.min(1, decay + Math.min(0.3, (fact.hits - 1) * 0.1));
}

export class SessionFactStore {
  private facts: SessionFact[] = [];

  reset() {
    this.facts = [];
  }

  ingest(segments: TranscriptSegment[]) {
    for (const segment of segments) {
      for (const fact of extractSessionFacts(segment.text, segment.endMs)) {
        const existing = this.facts.find(
          (f) => f.kind === fact.kind && f.value === fact.value
        );
        if (existing) {
          existing.hits += 1;
          existing.endMs = Math.max(existing.endMs, fact.endMs);
        } else {
          this.facts.push(fact);
        }
      }
    }
    if (this.facts.length > MAX_FACTS) {
      this.facts = this.facts
        .sort((a, b) => b.endMs - a.endMs)
        .slice(0, MAX_FACTS);
    }
  }

  /**
   * The compact anchor line injected into the prompt. Returns "" when nothing is
   * known yet — never a guess.
   */
  anchors(nowMs: number): string {
    const byKind = new Map<FactKind, SessionFact[]>();
    for (const fact of this.facts) {
      if (factSalience(fact, nowMs) < MIN_SALIENCE) continue;
      const list = byKind.get(fact.kind) ?? [];
      list.push(fact);
      byKind.set(fact.kind, list);
    }

    const order: FactKind[] = ["language", "framework", "storage", "topic"];
    const lines: string[] = [];
    for (const kind of order) {
      const list = (byKind.get(kind) ?? [])
        .sort((a, b) => factSalience(b, nowMs) - factSalience(a, nowMs))
        .slice(0, MAX_ANCHORS_PER_KIND);
      for (const fact of list) {
        lines.push(`- ${kind}: ${fact.value} (${formatAge(nowMs - fact.endMs)} ago)`);
      }
    }
    if (lines.length === 0) return "";
    return [
      "Session anchors (inferred from the WHOLE conversation, not just the recent window):",
      ...lines,
    ].join("\n");
  }

  size() {
    return this.facts.length;
  }
}

// --- long-range recall ------------------------------------------------------
// When the latest question is a bare callback ("那再说说第二个?", "它有什么缺点?"),
// find the earlier turn it refers to. Lexical overlap only — no embeddings, no
// LLM — and conservative: a thin match returns "" rather than a wrong turn.

const CJK_STOP = new Set([
  "这个", "那个", "什么", "怎么", "为什么", "可以", "我们", "你们", "他们", "自己",
  "一下", "一个", "没有", "就是", "还是", "但是", "因为", "所以", "如果", "然后",
  "现在", "之前", "刚才", "上面", "下面", "这里", "那里", "问题", "说说", "讲讲",
  "觉得", "认为", "知道", "理解", "看法", "介绍", "谈谈", "意思",
]);

const EN_STOP = new Set([
  "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
  "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
  "can", "that", "this", "these", "those", "it", "its", "you", "your", "we",
  "they", "them", "what", "when", "where", "which", "who", "why", "how", "about",
  "with", "for", "from", "into", "over", "under", "again", "then", "than",
  "there", "here", "more", "most", "tell", "talk", "think", "know", "mean",
]);

/** Bare-callback cues: the question points backwards without naming its subject. */
const CALLBACK_RE =
  /那(个|它|这个)?|它|这个|那个|上面|刚才|之前|前面|再说说|再讲讲|第二|第三|上一个|刚才那个|\b(it|that|this|there|the\s+(first|second|last)\s+one|earlier|above|again)\b/i;

export function isCallbackQuestion(text: string): boolean {
  const t = (text || "").trim();
  if (!t) return false;
  if (t.length > 80) return false; // long questions carry their own context
  return CALLBACK_RE.test(t);
}

/** Content keys: latin words + CJK bigrams, both stopword-filtered. */
function contentKeys(text: string): Set<string> {
  const lower = (text || "").toLowerCase();
  const keys = new Set<string>();

  for (const word of lower.match(/[a-z][a-z0-9+#.]{2,}/g) ?? []) {
    if (!EN_STOP.has(word)) keys.add(word);
  }

  const cjk = lower.replace(/[^\u4e00-\u9fa5]/g, "");
  for (let i = 0; i + 2 <= cjk.length; i++) {
    const bigram = cjk.slice(i, i + 2);
    if (!CJK_STOP.has(bigram)) keys.add(bigram);
  }
  return keys;
}

/**
 * Return the earlier transcript block a callback question most likely refers to.
 * Candidates are limited to turns older than `recentCutoffMs` (content inside the
 * live window is already in the prompt, so re-surfacing it would just waste
 * tokens). Empty string when nothing clears the threshold.
 */
export function recallEarlierTurns(
  question: string,
  durable: TranscriptSegment[],
  recentCutoffMs: number,
): string {
  const qKeys = contentKeys(question);
  if (qKeys.size === 0) return "";
  // Shared technical terms ("限流", "HashMap", "Redis") are far stronger evidence
  // than shared n-grams, so they carry a heavy weight. A bare pronoun callback
  // ("那它的根本原因呢") shares none — it falls back to the anchor line instead,
  // which is the correct fail-safe.
  const qTerms = new Set(
    extractSessionFacts(question, 0).map((fact) => fact.value.toLowerCase())
  );

  const scored: Array<{ segment: TranscriptSegment; score: number }> = [];
  for (const segment of durable) {
    if (segment.endMs >= recentCutoffMs) continue;
    const text = (segment.text || "").trim();
    if (text.length < 8) continue;
    const keys = contentKeys(text);
    let score = 0;
    for (const key of qKeys) if (keys.has(key)) score++;
    if (qTerms.size > 0) {
      for (const fact of extractSessionFacts(text, 0)) {
        if (qTerms.has(fact.value.toLowerCase())) score += 3;
      }
    }
    // Require real topical overlap. Failing safe is the point: a wrong turn is
    // worse than no turn.
    if (score >= 3) scored.push({ segment, score });
  }

  if (scored.length === 0) return "";

  scored.sort((a, b) => b.score - a.score || b.segment.endMs - a.segment.endMs);
  const top = scored.slice(0, MAX_RECALL_SEGMENTS).sort((a, b) => a.segment.endMs - b.segment.endMs);

  let body = top
    .map(({ segment }) => {
      const who = segment.speaker === "user" ? "ME" : "INTERVIEWER";
      return `[${who}]: ${segment.text.trim()}`;
    })
    .join("\n");
  if (body.length > MAX_RECALL_CHARS) body = `${body.slice(0, MAX_RECALL_CHARS)}…`;

  return `<earlier_context note="the latest question refers back to something said earlier in this conversation; this is the most relevant earlier turn, verbatim">\n${body}\n</earlier_context>`;
}

function formatAge(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.round(minutes / 60)}h`;
}
