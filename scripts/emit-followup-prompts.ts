/**
 * Emit the REAL coach prompts for the follow-up benchmark, interviewer-only mode.
 *
 * Earlier benchmarks hand-wrote an approximation of the prompt in Python, which
 * measures a prompt nobody ships. This drives the actual `ContextStore` +
 * `buildAgentPrompt` with `audioSource: "system"`, so what the models receive is
 * byte-identical to what the installed app sends.
 *
 * Each case is emitted twice:
 *   before  coach memory empty        (the app before 2026-09-11)
 *   after   previous answer in memory (what ships now)
 *
 * Run: ./node_modules/.bin/tsx scripts/emit-followup-prompts.ts
 * Then: python3 scripts/bench-followup-system.py
 */
import { writeFileSync } from "node:fs";
import { ContextStore } from "../src/runtime/agent/contextStore.ts";
import { buildAgentPrompt } from "../src/runtime/agent/prompt.ts";
import { detectSttWake } from "../src/runtime/agent/demand.ts";
import type { TranscriptSegment } from "../src/app/types.ts";

const WINDOW_MS = 120_000;

type Case = {
  key: string;
  label: string;
  /** What the interviewer asked first. */
  q1: string;
  /** What the coach answered (the memory under test). */
  a1: string;
  /** The follow-up. */
  q2: string;
  /** Terms a context-aware answer should contain. */
  expect: string[];
  /** Terms that prove the model drifted to the wrong referent. */
  avoid: string[];
  /** Gap between the two interviewer lines, ms. Default 40s (inside window). */
  gapMs?: number;
  notes: string;
};

/**
 * Deliberately mixed: some cases the model can answer from world knowledge even
 * with zero memory (so memory should show ~no gain), others are impossible
 * without it. A benchmark where memory always wins is measuring the scenarios,
 * not the feature.
 */
const CASES: Case[] = [
  {
    key: "pronoun_bare",
    label: "C1 裸代词 · 它的时间复杂度",
    q1: "请描述哈希映射的底层结构。",
    a1: "哈希表底层是数组加链表，JDK 8 之后链表长度超过 8 会转红黑树。数组存桶，哈希函数定位下标，冲突时挂到桶后面的链表上。",
    q2: "那它的时间复杂度是多少？",
    expect: ["o(1)", "哈希", "冲突"],
    avoid: [],
    notes: "八股题，模型自己就知道，记忆预期无增益（对照基线）",
  },
  {
    key: "ellipsis_ordinal",
    label: "C2 序数省略 · 第二种的重写机制",
    q1: "Redis 的持久化方式有哪些？",
    a1: "Redis 有两种持久化：RDB 是定时把内存快照写成二进制文件，恢复快但可能丢最后几分钟数据；AOF 记录每条写命令，靠重写压缩体积，数据更安全但文件更大。",
    q2: "那第二种的重写机制具体是怎么工作的？",
    expect: ["aof", "重写"],
    avoid: ["jit", "视图", "编译"],
    notes: "无记忆时 deepseek 系会把重写映射到 JVM/数据库领域编造",
  },
  {
    key: "open_design",
    label: "C3 开放设计题 · 刚才那个方案在分布式下",
    q1: "你们的服务是怎么做限流的？",
    a1: "我们分两层：网关层用单机令牌桶做粗粒度兜底，核心写接口用 Redis 加 Lua 脚本做分布式限流，保证判断和扣减的原子性，滑动窗口用 ZSET 按时间戳存请求。",
    q2: "刚才那个方案在分布式部署下有什么问题？",
    expect: ["redis", "令牌"],
    avoid: [],
    notes: "答案因人而异，没有记忆只能猜候选人用了什么",
  },
  {
    key: "conflict_wording",
    label: "C4 面试官措辞与记忆冲突 · Sentinel",
    q1: "你们的服务是怎么做限流的？",
    a1: "我们分两层：网关层用单机令牌桶兜底，核心写接口用 Redis 加 Lua 脚本做分布式限流，保证原子性。",
    q2: "你刚说用的是 Sentinel，那集群扩容的时候限流总量怎么保证？",
    expect: ["sentinel"],
    // Defending its own Redis answer against the interviewer is the failure.
    avoid: ["lua"],
    notes: "候选人实际说了 Sentinel（对面模式听不到）→ 必须跟面试官走",
  },
  {
    key: "topic_switch",
    label: "C5 换话题 · 记忆必须被忽略",
    q1: "你们的服务是怎么做限流的？",
    a1: "网关层单机令牌桶兜底，核心接口用 Redis 加 Lua 做分布式限流，滑动窗口用 ZSET。",
    q2: "说说 MySQL 的索引失效有哪些情况？",
    expect: ["索引"],
    // Dragging limiting into an index question = the block became a trap.
    avoid: ["限流", "令牌桶"],
    notes: "护栏的反向验证：换话题时记忆必须完全不干扰",
  },
  {
    key: "depth_probe",
    label: "C6 追问深挖 · 为什么不用另一种",
    q1: "分布式锁一般怎么实现？",
    a1: "常见是 Redis 的 SET NX EX，加随机 value 防误删，释放时用 Lua 校验 value 再删。需要更强一致就用 etcd 或 ZooKeeper 的临时顺序节点。",
    q2: "那为什么不直接用数据库的唯一索引来做？",
    expect: ["数据库", "索引"],
    avoid: [],
    notes: "追问句自带关键词，但需要承接上一轮的选型语境",
  },
  {
    key: "long_gap",
    label: "C7 超窗追问 · 隔 6 分钟回头问",
    q1: "你们的服务是怎么做限流的？",
    a1: "网关层单机令牌桶兜底，核心写接口用 Redis 加 Lua 做分布式限流，滑动窗口用 ZSET 按时间戳存请求。",
    q2: "回到刚才限流那块，热点 key 会不会把 Redis 打挂？",
    expect: ["热点", "redis"],
    avoid: [],
    // Past AGENT_CONTEXT_WINDOW_MS, so Q1 drops out of `Recent transcript`.
    gapMs: 360_000,
    notes: "Q1 已掉出 120s 窗口，只剩 coach 记忆 + anchors + 可能的召回",
  },
  {
    key: "correction",
    label: "C8 纠正性追问 · 我问的是 JDK 7",
    q1: "HashMap 的底层结构是什么？",
    a1: "数组加链表，JDK 8 起链表超过 8 且容量到 64 会转红黑树，查询从 O(n) 降到 O(log n)。",
    q2: "你说的不对，我问的是 JDK 7 的实现。",
    expect: ["链表"],
    avoid: ["红黑树"],
    notes: "追问句自带 JDK 7 关键词，记忆预期无增益（对照基线）",
  },
];

function seg(text: string, atMs: number, id: string): TranscriptSegment {
  return {
    id,
    speaker: "other",
    text,
    startMs: Math.max(atMs - 3_000, 0),
    endMs: atMs,
    isFinal: true,
  };
}

/**
 * Interviewer-only means the candidate NEVER lands in the transcript — not even
 * a "嗯，好的". The two interviewer lines sit directly adjacent.
 */
function buildPrompt(testCase: Case, withMemory: boolean) {
  const store = new ContextStore();
  store.setSessionConfig({ kind: "remote", audioSource: "system", goal: "" });
  store.setSessionId(`bench-${testCase.key}`);

  const t0 = 60_000;
  const gap = testCase.gapMs ?? 40_000;

  store.pushTranscript(seg(testCase.q1, t0, `${testCase.key}-q1`));

  if (withMemory) {
    store.pushCoachTurn({
      question: testCase.q1,
      answer: testCase.a1,
      atMs: t0 + 2_000,
      origin: "coach",
    });
  }

  const q2At = t0 + gap;
  const q2Segment = seg(testCase.q2, q2At, `${testCase.key}-q2`);
  store.pushTranscript(q2Segment);

  const snapshot = store.snapshot(WINDOW_MS);
  // Go through the real wake detector so `evidence` is populated exactly as it
  // is at runtime — that is what drives callback recall.
  const wake = detectSttWake(q2Segment, "remote", "candidate");
  if (!wake) {
    throw new Error(`case ${testCase.key}: 追问句没有被识别为问题，先修 detectSttWake 再跑基准`);
  }
  const prompt = buildAgentPrompt(wake, snapshot);

  return {
    text: prompt.text,
    recentCount: snapshot.recentTranscript.length,
    coachTurns: snapshot.previousCoachTurns.length,
    hasEarlier: prompt.text.includes("<earlier_context"),
    hasMemoryBlock: prompt.text.includes("<previous_coach_answers"),
    anchors: snapshot.anchors.split("\n").filter((l) => l.startsWith("- ")).length,
  };
}

const out = CASES.map((testCase) => {
  const before = buildPrompt(testCase, false);
  const after = buildPrompt(testCase, true);
  return {
    key: testCase.key,
    label: testCase.label,
    q1: testCase.q1,
    q2: testCase.q2,
    a1: testCase.a1,
    expect: testCase.expect,
    avoid: testCase.avoid,
    notes: testCase.notes,
    gapMs: testCase.gapMs ?? 40_000,
    before,
    after,
  };
});

const target = "/tmp/followup_prompts.json";
writeFileSync(target, JSON.stringify(out, null, 1), "utf8");

console.log(`emitted ${out.length} cases -> ${target}\n`);
for (const entry of out) {
  console.log(
    `${entry.label}\n  before: recent=${entry.before.recentCount} coachTurns=${entry.before.coachTurns} ` +
      `earlier=${entry.before.hasEarlier ? "hit" : "miss"} memBlock=${entry.before.hasMemoryBlock} anchors=${entry.before.anchors}\n` +
      `  after : recent=${entry.after.recentCount} coachTurns=${entry.after.coachTurns} ` +
      `earlier=${entry.after.hasEarlier ? "hit" : "miss"} memBlock=${entry.after.hasMemoryBlock} anchors=${entry.after.anchors}`
  );
}
