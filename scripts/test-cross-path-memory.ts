/**
 * The two ask paths used to have separate memories: the proactive coach kept
 * `previousCoachTurns` in its ContextStore, while manual ask (typed box / Fn
 * voice overlay) sent its own `turns` straight to Rust. Ask something by hand,
 * let the interviewer follow up on it, and the coach answered as if the topic
 * had never come up — and vice versa.
 *
 * This checks the merge from both directions, plus the two ways it can silently
 * go wrong: the same exchange being sent twice, and a manual turn being
 * presented to the model as something the interviewer said.
 *
 * Run: ./node_modules/.bin/tsx scripts/test-cross-path-memory.ts
 */
import { buildAgentChatHistory } from "../src/app/agentChat.ts";
import { ContextStore } from "../src/runtime/agent/contextStore.ts";
import { buildAgentPrompt } from "../src/runtime/agent/prompt.ts";
import { createEnterWake } from "../src/runtime/agent/wake.ts";
import type { AgentChatTurn, AssistantSuggestion } from "../src/app/types.ts";

let failures = 0;

function expect(label: string, ok: boolean) {
  if (!ok) failures += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${label}`);
}

function section(title: string) {
  console.log(`\n${title}`);
}

function suggestion(answer: string): AssistantSuggestion {
  return { answer, bullets: [], clarifyingQuestion: null, kind: "knowledge" };
}

function chatTurn(id: string, question: string, answer: string, createdAt: number): AgentChatTurn {
  return { id, createdAt, question, suggestion: suggestion(answer), error: null, toolTraces: [] };
}

// --- 1. coach answer -> manual ask ----------------------------------------
section("1. 教练答过的内容，手动 Ask 能看到");
{
  const store = new ContextStore();
  store.pushCoachTurn({
    question: "你们的限流是怎么做的？",
    answer: "网关层单机令牌桶兜底，核心接口用 Redis + Lua 做分布式限流，滑动窗口用 ZSET。",
    atMs: 1_000,
    origin: "coach",
  });

  const history = buildAgentChatHistory([], store.coachTurnSnapshot());
  expect("手动 Ask 的 turns 里出现了教练答案", history.length === 1);
  expect("内容是教练那轮原文（含 ZSET）", /ZSET/.test(history[0]?.suggestion.answer ?? ""));
}

// --- 2. manual ask -> coach -----------------------------------------------
section("2. 手动 Ask 答过的内容，教练能看到");
{
  const store = new ContextStore();
  store.pushCoachTurn({
    question: "帮我讲讲 Sentinel 的集群流控",
    answer: "Sentinel 集群流控由 token server 统一保存全局令牌数，各节点向它申请配额。",
    atMs: 2_000,
    origin: "manual",
  });
  store.pushTranscript({
    id: "s1",
    speaker: "other",
    text: "那扩容的时候总量怎么保证？",
    startMs: 3_000,
    endMs: 4_000,
    isFinal: true,
  });

  const snapshot = store.snapshot(120_000);
  const prompt = buildAgentPrompt(
    { ...createEnterWake(), evidence: ["那扩容的时候总量怎么保证？"] },
    snapshot
  ).text;

  expect("教练 prompt 里带上了手动 Ask 的答案", /token server/.test(prompt));
  expect(
    "标注了这是用户主动问的，不是面试官问的",
    /\(asked by the user, not the interviewer\)/.test(prompt)
  );
  expect(
    "transcript 里并没有这句（证明来自记忆而非转录）",
    !snapshot.durableTranscript.some((seg) => /token server/.test(seg.text))
  );
}

// --- 3. no double-send ------------------------------------------------------
section("3. 同一轮不会被发两遍");
{
  const store = new ContextStore();
  // The manual path records into coach memory *and* keeps its own chat turn, so
  // naive concatenation sends Rust the same exchange twice.
  store.pushCoachTurn({
    question: "讲讲 HashMap 扩容",
    answer: "扩容按两倍容量重建，JDK 8 用高低位链表拆分，避免重新计算 hash。",
    atMs: 5_000,
    origin: "manual",
  });
  const chat = [chatTurn("a1", "讲讲 HashMap 扩容", "扩容按两倍容量重建，JDK 8 用高低位链表拆分，避免重新计算 hash。", 5_000)];

  const history = buildAgentChatHistory(chat, store.coachTurnSnapshot());
  expect("只保留一份，没有重复", history.length === 1);
}

// --- 4. ordering ------------------------------------------------------------
section("4. 两条时间线按真实先后合并");
{
  const store = new ContextStore();
  store.pushCoachTurn({
    question: "介绍一下你们的缓存策略",
    answer: "读多写少走 Cache Aside，热点 key 做本地缓存兜底。",
    atMs: 1_000,
    origin: "coach",
  });
  const chat = [chatTurn("a1", "那缓存击穿怎么处理", "用互斥锁或逻辑过期，避免同一 key 同时回源。", 2_000)];

  const history = buildAgentChatHistory(chat, store.coachTurnSnapshot());
  expect("两轮都在", history.length === 2);
  expect("教练那轮排在前面（发生得更早）", /Cache Aside/.test(history[0]?.suggestion.answer ?? ""));
  expect("手动那轮排在后面", /互斥锁/.test(history[1]?.suggestion.answer ?? ""));
}

// --- 5. clear() wipes both ---------------------------------------------------
section("5. 新建会话后两边都清空");
{
  const store = new ContextStore();
  store.pushCoachTurn({ question: "q", answer: "上一场面试的答案", atMs: 1_000, origin: "coach" });
  store.clear();

  expect("教练记忆已清空", store.coachTurnSnapshot().length === 0);
  expect("合并后的手动历史也为空", buildAgentChatHistory([], store.coachTurnSnapshot()).length === 0);
}

console.log(`\n${failures === 0 ? "all probes ok" : `${failures} probe(s) failed`}`);
process.exit(failures === 0 ? 0 : 1);
