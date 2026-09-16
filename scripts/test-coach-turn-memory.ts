// End-to-end check of the follow-up memory at the RUNTIME level (the prompt
// probes in test-followup-memory.ts only exercise buildAgentPrompt directly).
//
// The question that matters is not "can we render the block" but "does the
// answer actually survive the round trip": the runtime must write what the
// coach just said into the store, and the NEXT wake must see it. Run:
//   npx tsx scripts/test-coach-turn-memory.ts

import assert from "node:assert/strict";
import { AgentRuntime, ContextStore, createEnterWake } from "../src/runtime/agent/index.ts";
import type { WakeEvent } from "../src/runtime/agent/wake.ts";
import type { AssistantSuggestion } from "../src/app/types.ts";

Object.defineProperty(globalThis, "window", { configurable: true, value: globalThis });

const A1 =
  "限流分三层：网关做单机令牌桶粗筛，Redis 用 Lua 脚本做集群精筛，滑动窗口用 ZSET 存时间戳。";
const A2 = "这是第二轮的答案。";

const prompts: string[] = [];
let call = 0;
const context = new ContextStore();

const runtime = new AgentRuntime(
  context,
  {
    complete: async (prompt): Promise<AssistantSuggestion> => {
      prompts.push(prompt.text);
      call += 1;
      return {
        answer: call === 1 ? A1 : A2,
        bullets: [],
        clarifyingQuestion: null,
      };
    },
  },
  {
    onMessage: () => {},
    onError: (message) => assert.fail(message),
  }
);

// Wakes go through `enter` rather than `stt_question`: the transcript path also
// enforces a 10s inter-question cooldown, and a unit test should not sleep.
const ask = (text: string): WakeEvent => ({ ...createEnterWake(), evidence: [text] });

async function tick() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

runtime.wake(ask("你们的限流是怎么做的？"));
await tick();
await tick();

runtime.wake(ask("刚才那个方案在分布式下有什么问题？"));
await tick();
await tick();

assert.equal(prompts.length, 2, `expected 2 coach calls, got ${prompts.length}`);
const [first, second] = prompts;

assert.ok(!first.includes("<previous_coach_answers"), "第一轮不该有前文（此时还没有答过）");
assert.ok(
  second.includes("<previous_coach_answers"),
  "第二轮必须带上教练上一轮的答案"
);
assert.ok(second.includes("ZSET"), "第二轮必须能读到上一轮的具体方案内容");
assert.ok(
  second.includes("build on that answer and go one level deeper"),
  "注入块存在时必须同时带上护栏指令（否则模型会接着自己上轮说）"
);

// A cancelled run must not wipe the memory: only "new conversation" does that.
assert.equal(
  context.snapshot(120_000).previousCoachTurns.length,
  2,
  "两轮答案都应在记忆里"
);
runtime.cancel("test");
assert.equal(
  context.snapshot(120_000).previousCoachTurns.length,
  2,
  "cancel() 不该清记忆 —— 用户只是放弃了这一轮，没有重开会话"
);
context.clear();
assert.equal(
  context.snapshot(120_000).previousCoachTurns.length,
  0,
  "只有 clear()（新建会话）才清空"
);

console.log("  ok   第一轮 prompt 不含前文");
console.log("  ok   第二轮 prompt 带上了上一轮答案（含 ZSET 细节）");
console.log("  ok   护栏指令随注入块一起出现");
console.log("  ok   cancel() 不清记忆，只有新建会话才 clear");
console.log("\ncoach turn memory ok");
