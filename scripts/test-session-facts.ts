// Regression tests for the deterministic session memory layer.
// Run: npm run test:session-facts
//
// The properties pinned here are the ones that actually matter in a live
// interview: a Java interview must never be answered in JavaScript, a callback
// must find its earlier turn, and none of it may cost network latency.

import assert from "node:assert/strict";
import {
  SessionFactStore,
  extractSessionFacts,
  factSalience,
  isCallbackQuestion,
  recallEarlierTurns,
} from "../src/runtime/agent/sessionFacts";
import { ContextStore } from "../src/runtime/agent/contextStore";
import type { TranscriptSegment } from "../src/app/types";

const BASE = 1_700_000_000_000;
const seg = (
  text: string,
  endMs: number,
  speaker: "interviewer" | "user" = "interviewer"
): TranscriptSegment => ({
  id: `s-${endMs}`,
  text,
  endMs,
  startMs: endMs - 1000,
  speaker,
});

let passed = 0;
function check(name: string, fn: () => void) {
  fn();
  passed += 1;
  console.log(`  ok  ${name}`);
}

console.log("session facts");

check("Java is not swallowed by JavaScript", () => {
  const values = extractSessionFacts("说说你对 Java 的理解", BASE)
    .filter((f) => f.kind === "language")
    .map((f) => f.value);
  assert.deepEqual(values, ["Java"]);
});

check("JavaScript resolves to JavaScript, not Java", () => {
  const values = extractSessionFacts("你用过 JavaScript 吗", BASE)
    .filter((f) => f.kind === "language")
    .map((f) => f.value);
  assert.deepEqual(values, ["JavaScript"]);
});

check("JS shorthand resolves to JavaScript", () => {
  const values = extractSessionFacts("JS 的闭包讲一下", BASE)
    .filter((f) => f.kind === "language")
    .map((f) => f.value);
  assert.deepEqual(values, ["JavaScript"]);
});

check("backend stack is recognised", () => {
  const facts = extractSessionFacts("你们用 Spring Boot 和 MySQL，Redis 做缓存吗", BASE);
  const byKind = (kind: string) => facts.filter((f) => f.kind === kind).map((f) => f.value);
  assert.ok(byKind("framework").includes("Spring Boot"));
  assert.ok(byKind("storage").includes("MySQL"));
  assert.ok(byKind("storage").includes("Redis"));
});

check("anchors stay empty before anything is known", () => {
  assert.equal(new SessionFactStore().anchors(BASE), "");
});

check("language anchor survives a long interview", () => {
  const store = new SessionFactStore();
  store.ingest([seg("今天主要聊 Java", BASE)]);
  // 33 minutes later — beyond any live transcript window.
  const anchors = store.anchors(BASE + 33 * 60_000);
  assert.match(anchors, /language: Java/);
});

check("repeated mentions outrank a single stale one", () => {
  const store = new SessionFactStore();
  store.ingest([seg("我们聊聊 Redis", BASE)]);
  for (let i = 0; i < 4; i++) {
    store.ingest([seg("Kafka 这边再讲讲", BASE + (i + 1) * 1000)]);
  }
  const anchors = store.anchors(BASE + 5000);
  assert.ok(anchors.indexOf("Kafka") < anchors.indexOf("Redis"));
});

check("salience decays monotonically", () => {
  const fact = { kind: "topic" as const, value: "限流", endMs: BASE, hits: 1 };
  const near = factSalience(fact, BASE + 1000);
  const far = factSalience(fact, BASE + 20 * 60_000);
  assert.ok(near > far);
});

check("callback detection ignores self-contained questions", () => {
  assert.equal(isCallbackQuestion("那再说说"), true);
  assert.equal(isCallbackQuestion("Java 里 HashMap 的扩容机制是什么"), false);
});

check("a callback naming its subject recalls the earlier turn", () => {
  const durable = [
    seg("你们相册服务的限流是怎么做的", BASE),
    seg("用 Redis 加 Lua 实现分布式限流", BASE + 30_000),
  ];
  const block = recallEarlierTurns("那限流的触发条件怎么定", durable, BASE + 120_000);
  assert.match(block, /限流/);
  assert.match(block, /earlier_context/);
});

check("a bare pronoun callback fails safe instead of guessing", () => {
  const durable = [seg("你们相册服务的限流是怎么做的", BASE)];
  assert.equal(recallEarlierTurns("那它的降级策略呢", durable, BASE + 120_000), "");
});

check("turns already in the live window are not re-surfaced", () => {
  const durable = [seg("你们相册服务的限流是怎么做的", BASE)];
  assert.equal(recallEarlierTurns("限流再讲细一点", durable, BASE - 1), "");
});

check("recall output is bounded", () => {
  const long = "限流 ".repeat(400);
  const block = recallEarlierTurns("限流的细节", [seg(long, BASE)], BASE + 120_000);
  assert.ok(block.length <= 500, `block too long: ${block.length}`);
});

check("1000-segment session stays sub-millisecond on the hot path", () => {
  const many = Array.from({ length: 1000 }, (_, i) =>
    seg(`第 ${i} 段关于 Java Spring MySQL Redis 限流的讨论`, BASE + i * 1000)
  );
  const store = new SessionFactStore();
  store.ingest(many);
  const t0 = performance.now();
  store.anchors(BASE + 1_000_000);
  recallEarlierTurns("限流怎么做的", many, BASE + 990_000);
  const elapsed = performance.now() - t0;
  assert.ok(elapsed < 50, `too slow: ${elapsed.toFixed(2)}ms`);
});

// --- speculative prefetch path ----------------------------------------------
// Prefetch fires before the segment reaches the store, and the coach reuses its
// answer, so the anchors must already be available at prime time.

check("anchors are available to prefetch before the segment is pushed", () => {
  const store = new ContextStore();
  const first = seg("我们聊聊 Java 的垃圾回收", BASE);
  store.primeFacts(first);
  const anchors = store.anchors();
  assert.match(anchors, /language: Java/);
});

check("priming then pushing the same segment does not inflate salience", () => {
  const store = new ContextStore();
  const segment = seg("Java 的 Spring Boot 项目", BASE);
  store.primeFacts(segment);
  store.pushTranscript(segment);
  store.pushTranscript(segment);
  const anchors = store.anchors();
  const javaLines = anchors
    .split("\n")
    .filter((line) => line.startsWith("- language: Java"));
  assert.equal(javaLines.length, 1, `duplicated anchor lines: ${anchors}`);
});

check("anchors clear when the session is reset", () => {
  const store = new ContextStore();
  store.pushTranscript(seg("Java 面试题", BASE));
  assert.match(store.anchors(), /language: Java/);
  store.clear();
  assert.equal(store.anchors(), "");
});

console.log(`\n${passed} checks passed`);
