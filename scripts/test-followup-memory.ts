// Probes what the coach model actually SEES when the interviewer asks a
// follow-up question. Run: npx tsx scripts/test-followup-memory.ts
//
// The interesting failure mode is not "the model forgot" — it is "the prompt
// never contained the earlier turn", which no amount of model quality can fix.
// So this script prints the assembled prompt rather than asserting on internals.

import { ContextStore } from "../src/runtime/agent/contextStore";
import { buildAgentPrompt } from "../src/runtime/agent/prompt";
import type { TranscriptSegment } from "../src/app/types";

const BASE = 1_700_000_000_000;
const WINDOW_MS = 120_000; // matches AGENT_CONTEXT_WINDOW_MS in runtime.ts

let idx = 0;
const seg = (
  text: string,
  endMs: number,
  speaker: "interviewer" | "user" = "interviewer"
): TranscriptSegment => ({
  id: `s-${idx++}-${endMs}`,
  text,
  endMs,
  startMs: endMs - 8_000,
  speaker,
});

/** A realistic interview opening: the stack gets established early. */
function seedSession(): ContextStore {
  const store = new ContextStore();
  store.setSessionConfig({ kind: "remote", audioSource: "system", goal: "" });
  store.pushTranscript(seg("我们今天主要聊 Java 后端，你做过的项目里用 Redis 多吗？", BASE));
  store.pushTranscript(seg("用得挺多的，主要做缓存和分布式锁。", BASE + 10_000, "user"));
  store.pushTranscript(seg("那你说说 HashMap 的底层结构是什么？", BASE + 20_000));
  store.pushTranscript(
    seg(
      "HashMap 底层是数组加链表加红黑树，key 先算哈希再取模定位到桶，冲突了挂链表，链表超过 8 转成红黑树。",
      BASE + 30_000,
      "user"
    )
  );
  return store;
}

function promptFor(store: ContextStore, followUp: string, atMs: number) {
  store.pushTranscript(seg(followUp, atMs));
  const snapshot = store.snapshot(WINDOW_MS);
  return {
    prompt: buildAgentPrompt(
      {
        kind: "stt_question",
        reason: "question_detected",
        evidence: [followUp],
      } as never,
      snapshot
    ).text,
    snapshot,
  };
}

function section(title: string) {
  console.log(`\n${"=".repeat(76)}\n${title}\n${"=".repeat(76)}`);
}

function showPrompt(prompt: string) {
  // Trim the invariant preamble so the output shows only what carries memory.
  const start = prompt.indexOf("Session anchors") ;
  const anchorIdx = start >= 0 ? start : prompt.indexOf("- language:");
  const body = anchorIdx >= 0 ? prompt.slice(anchorIdx) : prompt;
  console.log(body.split("\n").slice(0, 24).join("\n"));
}

let failures = 0;
function expect(name: string, ok: boolean, detail = "") {
  console.log(`${ok ? "  ok  " : "  FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
}

// --- Case A: follow-up inside the live window -------------------------------
section("A. 追问落在 120s 窗口内（间隔 40 秒）");
{
  const store = seedSession();
  const { prompt, snapshot } = promptFor(store, "那它的时间复杂度是多少？", BASE + 70_000);
  const recent = snapshot.recentTranscript.length;
  const hasEarlier = /HashMap 底层是数组/.test(prompt);
  console.log(`recentTranscript=${recent} 段, anchors=${JSON.stringify(snapshot.anchors.split("\n")[0] ?? "")}`);
  showPrompt(prompt);
  expect("前一轮问答仍在 Recent transcript 中", hasEarlier, hasEarlier ? "" : "模型看不到上一轮");
  expect("anchors 已建立 Java/Redis", /language: Java/.test(snapshot.anchors));
}

// --- Case B: follow-up OUTSIDE the live window ------------------------------
section("B. 追问落在窗口外（间隔 6 分钟，仍在 30 分钟保留期内）");
{
  const store = seedSession();
  // 6 minutes of unrelated chatter pushes the HashMap turn out of the window.
  for (let i = 0; i < 6; i += 1) {
    store.pushTranscript(seg(`你平时怎么做 Code Review 的？第 ${i + 1} 轮闲聊。`, BASE + 60_000 + i * 60_000));
    store.pushTranscript(seg("一般会看命名、边界条件和测试覆盖。", BASE + 65_000 + i * 60_000, "user"));
  }
  const { prompt, snapshot } = promptFor(
    store,
    "刚才聊的 HashMap，冲突多了会退化成什么？",
    BASE + 430_000
  );
  const recent = snapshot.recentTranscript.length;
  // The recall block is an `<earlier_context>` XML element — assert on the tag,
  // not on prose that happens to describe it.
  const recalled = /<earlier_context/.test(prompt);
  const recentBlock = prompt.slice(
    prompt.indexOf("Recent transcript:"),
    prompt.indexOf("<earlier_context")
  );
  const hasHashMap = /HashMap/.test(prompt);
  console.log(`recentTranscript=${recent} 段, durable=${snapshot.durableTranscript.length} 段`);
  showPrompt(prompt);
  expect("Recent transcript 块内已不含 HashMap 那轮", !/HashMap 底层是数组/.test(recentBlock));
  expect("long-range recall 命中（<earlier_context> 块）", recalled);
  expect("召回块里确实有 HashMap", hasHashMap);
}

// --- Case C: bare-pronoun callback ------------------------------------------
section("C. 裸代词追问（那它的根本原因呢 / 无任何关键词）");
{
  const store = seedSession();
  for (let i = 0; i < 4; i += 1) {
    store.pushTranscript(seg(`聊聊你们团队的发布流程，第 ${i + 1} 段。`, BASE + 60_000 + i * 60_000));
    store.pushTranscript(seg("我们是灰度发布加回滚预案。", BASE + 66_000 + i * 60_000, "user"));
  }
  const { prompt, snapshot } = promptFor(store, "那它的根本原因呢？", BASE + 320_000);
  const recalled = /Earlier in this session|Earlier conversation/i.test(prompt);
  console.log(`anchors=${JSON.stringify(snapshot.anchors.replace(/\n/g, " | "))}`);
  showPrompt(prompt);
  console.log(
    recalled
      ? "  → recall 命中"
      : "  → recall 未命中（fail-safe：靠 anchors 提供主题，模型只能泛答）"
  );
}

// --- Case D: anchors survive a long session ---------------------------------
section("D. 长会话 40 分钟后 anchors 是否还在（30 分钟淘汰线）");
{
  const store = seedSession();
  for (let i = 0; i < 12; i += 1) {
    const at = BASE + 60_000 + i * 200_000; // ~40 minutes total
    store.pushTranscript(seg(`第 ${i + 1} 段无关话题，聊监控告警。`, at));
    store.pushTranscript(seg("我们用 Prometheus 加 Grafana。", at + 5_000, "user"));
  }
  const anchors = store.anchors();
  const durable = store.snapshot(WINDOW_MS).durableTranscript.length;
  console.log(`anchors=${JSON.stringify(anchors.replace(/\n/g, " | "))}`);
  console.log(`durableTranscript=${durable} 段（应已被 30 分钟淘汰线裁掉早期内容）`);
  expect(
    "Java 锚点仍存活（事实记忆独立于 transcript 淘汰）",
    /language: Java/.test(anchors),
    anchors ? "" : "anchors 为空"
  );
}

// --- Case E: the coach remembers its OWN previous answer -------------------
// The silent-candidate case from the 2026-09-11 bench: the candidate answers
// "嗯，好的，我了解的。" instead of reading the card, so the transcript carries
// nothing. Before this existed the follow-up had no way to know what the coach
// already said and degraded into a generic answer.
section("E. 候选人只说「嗯，好的」时，教练记得自己上一轮答了什么");
{
  const store = new ContextStore();
  store.setSessionConfig({ kind: "remote", audioSource: "system", goal: "" });
  store.pushTranscript(seg("你们的限流是怎么做的？", BASE));
  store.pushCoachTurn({
    question: "你们的限流是怎么做的？",
    answer:
      "限流分三层：网关做单机令牌桶粗筛，Redis 用 Lua 脚本做集群精筛，滑动窗口用 ZSET 存时间戳。",
    atMs: BASE + 5_000,
  });
  // The candidate does NOT read the card out loud.
  store.pushTranscript(seg("嗯，好的，我了解了。", BASE + 12_000, "user"));

  const { prompt, snapshot } = promptFor(store, "刚才那个方案在分布式下有什么问题？", BASE + 40_000);
  const hasBlock = /<previous_coach_answers/.test(prompt);
  const hasSubstance = /ZSET/.test(prompt) && /Redis/.test(prompt);
  console.log(`previousCoachTurns=${snapshot.previousCoachTurns.length} 段`);
  const start = prompt.indexOf("<previous_coach_answers");
  const end = prompt.indexOf("</previous_coach_answers>");
  console.log(start >= 0 ? prompt.slice(start, end + 25) : "  (块未出现)");
  expect("prompt 里出现 <previous_coach_answers> 块", hasBlock);
  expect("块里带着上一轮的具体方案（Redis / ZSET）", hasSubstance);
  expect(
    "transcript 里确实没有这些内容（证明记忆真的来自 coach turn）",
    !snapshot.recentTranscript.some((s) => /ZSET/.test(s.text))
  );
}

// --- Case G: interviewer-only audio ---------------------------------------
// With the mic off the candidate's voice never reaches the transcript, so it is
// literally two interviewer lines back to back. This is the mode where coach-turn
// memory matters most: it becomes the ONLY source of "what was already answered",
// where before there was none at all.
section("G. 只录对面（mic 关）：transcript 只剩面试官，教练记忆是唯一前文来源");
{
  const store = new ContextStore();
  store.setSessionConfig({ kind: "remote", audioSource: "system", goal: "" });
  store.pushTranscript(seg("你们的限流是怎么做的？", BASE));
  // Nothing from the candidate: the mic is off.
  store.pushCoachTurn({
    question: "你们的限流是怎么做的？",
    answer:
      "限流分三层：网关做单机令牌桶粗筛，Redis 用 Lua 脚本做集群精筛，滑动窗口用 ZSET 存时间戳。",
    atMs: BASE + 5_000,
  });

  const { prompt, snapshot } = promptFor(store, "刚才那个方案在分布式下有什么问题？", BASE + 40_000);
  const from = prompt.indexOf("Recent transcript:");
  const to = prompt.indexOf("<previous_coach_answers");
  console.log(`Recent transcript 块：${JSON.stringify(prompt.slice(from, to).replace(/\n/g, " ⏎ "))}`);
  expect(
    "transcript 里没有候选人任何一句话（mic 关，声音从未入库）",
    !snapshot.recentTranscript.some((s) => s.speaker === "user")
  );
  expect("面试官两句话直接相邻", snapshot.recentTranscript.length === 2);
  expect("追问仍然拿到上一轮方案（ZSET / Redis）", /ZSET/.test(prompt) && /Redis/.test(prompt));
  expect(
    "Audio source 标注为 system",
    prompt.split("\n").some((line) => line.trim() === "Audio source: system")
  );
  // The 2026-09-11 A/B: with the mic off the remembered answer is an assumption,
  // not a record. Plain wording made v4.1-flash answer an interviewer question
  // about Sentinel with the Redis design it had suggested itself.
  expect(
    "只录对面时块的 note 声明「无从得知候选人是否照说」",
    /CANNOT know whether the candidate actually said/.test(prompt)
  );
  expect("并带上「以面试官为准」的护栏", /the interviewer wins/.test(prompt));
}

// --- Case H: same memory, mic on -------------------------------------------
// Controls for Case G: when the candidate IS transcribed the remembered answer is
// corroborated by the transcript, so the hedge must NOT appear — it would make the
// model distrust a memory that is actually reliable.
section("H. 对面 + 我：不该加「无从得知」的免责声明");
{
  const store = new ContextStore();
  store.setSessionConfig({ kind: "remote", audioSource: "microphone", goal: "" });
  store.pushTranscript(seg("你们的限流是怎么做的？", BASE));
  store.pushTranscript(seg("我们用网关做单机桶，Redis 做集群限流。", BASE + 8_000, "user"));
  store.pushCoachTurn({
    question: "你们的限流是怎么做的？",
    answer: "限流分三层：网关做单机令牌桶粗筛，Redis 用 Lua 脚本做集群精筛。",
    atMs: BASE + 5_000,
  });

  const { prompt } = promptFor(store, "刚才那个方案在分布式下有什么问题？", BASE + 40_000);
  expect("块仍然注入", /<previous_coach_answers/.test(prompt));
  expect(
    "但不带免责声明（候选人的话在 transcript 里，记忆可信）",
    !/CANNOT know whether the candidate actually said/.test(prompt)
  );
  expect("护栏用普通版", !/the interviewer wins/.test(prompt));
}

// --- Case F: bounds + reset ------------------------------------------------
section("F. 上限与重置");
{
  const store = new ContextStore();
  for (let i = 0; i < 6; i += 1) {
    store.pushCoachTurn({ question: `Q${i}`, answer: `A${i} `.repeat(200), atMs: BASE + i });
  }
  const turns = store.snapshot(WINDOW_MS).previousCoachTurns;
  const chars = turns.reduce((n, t) => n + t.answer.length, 0);
  console.log(`6 条后保留 ${turns.length} 条, 总长 ${chars} 字符, 末条为 ${turns[turns.length - 1].question}`);
  expect("最多保留 3 轮", turns.length === 3, `实际 ${turns.length}`);
  expect("总长受 3000 字符预算约束", chars <= 3_200, `实际 ${chars}`);
  expect("保留的是最近三轮", turns[turns.length - 1].question === "Q5");

  store.clear();
  expect("clear() 后教练记忆一并清空（新建会话不串台）", store.snapshot(WINDOW_MS).previousCoachTurns.length === 0);
}

console.log(`\n${failures === 0 ? "all probes ok" : `${failures} probe(s) failed`}`);
