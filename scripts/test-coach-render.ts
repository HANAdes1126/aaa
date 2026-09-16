/**
 * Regression guard for the coach card rendering path.
 *
 * Two bugs this locks down:
 *  1. `useAgentRuntime` used to hand only `suggestion.answer` to
 *     `buildCoachMessage`, silently discarding everything else.
 *  2. `bullets` used to be spliced in as body points, which turned a spoken
 *     answer into a flat list — the exact "表达一般般" complaint. They are
 *     reserve 追问 lines now and must render below a labelled separator,
 *     never as part of the answer.
 */
import { formatSuggestionText } from "../src/app/coachMessageFormat";
import type { AssistantSuggestion } from "../src/app/types";

let passed = 0;
let failed = 0;

function check(name: string, condition: boolean, detail = "") {
  if (condition) {
    passed++;
    console.log(`  ✓ ${name}`);
  } else {
    failed++;
    console.log(`  ✗ ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

function suggestion(answer: string, bullets: string[]): AssistantSuggestion {
  return { answer, bullets, clarifyingQuestion: null };
}

console.log("\n== the spoken answer renders as one paragraph ==");
{
  const answer =
    "HashMap 底层是数组加链表：key 算哈希后对数组长度取模定位到桶，所以平均读写是 O(1)；多个 key 撞到同一个桶就叫冲突，用链表串起来，冲突多了会退化成 O(n)。Java 8 之后链表超过 8 会转成红黑树，把最坏情况压到 O(log n)。";
  const text = formatSuggestionText(
    suggestion(answer, ["扩容那次要 rehash 全部元素，单次是 O(n)，摊还下来还是 O(1)。"])
  );
  const lines = text.split("\n");
  check("answer stays first and whole", lines[0] === answer);
  check("answer is not broken into bullet lines", lines[0].split("。").length > 2);
  check("reserve line sits below a labelled separator", lines[1] === "" && lines[2] === "追问可补", lines[2]);
  check("reserve line is marked, not merged into the answer", lines[3]?.startsWith("· "), lines[3]);
  check("no reserve text leaks into line 0", !lines[0].includes("rehash"));
}

console.log("\n== prose-only answer stays untouched ==");
{
  const text = formatSuggestionText(suggestion("有的，我用过 Redis 做分布式锁。", []));
  check("no stray bullet marker", !text.includes("·"), text);
  check("no stray follow-up label", !text.includes("追问可补"), text);
  check("text preserved verbatim", text === "有的，我用过 Redis 做分布式锁。", text);
}

console.log("\n== noisy input is cleaned, not rendered ==");
{
  const text = formatSuggestionText(
    suggestion("  结论。  ", ["  追问一。  ", "   ", "追问二。"])
  );
  const lines = text.split("\n");
  check("answer trimmed", lines[0] === "结论。", lines[0]);
  check("blank bullets dropped", lines.filter((line) => line.startsWith("· ")).length === 2);
  check(
    "layout is answer / gap / label / lines",
    text === "结论。\n\n追问可补\n· 追问一。\n· 追问二。",
    JSON.stringify(text)
  );
}

console.log("\n== the exact flat paragraph from the complaint ==");
{
  // What a provider that ignores the JSON contract produces: one paragraph.
  // This must render as-is — the old fallback split it into a conclusion plus
  // bullets, which is what made it read like a list.
  const flat =
    "哈希表的底层结构，核心是一个数组加链表的组合。数组的每个下标对应一个桶，key 经过哈希函数映射成数组下标。当链表过长时会转成红黑树，比如 Java 的 HashMap。另外超过负载因子会触发扩容。";
  const text = formatSuggestionText(suggestion(flat, []));
  check("single-paragraph answer renders as-is", text === flat);
  check("no bullet markers were synthesised", !text.includes("·"));
}

console.log("\n== behavioural questions coach instead of answering ==");
{
  // The answer is the candidate's own history, so `answer` holds coaching.
  // It must never read as a script: without the headline the candidate could
  // say "这题只能讲你自己的真事" out loud to the interviewer.
  const coaching =
    "这题只能讲你自己的真事，我替不了你。挑一个你真的卡住过的难题，重点说清当时怎么定位、你做了什么取舍。";
  const text = formatSuggestionText({
    answer: coaching,
    bullets: ["一定要给具体数字，比如耗时从多久降到多久。"],
    clarifyingQuestion: null,
    kind: "behavioral",
  });
  const lines = text.split("\n");
  check("headline marks it as coaching, not an answer", lines[0] === "这题要用你自己的经历", lines[0]);
  check("coaching text follows the headline", lines[1] === coaching);
  check("tips are labelled 怎么讲, not 追问可补", lines[3] === "怎么讲", lines[3]);
  check("tip line is marked", lines[4]?.startsWith("· "), lines[4]);
  check("no 追问可补 label leaks into a behavioural card", !text.includes("追问可补"));
}

console.log("\n== design questions show their outline as primary content ==");
{
  const overview = "这是带保底的加权抽奖，核心是权重模型和用户状态分开维护。";
  const points = [
    "奖品配置按稀有度保存基础权重和库存。",
    "用户状态按 user_id 保存幸运值，抽中稀有奖后清零。",
    "抽奖记录持久化请求号和结果，用唯一约束保证幂等。",
    "服务端原子计算结果并更新状态，前端只播放对应动画。",
  ];
  const text = formatSuggestionText({
    answer: overview,
    bullets: points,
    clarifyingQuestion: null,
    kind: "design",
  });
  const lines = text.split("\n");
  check("design overview stays first", lines[0] === overview, lines[0]);
  check("design points use the primary label", lines[2] === "设计思路", lines[2]);
  check("all design points remain visible", lines.filter((line) => line.startsWith("· ")).length === 4);
  check("design points are not labelled as follow-up", !text.includes("追问可补"));
}

console.log("\n== unknown or missing kind still answers ==");
{
  // Backend resolves anything unrecognized to "knowledge"; the frontend must
  // agree, because withholding an answer is the worse failure.
  const answer = "HashMap 底层是数组加链表加红黑树。";
  for (const kind of [undefined, "", "knowledge", "something-else"]) {
    const text = formatSuggestionText({
      answer,
      bullets: [],
      clarifyingQuestion: null,
      ...(kind === undefined ? {} : { kind }),
    });
    check(`kind=${JSON.stringify(kind)} renders as a plain answer`, text === answer, text);
  }
}

console.log(`\n${passed} checks passed, ${failed} failed`);
if (failed > 0) process.exit(1);
