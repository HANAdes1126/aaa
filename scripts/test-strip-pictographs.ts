/**
 * Coloured icons on the coach card are a tell.
 *
 * During a live interview the candidate is often sharing their screen, and a ✅
 * mid-sentence is something no notes app or IDE produces — it reads instantly
 * as "these words came from a tool". The model is told not to emit them, but a
 * prompt is a request, so the UI strips them too. This locks both halves of
 * that behaviour, and in particular guards the failure mode that matters more
 * than over-stripping: eating legitimate technical notation.
 *
 * Run: ./node_modules/.bin/tsx scripts/test-strip-pictographs.ts
 */
import { formatSuggestionText, stripPictographs } from "../src/app/coachMessageFormat.ts";
import type { AssistantSuggestion } from "../src/app/types.ts";

let failures = 0;

function expect(label: string, ok: boolean, detail = "") {
  if (!ok) failures += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${label}${detail ? `  ← ${detail}` : ""}`);
}

function eq(label: string, actual: string, want: string) {
  expect(label, actual === want, actual === want ? "" : `得到 ${JSON.stringify(actual)}`);
}

function section(title: string) {
  console.log(`\n${title}`);
}

function suggestion(partial: Partial<AssistantSuggestion>): AssistantSuggestion {
  return {
    answer: "",
    bullets: [],
    clarifyingQuestion: null,
    kind: "knowledge",
    ...partial,
  };
}

section("1. 剥离彩色图标");
eq("行首的 ✅", stripPictographs("✅ 方案可行"), "方案可行");
eq("句中的 ✅", stripPictographs("这个方案 ✅ 可行"), "这个方案 可行");
eq("❌ 与 ⚠️", stripPictographs("❌ 不行，⚠️ 有风险"), "不行，有风险");
eq("带变体选择符的 ⚠️", stripPictographs("⚠️ 注意"), "注意");
eq("彩色圆点", stripPictographs("🔴 高优先级 🟢 低优先级"), "高优先级 低优先级");
eq("表情与手势", stripPictographs("搞定 🎉 👍"), "搞定");
eq("方块符号", stripPictographs("■ 第一点 ▶ 第二点"), "第一点 第二点");

section("2. 不能误伤技术符号（这比漏剥更严重）");
eq("箭头表示演化", stripPictographs("复杂度从 O(1) → O(n)"), "复杂度从 O(1) → O(n)");
eq("我们自己的项目符号", stripPictographs("· 第一条"), "· 第一条");
eq("数学比较符", stripPictographs("要求 QPS ≥ 1000 且延迟 ≤ 50ms"), "要求 QPS ≥ 1000 且延迟 ≤ 50ms");
eq("约等于与无穷", stripPictographs("≈ 2 倍，最坏 ∞"), "≈ 2 倍，最坏 ∞");
eq(
  "中文标点与括号",
  stripPictographs("数组 + 链表（JDK 8 起转红黑树）；查询 O(1)。"),
  "数组 + 链表（JDK 8 起转红黑树）；查询 O(1)。"
);
eq("代码里的符号", stripPictographs("if (a && b) { return c ?? d; }"), "if (a && b) { return c ?? d; }");
eq("百分号与货币", stripPictographs("命中率 99.9%，成本 ¥120"), "命中率 99.9%，成本 ¥120");

section("3. 代码缩进必须原样保留（2026-09-12 真实 bug）");
{
  // The first version trimmed every line and collapsed runs of spaces, so a
  // Python answer arrived flush-left and would not run. Indentation is content.
  const py = [
    "def heapify(arr, n, i):",
    "    largest = i",
    "    left = 2 * i + 1",
    "    if left < n and arr[left] > arr[largest]:",
    "        largest = left",
    "    return arr",
  ].join("\n");
  eq("无图标的代码原样返回", stripPictographs(py), py);

  const stripped = stripPictographs(`✅ 堆排序实现：\n${py}`);
  expect("图标被剥掉", !/✅/u.test(stripped));
  expect("四格缩进仍在", stripped.includes("\n    largest = i"));
  expect("八格缩进仍在", stripped.includes("\n        largest = left"));

  const tabs = "function f() {\n\tif (x) {\n\t\treturn 1;\n\t}\n}";
  eq("Tab 缩进原样返回", stripPictographs(tabs), tabs);
  eq(
    "对齐用的多空格不被压缩",
    stripPictographs("const a   = 1;\nconst bbb = 2;"),
    "const a   = 1;\nconst bbb = 2;"
  );
  eq("代码块围栏", stripPictographs("```python\n    pass\n```"), "```python\n    pass\n```");
}

section("4. 剥离后的空白收拾干净");
eq("不留双空格", stripPictographs("方案 ✅ 可行"), "方案 可行");
eq("行首不留空格", stripPictographs("✅  可行"), "可行");
eq("行尾不留空格", stripPictographs("可行 ✅"), "可行");
eq("多行各自清理", stripPictographs("✅ 第一行\n❌ 第二行"), "第一行\n第二行");
expect(
  "重复调用结果稳定（正则无 lastIndex 残留）",
  stripPictographs("✅ 可行") === stripPictographs("✅ 可行")
);

section("4. 空值与边界");
eq("空字符串", stripPictographs(""), "");
eq("纯图标变成空", stripPictographs("✅"), "");
eq("没有图标时原样返回", stripPictographs("正常一句话"), "正常一句话");

section("5. 走完整渲染路径");
{
  const text = formatSuggestionText(
    suggestion({
      answer: "✅ 哈希表底层是数组加链表，JDK 8 起会转红黑树。",
      bullets: ["⚠️ 并发下要用 ConcurrentHashMap", "🔴 扩容是 O(n)"],
    })
  );
  expect("卡片正文里没有图标", !/[✅⚠🔴]/u.test(text), text.slice(0, 60));
  expect("内容本身保留", /数组加链表/.test(text) && /ConcurrentHashMap/.test(text));
  expect("没有残留的双空格", !/ {2}/.test(text), JSON.stringify(text));
}
{
  const text = formatSuggestionText(
    suggestion({
      kind: "behavioral",
      answer: "🎯 你要挑一个自己真的做过决策的项目。",
      bullets: ["👍 一定要给一个具体数字"],
    })
  );
  expect("行为题同样剥离", !/[🎯👍]/u.test(text), text.slice(0, 60));
  expect("行为题抬头仍在", /这题要用你自己的经历/.test(text));
}
{
  const text = formatSuggestionText(
    suggestion({ answer: "复杂度从 O(n) → O(log n)，因为转成了红黑树。" })
  );
  expect("技术箭头没被吃掉", /O\(n\) → O\(log n\)/.test(text), text);
}

console.log(`\n${failures === 0 ? "all probes ok" : `${failures} probe(s) failed`}`);
process.exit(failures === 0 ? 0 : 1);
