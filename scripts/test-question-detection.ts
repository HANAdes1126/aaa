// Regression tests for interview question detection.
//
// The bug this guards against: "请描述哈希映射（hash map）的底层结构。" produced
// NO answer at all. It has no "?" and no 吗/呢/什么, so both detection paths
// scored it 0 and returned nothing — silently. Run after touching
// interviewLogic.ts or runtime/agent/demand.ts.

import {
  detectQuestionCandidate,
  hasInterviewRequest,
  isLikelyDuplicateTranscript,
} from "../src/app/interviewLogic";
import { TRANSCRIPT_DEDUPE_WINDOW_MS } from "../src/app/constants";
import { detectSttWake } from "../src/runtime/agent/demand";
import type { TranscriptSegment } from "../src/app/types";

let passed = 0;
let failed = 0;

function check(label: string, condition: boolean, detail = "") {
  if (condition) {
    passed += 1;
    console.log(`  ok   ${label}${detail ? ` (${detail})` : ""}`);
  } else {
    failed += 1;
    console.log(`  FAIL ${label}${detail ? ` (${detail})` : ""}`);
  }
}

function seg(text: string, endMs = 1_000): TranscriptSegment {
  return {
    id: `seg-${Math.random().toString(16).slice(2)}`,
    text,
    endMs,
    source: "system",
  } as TranscriptSegment;
}

// Both paths must agree: prefetch candidate (>= 0.88 gate) AND stt wake.
function expectQuestion(text: string) {
  const candidate = detectQuestionCandidate(seg(text), []);
  const wake = detectSttWake(seg(text), "remote", "candidate");
  const conf = candidate ? candidate.confidence.toFixed(2) : "null";
  const kind = wake ? (wake as { kind?: string }).kind ?? "?" : "null";
  check(
    `question: ${text}`,
    !!candidate && candidate.confidence >= 0.88 && !!wake,
    `conf=${conf} wake=${kind}`
  );
}

function expectIgnored(text: string) {
  const candidate = detectQuestionCandidate(seg(text), []);
  const wake = detectSttWake(seg(text), "remote", "candidate");
  const conf = candidate ? candidate.confidence.toFixed(2) : "null";
  check(
    `ignored: ${text}`,
    !candidate && !wake,
    `candidate=${candidate ? conf : "null"} wake=${wake ? "yes" : "null"}`
  );
}

console.log("\n[1] imperative request questions (the reported bug)");
expectQuestion("请描述哈希映射（hash map）的底层结构。");
expectQuestion("请解释一下 Java 的类加载机制");
expectQuestion("请介绍一下 Spring 的 IoC 容器");
expectQuestion("请说说 JVM 的垃圾回收算法");
expectQuestion("请实现一个 LRU 缓存");

console.log("\n[2] verb-initial and X一下 shapes");
expectQuestion("描述一下 HashMap 的扩容过程");
expectQuestion("讲一下双亲委派模型");
expectQuestion("总结一下 Redis 的持久化方案");
expectQuestion("实现一个单例模式");
expectQuestion("说说你对双亲委派模型的理解。");
expectQuestion("手写一下快排");

console.log("\n[3] regressions — shapes that already worked");
expectQuestion("哈希表的底层结构是什么？");
expectQuestion("为什么要重写 equals 和 hashCode？");
expectQuestion("What is the time complexity of quicksort?");
expectQuestion("Can you explain the CAP theorem?");

console.log("\n[4] must NOT fire (filler, setup, own answers)");
expectIgnored("嗯");
expectIgnored("好的");
expectIgnored("能听到吗");
expectIgnored("请稍等一下");
expectIgnored("这个系统的底层结构是由数组和链表组成的");
expectIgnored("HashMap 的底层主要是数组加链表");

console.log("\n[5] hasInterviewRequest predicate");
check("请描述 -> true", hasInterviewRequest("请描述 X"));
check("麻烦解释 -> true", hasInterviewRequest("麻烦解释一下"));
check("你来讲 -> true", hasInterviewRequest("你来讲讲"));
check("描述一下 -> true", hasInterviewRequest("描述一下 X"));
check("plain statement -> false", !hasInterviewRequest("这个系统用的是数组"));
check("请稍等 -> false", !hasInterviewRequest("请稍等一下"));

// The bug this guards against: asking the same question again, minutes later,
// produced NO answer. Duplicate suppression compared text only — no time
// window — so an interviewer re-asking after 230s was swallowed as an STT
// echo. Suppression is for echoes (same audio recognised twice, ~1-2s apart),
// not for re-asking.
console.log("\n[6] transcript duplicate suppression is time-bounded");
const REASK = "说说你遇到的一个难题，然后你是怎么解决的？";

function segAt(text: string, startMs: number, endMs: number, source: "microphone" | "system" = "system"): TranscriptSegment {
  return { id: `seg-${startMs}`, text, startMs, endMs, source } as TranscriptSegment;
}

// STT echo: the same utterance lands again ~2s later. Must still be swallowed,
// otherwise every question renders twice.
check(
  "same text 2s later -> duplicate",
  isLikelyDuplicateTranscript(segAt(REASK, 12_000, 13_000), [segAt(REASK, 10_000, 11_000)])
);

// The actual regression: re-asked after ~4 minutes. Must NOT be swallowed.
check(
  "same text 230s later -> NOT duplicate",
  !isLikelyDuplicateTranscript(segAt(REASK, 231_000, 232_000), [segAt(REASK, 1_000, 2_000)]),
  `window=${TRANSCRIPT_DEDUPE_WINDOW_MS}ms`
);

// Boundary: just inside vs just outside the window.
check(
  "just inside window -> duplicate",
  isLikelyDuplicateTranscript(
    segAt(REASK, 2_000 + TRANSCRIPT_DEDUPE_WINDOW_MS - 500, 2_000 + TRANSCRIPT_DEDUPE_WINDOW_MS),
    [segAt(REASK, 1_000, 2_000)]
  )
);
check(
  "just outside window -> NOT duplicate",
  !isLikelyDuplicateTranscript(
    segAt(REASK, 2_000 + TRANSCRIPT_DEDUPE_WINDOW_MS + 500, 2_000 + TRANSCRIPT_DEDUPE_WINDOW_MS + 1_500),
    [segAt(REASK, 1_000, 2_000)]
  )
);

// Different capture sources are always distinct utterances.
check(
  "different source -> NOT duplicate even immediately",
  !isLikelyDuplicateTranscript(segAt(REASK, 12_000, 13_000, "microphone"), [
    segAt(REASK, 10_000, 11_000, "system"),
  ])
);

// Substring containment is time-bounded too, not just exact equality.
check(
  "contained text 230s later -> NOT duplicate",
  !isLikelyDuplicateTranscript(segAt(REASK, 231_000, 232_000), [
    segAt("说说你遇到的一个难题", 1_000, 2_000),
  ])
);

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
