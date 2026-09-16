// Probe: what happens when the 15s hard cap splits one long question.
//
// The VAD force-flush cuts a monologue into ~15s pieces. This asks whether the
// 8s merge window in detectQuestionCandidateWithContext can still stitch the
// pieces back into one question, or whether the coach ends up answering half
// a sentence.

import { detectQuestionCandidateWithContext } from "../src/app/interviewLogic";
import type { TranscriptSegment } from "../src/app/types";

function seg(id: string, text: string, startMs: number, endMs: number): TranscriptSegment {
  return {
    id,
    source: "system",
    speaker: "interviewer",
    text,
    startMs,
    endMs,
    start_ms: startMs,
    end_ms: endMs,
  } as TranscriptSegment;
}

// One continuous 21-second question, cut by the cap at 15s. The trailing gap
// between the two pieces is 6s, inside the 8s merge window.
const shortGapHead =
  "我们来聊一个系统设计的问题。假设你现在要设计一个支持千万级用户的短链接服务，需要考虑存储的选型、缓存的策略，还有高可用的方案应该怎么做";
const shortGapTail = "，另外如果这个服务还要支持自定义短码和过期时间，你会怎么设计？";

// Same question but the interviewer keeps going for 27s, so the second piece
// only lands 12s after the first — outside the 8s merge window.
const longGapHead =
  "我们来聊一个系统设计的问题。假设你现在要设计一个支持千万级用户的短链接服务，需要考虑存储的选型、缓存的策略，高可用的方案，还有怎么防止短码被遍历，以及如果流量突然涨十倍你要怎么扩容";
const longGapTail = "，另外如果这个服务还要支持自定义短码和过期时间，你会怎么设计？";

function run(label: string, head: string, tail: string, gapMs: number) {
  const headEnd = 15_000;
  const tailEnd = headEnd + gapMs;
  const headSeg = seg("a", head, 0, headEnd);
  const tailSeg = seg("b", tail, headEnd, tailEnd);

  console.log(`\n=== ${label} (两段间隔 ${gapMs / 1000}s) ===`);

  const headCandidate = detectQuestionCandidateWithContext(headSeg, [headSeg]);
  console.log(
    `前半段单独判定: ${headCandidate ? `命中 conf=${headCandidate.confidence.toFixed(2)} reason=${headCandidate.reason}` : "未命中"}`
  );

  const tailCandidate = detectQuestionCandidateWithContext(tailSeg, [headSeg, tailSeg]);
  if (!tailCandidate) {
    console.log("后半段到达时: 未命中 → 长句不触发答案");
    return;
  }

  const merged = tailCandidate.text.includes(head.slice(0, 12));
  console.log(`后半段到达时: conf=${tailCandidate.confidence.toFixed(2)} reason=${tailCandidate.reason}`);
  console.log(`  教练拿到的是: ${merged ? "合并后的完整问题" : "只有后半句（半句话）"}`);
  console.log(`  文本: ${tailCandidate.text.slice(0, 60)}…`);
}

run("说 21 秒", shortGapHead, shortGapTail, 6_000);
run("说 27 秒", longGapHead, longGapTail, 12_000);
