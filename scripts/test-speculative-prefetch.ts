import { speculativeSimilarity, hasPolarityFlip } from "../src/app/speculativePrefetch";

type Expectation = {
  pair: [string, string];
  /** 期望复用：相似度应 ≥ 0.78 且不判为翻转 */
  reuse: boolean;
};

const cases: Expectation[] = [
  { pair: ["你们项目里限流是怎么实现的", "你们项目里限流是怎么实现的吗"], reuse: true },
  { pair: ["介绍一下你的限流方案", "能介绍一下你的限流方案吗"], reuse: true },
  // 反义词 / 否定翻转 → 不可复用
  { pair: ["你项目里最好的设计是什么", "你项目里最差的设计是什么"], reuse: false },
  { pair: ["说说这个方案的优点", "说说这个方案的缺点"], reuse: false },
  { pair: ["what is the best approach", "what is the worst approach"], reuse: false },
  { pair: ["你会用 Redis 吗", "你不会用 Redis 吗"], reuse: false },
  { pair: ["你了解这个机制吗", "你不了解这个机制吗"], reuse: false },
  // 完全不同的问题 → 不可复用
  { pair: ["介绍一下你自己", "系统设计题怎么做"], reuse: false },
];

const REUSE_THRESHOLD = 0.78;
let failures = 0;

for (const { pair, reuse } of cases) {
  const [a, b] = pair;
  const sim = speculativeSimilarity(a, b);
  const flip = hasPolarityFlip(a, b);
  const wouldReuse = sim >= REUSE_THRESHOLD && !flip;

  const ok = wouldReuse === reuse;
  if (!ok) failures += 1;
  const mark = ok ? "PASS" : "FAIL";
  console.log(
    `${mark}  sim=${sim.toFixed(2)} flip=${flip} expect_reuse=${reuse}  [${a}] vs [${b}]`
  );
}

// 负向补充：断言纯逻辑函数可被 import 且无副作用（smoke）
if (typeof speculativeSimilarity !== "function" || typeof hasPolarityFlip !== "function") {
  failures += 1;
  console.log("FAIL  exported functions missing");
}

if (failures > 0) {
  console.log(`\n${failures} assertion(s) failed`);
  process.exit(1);
}

console.log("\nall speculative-prefetch assertions passed");
