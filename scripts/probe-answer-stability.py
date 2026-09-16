"""Sample the same question N times and measure how stable the answer shape is.

A single sample tells you what the model *can* say, not what it *will* say.
This repeats one question and reports where a given signal lands — in `answer`
(the candidate reads it out) versus `bullets` (only surfaces if the interviewer
digs further). For a live coach that distinction decides whether the point is
spoken at all.

Signals are plain substrings grouped per concept, because the model paraphrases
freely ("冲突" / "矛盾" / "兜不住" all mean the same thing here).

Usage:
    python3 scripts/probe-answer-stability.py --runs 8
    python3 scripts/probe-answer-stability.py --runs 5 --model deepseek-v4-pro-ioa
"""

import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "ask_one", Path(__file__).resolve().parent / "ask-one.py"
)
_ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ask)

ROOT = Path(__file__).resolve().parent.parent

QUESTION = (
    "给你1000个人抢100个奖品，且每个中奖概率是30%，"
    "请问你怎么去设计数据结构，表格，和接口啥的"
)

# Signals worth having in the spoken answer.
#
# "点破矛盾" used to live here and was driven from 3/8 to 20/20 before the user
# pointed out it should not be there at all: a 30% gate in front of 100 units
# is a filter followed by a limit, not a contradiction, and opening with it
# burns the one sentence the interviewer is listening to. It now lives in
# ANTI_SIGNALS. Kept as a comment because deleting it loses the lesson.
SIGNALS = {
    "归类": ["超卖", "超发", "秒杀", "抢购", "限量", "库存竞争", "先到先得"],
    "Redis 原子": ["lua", "原子", "incrby", "decr"],
    "唯一索引": ["唯一索引", "unique"],
    "幂等": ["幂等", "setnx", "去重"],
}

# Phrases that must NOT open the answer. Measured on the opening clause only —
# mentioning a caveat later is fine, leading with one is throat-clearing.
#
# Bare substring matching produced a false positive immediately: "插入成功就是
# 中奖、冲突就是重复" is a database unique-key conflict, not a challenge to the
# premise, and it sat mid-sentence in an otherwise perfect answer. So each
# term carries the technical senses that must NOT count.
ANTI_SIGNALS = {
    "冲突": ["唯一", "主键", "索引", "键冲突", "写冲突", "版本"],
    "矛盾": [],
    "对不上": [],
    "跟产品对齐": [],
    "先要说清": [],
    "二选一": [],
    "不可能同时": [],
    "这个前提": [],
    "前提是": [],
}

# Hedging happens at the very start or not at all; scanning a whole sentence
# catches ordinary technical vocabulary further in.
OPENING_CHARS = 30


def opens_by_hedging(answer: str) -> bool:
    opening = answer[:OPENING_CHARS]
    for term, technical in ANTI_SIGNALS.items():
        if term not in opening:
            continue
        if any(sense in opening for sense in technical):
            continue
        return True
    return False


def locate(parsed: dict) -> dict:
    """Where each signal landed: spoken, follow-up only, or missing."""
    answer = parsed["answer"].lower()
    bullets = " ".join(parsed["bullets"]).lower()
    placement = {}
    for name, terms in SIGNALS.items():
        in_answer = any(t.lower() in answer for t in terms)
        in_bullets = any(t.lower() in bullets for t in terms)
        placement[name] = (
            "answer" if in_answer else "bullets" if in_bullets else "missing"
        )
    return placement


def first_sentence(text: str) -> str:
    for sep in ("。", "；", "\n"):
        if sep in text:
            return text.split(sep, 1)[0]
    return text[:80]


def one_run(model: str) -> dict:
    result = _ask.call(model, QUESTION)
    parsed = _ask.parse(result["raw"])
    head = first_sentence(parsed["answer"])
    return {
        "total_ms": result["total_ms"],
        "kind": parsed["kind"],
        "chars": len(parsed["answer"]),
        "bullets": len(parsed["bullets"]),
        "has_clarifying": bool(parsed["clarifying"]),
        "placement": locate(parsed),
        # Scoped to the opening on purpose: a caveat further down is
        # legitimate, leading with one is the failure.
        "hedged_open": opens_by_hedging(parsed["answer"]),
        "head": head,
        "answer": parsed["answer"],
    }


def self_check() -> int:
    """Verify the hedging detector before spending money on model calls.

    Added after the detector fired on "插入成功就是中奖、冲突就是重复" — a
    database unique-key conflict inside an otherwise ideal answer. Widening
    the term list is the wrong reflex; what the checker needs is proof that it
    still catches real hedging after every tweak.
    """
    cases = [
        ("这两个约束是冲突的：1000 人每人 30% 中奖", True),
        ("这个前提是矛盾的，期望 300 人但只有 100 个奖品", True),
        ("先跟产品对齐口径，再说怎么设计", True),
        ("这里两个条件对不上，我按库存为准", True),
        # Technical senses that must not count as hedging:
        ("这本质就是带唯一约束的去重问题，插入冲突就是重复", False),
        ("这其实是限量秒杀抽奖的经典问题，先过概率再抢库存", False),
        ("一张中奖记录表，user_id 唯一索引，写冲突就当未中奖", False),
        ("用乐观锁版本号解决并发写冲突", False),
    ]
    failures = 0
    for text, expected in cases:
        actual = opens_by_hedging(text)
        if actual != expected:
            failures += 1
            print(f"  FAIL 期望={'质疑' if expected else '直答'} 实际="
                  f"{'质疑' if actual else '直答'} | {text[:34]}")
    print(f"判定自检 {len(cases) - failures}/{len(cases)}"
          + ("" if failures == 0 else "  <-- 先修判定再跑模型"))
    return failures


if __name__ == "__main__":
    runs = 8
    model = _ask.DEFAULT_MODEL
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    print(f"{model} x {runs} 次\n")
    # A broken scorer invalidates every number below it, so prove it works
    # before paying for any completions.
    if self_check() > 0:
        sys.exit(1)
    print()
    # Serial: concurrent bursts against this gateway have produced fake
    # slowdowns before, and latency is one of the things being measured.
    results = [one_run(model) for _ in range(runs)]

    print(f"{'#':>2}  {'ms':>5}  {'字':>4}  {'补':>2}  {'开头':>4}  信号落点")
    for index, row in enumerate(results, 1):
        marks = " ".join(
            f"{name}={row['placement'][name]}" for name in SIGNALS
        )
        print(
            f"{index:>2}  {row['total_ms']:>5}  {row['chars']:>4}  "
            f"{row['bullets']:>2}  {'质疑' if row['hedged_open'] else '直答':>4}  {marks}"
        )

    print("\n=== 落点分布 ===")
    for name in SIGNALS:
        counts = {"answer": 0, "bullets": 0, "missing": 0}
        for row in results:
            counts[row["placement"][name]] += 1
        print(
            f"  {name:10s} 正文 {counts['answer']}/{runs}  "
            f"仅补充 {counts['bullets']}/{runs}  未提 {counts['missing']}/{runs}"
        )

    hedged = sum(r["hedged_open"] for r in results)
    print(f"\n开头质疑前提 {hedged}/{runs}  (必须为 0 — 详见脚本顶部注释)")
    if hedged:
        for row in results:
            if row["hedged_open"]:
                print(f"    {row['head'][:70]}")

    lat = [r["total_ms"] for r in results]
    chars = [r["chars"] for r in results]
    print(
        f"延迟 均 {statistics.mean(lat):.0f}ms  最慢 {max(lat)}ms  "
        f"|  字数 均 {statistics.mean(chars):.0f}  区间 {min(chars)}-{max(chars)}"
    )
    print(f"反问出现 {sum(r['has_clarifying'] for r in results)}/{runs}")

    target = ROOT / ".workbuddy" / "answer_stability.json"
    json.dump(results, open(target, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {target.relative_to(ROOT)}")
