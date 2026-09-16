"""Can the coach hand over runnable code, and does it survive the UI?

Three things the user asked about on 2026-09-12:
  1. code indentation — deepseek v4/v4.1 answers arrived flush-left
  2. spoken coding questions — "手写一个快排" through the voice path
  3. extension requirements — "用了优先队列，再手写堆排序"

The indentation bug had two possible homes: the model emitting flat code, or
the UI stripper eating leading whitespace. The stripper is fixed and unit
tested; this measures the model side against the REAL system prompt, so a pass
here means the whole path is clean.

Run: cargo test dump_interview_prompt_smoke -- --nocapture   (refresh the prompt)
     python3 scripts/bench-coding-answer.py
"""

import json
import re
from importlib import util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent

_spec = util.spec_from_file_location("bench_followup", ROOT / "scripts" / "bench-followup.py")
_bf = util.module_from_spec(_spec)
_spec.loader.exec_module(_bf)
MODELS, call, parse = _bf.MODELS, _bf.call, _bf.parse

SYSTEM = open("/tmp/interview_prompt.txt", encoding="utf-8").read()
assert 'kind = "coding"' in SYSTEM, "prompt 未重新导出，先跑 dump_interview_prompt_smoke"

# `kind` is what drives the shape now, so every case declares the label it
# should get. The boundary cases matter more than the obvious ones: a 思路 question
# must stay knowledge, and a war story that happens to mention code must stay
# behavioral, otherwise the coach starts inventing code for an experience question.
CASES = [
    {
        "key": "spoken_quicksort",
        "label": "口述手撕题 · 手写快排",
        "q": "来手写一个快速排序吧，用 Java。",
        "kind": "coding",
        "want_code": True,
        "want": ["partition", "pivot", "基准"],
    },
    {
        "key": "spoken_lru",
        "label": "口述手撕题 · 实现 LRU",
        "q": "实现一个 LRU 缓存，说一下思路然后写出来。",
        "kind": "coding",
        "want_code": True,
        "want": ["hashmap", "链表", "linkedhashmap", "双向"],
    },
    {
        "key": "extension_heapsort",
        "label": "扩展要求 · 用了优先队列还要手写堆排序",
        "q": "这题你用了优先队列，那你把堆排序也手写出来给我看看。",
        "kind": "coding",
        "want_code": True,
        "want": ["heapify", "sift", "下沉", "调整"],
    },
    {
        "key": "boundary_idea_only",
        "label": "边界 · 只问思路不要代码 → 仍是 knowledge",
        "q": "讲讲快速排序的思路就行，不用写代码。",
        "kind": "knowledge",
        "want_code": False,
        "want": ["分治", "基准", "递归"],
    },
    {
        "key": "boundary_story",
        "label": "边界 · 提到写代码的经历题 → 仍是 behavioral",
        "q": "说说你写代码时遇到过的一个最棘手的 bug，怎么解决的？",
        "kind": "behavioral",
        "want_code": False,
        "want": ["你"],
    },
    {
        "key": "concept_no_code",
        "label": "基线 · 概念题不该给代码",
        "q": "说说 HashMap 和 ConcurrentHashMap 的区别。",
        "kind": "knowledge",
        "want_code": False,
        "want": ["分段", "cas", "segment", "synchronized", "并发"],
    },
]

FENCE = re.compile(r"```")
# A code block whose body lines are all flush-left is the bug we are hunting.
INDENT = re.compile(r"^[ \t]+\S", re.M)


def analyse(answer: str):
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", answer, re.S)
    has_code = bool(blocks)
    indented = any(INDENT.search(b) for b in blocks) if blocks else False
    # Longest run of leading spaces seen, as a rough "is it nested properly"
    # signal — a real implementation nests at least two levels.
    depth = 0
    for b in blocks:
        for line in b.splitlines():
            stripped = line.lstrip(" \t")
            if stripped:
                depth = max(depth, len(line) - len(stripped))
    return has_code, indented, depth, blocks


def run(job):
    model, case = job
    try:
        r = call(model, [{"role": "user", "content": case["q"]}])
        p = parse(r["text"])
        answer = p["answer"]
        has_code, indented, depth, blocks = analyse(answer)
        low = answer.lower()
        return {
            "key": case["key"],
            "label": case["label"],
            "model": model,
            "total_ms": r["total_ms"],
            "kind": p["kind"],
            "kind_expected": case["kind"],
            "kind_ok": p["kind"] == case["kind"],
            "answer": answer,
            "bullets": p["bullets"],
            "chars": len(answer),
            "has_code": has_code,
            "indented": indented,
            "max_indent": depth,
            "want_hits": [w for w in case["want"] if w in low],
            "code_expected": case["want_code"],
        }
    except Exception as exc:
        return {"key": case["key"], "model": model, "error": str(exc)[:160]}


if __name__ == "__main__":
    jobs = [(m, c) for c in CASES for m in MODELS]
    print(f"{len(jobs)} jobs ({len(CASES)} cases x {len(MODELS)} models)\n")

    out = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(run, jobs):
            out.append(r)
            if r.get("error"):
                print(f"  {r['key']:20s} {r['model'].replace('-ioa',''):22s} ERR {r['error'][:42]}")
                continue
            kind_mark = "ok " if r["kind_ok"] else f"!{r['kind'][:9]}"
            if r["code_expected"]:
                body = (
                    f"代码{'有' if r['has_code'] else '无 <<'} "
                    f"缩进{'正常' if r['indented'] else '塌了'}"
                )
            else:
                body = "无代码(对)" if not r["has_code"] else "多给代码 <<"
            print(
                f"  {r['key']:20s} {r['model'].replace('-ioa',''):22s} "
                f"{r['total_ms']:6d}ms kind={kind_mark:10s} {body}"
            )

    json.dump(out, open(ROOT / ".workbuddy" / "coding_answer.json", "w"), ensure_ascii=False, indent=1)

    ok = [r for r in out if not r.get("error")]
    print(f"\n=== 分类准确率 {sum(1 for r in ok if r['kind_ok'])}/{len(ok)} ===")
    for case in CASES:
        rows = [r for r in ok if r["key"] == case["key"]]
        good = sum(1 for r in rows if r["kind_ok"])
        wrong = {r["kind"] for r in rows if not r["kind_ok"]}
        note = f"  误判成 {'/'.join(sorted(wrong))}" if wrong else ""
        print(f"  {case['label']:42s} {good}/{len(rows)}{note}")

    need = [r for r in ok if r["code_expected"]]
    print(f"\n=== 需要代码的 {len(need)} 项 ===")
    print(f"  给出代码 : {sum(1 for r in need if r['has_code'])}/{len(need)}")
    print(f"  缩进正常 : {sum(1 for r in need if r['indented'])}/{len(need)}")
    base = [r for r in ok if not r["code_expected"]]
    print(f"不该给代码的 {len(base)} 项: {sum(1 for r in base if not r['has_code'])}/{len(base)} 正确")
    lat = [r["total_ms"] for r in need]
    if lat:
        print(f"代码题延迟: 均 {sum(lat)//len(lat)}ms  最慢 {max(lat)}ms")
