"""Follow-up benchmark, interviewer-only mode (只录对面), 8 cases x 4 models x 2 arms.

Unlike the earlier stage-3 benchmark, the prompts here are NOT reconstructed in
Python — they are emitted by the real `ContextStore` + `buildAgentPrompt` via
`scripts/emit-followup-prompts.ts` with `audioSource: "system"`. So this measures
the prompt the installed app actually sends, including the interviewer-only
disclaimer and the follow-up guard rail.

Two arms per case:
  before   coach memory empty        (the app before 2026-09-11)
  after    previous answer in memory (what ships now)

Every case carries `expect` (terms a context-aware answer should contain) and
`avoid` (terms that prove the model drifted or defended its own stale answer).
Two cases are deliberate baselines where memory should show ~no gain — a
benchmark where the feature always wins is measuring the scenarios, not the code.

Run: ./node_modules/.bin/tsx scripts/emit-followup-prompts.ts
     python3 scripts/bench-followup-system.py
"""

import json
from importlib import util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent

# bench-followup.py has a hyphen, so it cannot be imported normally.
_spec = util.spec_from_file_location("bench_followup", ROOT / "scripts" / "bench-followup.py")
_bf = util.module_from_spec(_spec)
_spec.loader.exec_module(_bf)
MODELS, call, parse = _bf.MODELS, _bf.call, _bf.parse

CASES = json.load(open("/tmp/followup_prompts.json"))
ARMS = ("before", "after")

# A bare substring test flags "JDK 7 没有红黑树" as a violation even though that
# is the correct answer. Two legitimate shapes have to be excluded:
#   negation   "还没有红黑树" / "不是红黑树"
#   contrast   "这也是 JDK 8 改成尾插加红黑树的原因"
# Only an unqualified claim counts as drift. The negation window looks backwards
# because Chinese puts the negation first; the contrast check scans the whole
# sentence the term sits in, since the version marker can be far from the term.
NEGATIONS = ("没有", "不是", "还没", "无", "才有", "才加", "不存在", "并非", "没", "才引入")
NEGATION_WINDOW = 12
SENTENCE_BREAKS = "。；;！!？?\n"
CONTRAST_MARKERS = ("jdk 8", "jdk8", "1.8", "java 8", "改成", "才", "之后", "以后", "开始")


def _sentence_around(text: str, at: int) -> str:
    start = max((text.rfind(ch, 0, at) for ch in SENTENCE_BREAKS), default=-1)
    ends = [pos for pos in (text.find(ch, at) for ch in SENTENCE_BREAKS) if pos != -1]
    return text[start + 1 : min(ends) if ends else len(text)]


def avoid_violated(text: str, term: str) -> bool:
    """True only when `term` appears as a positive claim about the asked version."""
    lowered = text.lower()
    term = term.lower()
    start = 0
    while True:
        at = lowered.find(term, start)
        if at < 0:
            return False
        prefix = lowered[max(at - NEGATION_WINDOW, 0) : at]
        sentence = _sentence_around(lowered, at)
        negated = any(neg in prefix for neg in NEGATIONS)
        contrasted = any(marker in sentence for marker in CONTRAST_MARKERS)
        if not negated and not contrasted:
            return True
        start = at + len(term)


def build_jobs():
    jobs = []
    for case in CASES:
        for arm in ARMS:
            for model in MODELS:
                jobs.append((case, arm, model))
    return jobs


def run(job):
    case, arm, model = job
    prompt = case[arm]["text"]
    base = {"key": case["key"], "label": case["label"], "arm": arm, "model": model}
    try:
        r = call(model, [{"role": "user", "content": prompt}])
        p = parse(r["text"])
        text = (p["answer"] + " " + " ".join(p["bullets"])).lower()
        return {
            **base,
            "total_ms": r["total_ms"],
            "ttft_ms": r["ttft_ms"],
            "json": p["json"],
            "kind": p["kind"],
            "answer": p["answer"],
            "bullets": p["bullets"],
            "chars": len(p["answer"]),
            "hits": [t for t in case["expect"] if t.lower() in text],
            "misses": [t for t in case["expect"] if t.lower() not in text],
            "avoid_hit": [t for t in case["avoid"] if avoid_violated(text, t)],
        }
    except Exception as exc:
        return {**base, "error": str(exc)[:200]}


if __name__ == "__main__":
    jobs = build_jobs()
    print(f"{len(jobs)} jobs ({len(CASES)} cases x {len(ARMS)} arms x {len(MODELS)} models)\n")
    out = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(run, jobs):
            out.append(r)
            if r.get("error"):
                print(f"  {r['key']:18s} {r['arm']:6s} {r['model']:24s} ERR {r['error'][:60]}")
                continue
            bad = "!" if r["avoid_hit"] else " "
            print(
                f"  {r['key']:18s} {r['arm']:6s} {r['model'].replace('-ioa',''):22s} "
                f"{r['total_ms']:5d}ms  expect={len(r['hits'])}/{len(r['hits']) + len(r['misses'])} "
                f"avoid={len(r['avoid_hit'])}{bad}"
            )

    target = ROOT / ".workbuddy" / "followup_system.json"
    json.dump(out, open(target, "w"), ensure_ascii=False, indent=1)

    print("\n--- expect hits / avoid violations by arm ---")
    for arm in ARMS:
        rows = [r for r in out if r["arm"] == arm and not r.get("error")]
        hits = sum(len(r["hits"]) for r in rows)
        total = sum(len(r["hits"]) + len(r["misses"]) for r in rows)
        viol = sum(1 for r in rows if r["avoid_hit"])
        lat = [r["total_ms"] for r in rows]
        print(
            f"  {arm:6s} expect={hits}/{total}  avoid_violations={viol}/{len(rows)}  "
            f"avg={sum(lat) // max(len(lat), 1)}ms"
        )

    print("\n--- per case (expect hits before -> after, avoid violations) ---")
    for case in CASES:
        rows = {a: [r for r in out if r["key"] == case["key"] and r["arm"] == a and not r.get("error")] for a in ARMS}
        line = []
        for arm in ARMS:
            h = sum(len(r["hits"]) for r in rows[arm])
            t = sum(len(r["hits"]) + len(r["misses"]) for r in rows[arm])
            v = sum(1 for r in rows[arm] if r["avoid_hit"])
            line.append(f"{arm}={h}/{t} avoid={v}")
        print(f"  {case['label']:38s} {' | '.join(line)}")

    print(f"\n{len(out)} done -> {target}")
