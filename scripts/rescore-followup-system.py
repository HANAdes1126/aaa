"""Re-score an existing followup_system.json run without re-calling any model.

The first run flagged "JDK 7 没有红黑树" as an avoid-violation, which is the
correct answer — a bare substring test cannot see the negation. Rather than
paying for 64 more LLM calls, re-apply the fixed scorer to the stored answers.

Run: python3 scripts/rescore-followup-system.py
"""

import json
from importlib import util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = util.spec_from_file_location(
    "bench_followup_system", ROOT / "scripts" / "bench-followup-system.py"
)
_bs = util.module_from_spec(_spec)
_spec.loader.exec_module(_bs)
avoid_violated, CASES, ARMS = _bs.avoid_violated, _bs.CASES, _bs.ARMS

BY_KEY = {c["key"]: c for c in CASES}
target = ROOT / ".workbuddy" / "followup_system.json"
rows = json.load(open(target))

changed = 0
for r in rows:
    if r.get("error"):
        continue
    case = BY_KEY[r["key"]]
    text = (r["answer"] + " " + " ".join(r["bullets"])).lower()
    fixed = [t for t in case["avoid"] if avoid_violated(text, t)]
    if fixed != r["avoid_hit"]:
        changed += 1
        r["avoid_hit_raw_substring"] = r["avoid_hit"]
    r["avoid_hit"] = fixed

json.dump(rows, open(target, "w"), ensure_ascii=False, indent=1)
print(f"rescored, {changed} verdict(s) corrected\n")

print("--- by arm ---")
for arm in ARMS:
    sel = [r for r in rows if r["arm"] == arm and not r.get("error")]
    hits = sum(len(r["hits"]) for r in sel)
    total = sum(len(r["hits"]) + len(r["misses"]) for r in sel)
    viol = sum(1 for r in sel if r["avoid_hit"])
    lat = [r["total_ms"] for r in sel]
    print(
        f"  {arm:6s} expect={hits}/{total}  avoid_violations={viol}/{len(sel)}  "
        f"avg={sum(lat) // max(len(lat), 1)}ms"
    )

print("\n--- per case ---")
for case in CASES:
    parts = []
    for arm in ARMS:
        sel = [r for r in rows if r["key"] == case["key"] and r["arm"] == arm and not r.get("error")]
        h = sum(len(r["hits"]) for r in sel)
        t = sum(len(r["hits"]) + len(r["misses"]) for r in sel)
        v = sum(1 for r in sel if r["avoid_hit"])
        parts.append(f"{arm}={h}/{t} avoid={v}")
    print(f"  {case['label']:38s} {' | '.join(parts)}")

print("\n--- latency by model (after arm) ---")
models = sorted({r["model"] for r in rows})
for m in models:
    sel = [r["total_ms"] for r in rows if r["model"] == m and r["arm"] == "after" and not r.get("error")]
    if not sel:
        continue
    print(f"  {m:26s} avg={sum(sel) // len(sel):5d}ms  min={min(sel):5d}  max={max(sel):5d}")
