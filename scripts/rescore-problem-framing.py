"""Recompute problem-framing metrics from stored answers. Calls no models.

The keyword lists in probe-problem-framing.py have been wrong three times now —
each time the model used a perfectly good synonym the list did not contain
("超发" instead of "超卖", "库存扣减" instead of "秒杀"), and each time the honest
fix was to widen the list, not to change the product. Re-running 24 requests to
validate a word list is both slow and noisy: sampling variance then gets mixed
into what should be a pure scoring change.

Usage:
    python3 scripts/rescore-problem-framing.py
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).resolve().parent / "probe-problem-framing.py"
)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)

CASES = {case["key"]: case for case in _probe.CASES}


def primary_text(row: dict) -> str:
    text = row["answer"]
    if row.get("kind") == "design":
        text += "\n" + "\n".join(row.get("bullets", []))
    return text


if __name__ == "__main__":
    stored = json.load(open(ROOT / ".workbuddy" / "problem_framing.json"))

    by_key: dict[str, list] = {}
    for row in stored:
        by_key.setdefault(row["key"], []).append(row)

    print("重算（不调模型）\n")
    for key, rows in by_key.items():
        case = CASES.get(key)
        if not case:
            print(f"  {key:20s} 已从 case 表移除，跳过")
            continue
        runs = len(rows)
        if case["expect_any"]:
            named = sum(
                any(t.lower() in primary_text(r).lower() for t in case["expect_any"])
                for r in rows
            )
            early = sum(
                any(
                    t.lower() in _probe.first_sentence(r["answer"]).lower()
                    for t in case["expect_any"]
                )
                for r in rows
            )
            print(
                f"  {key:20s} [{case['field']}] 命中 {named}/{runs}  开头就说 {early}/{runs}"
            )
            for name, terms in case.get("require_all", {}).items():
                got = sum(any(t in primary_text(r) for t in terms) for r in rows)
                flag = "" if got == runs else "   <-- 漏层"
                print(f"       └ {name:14s} {got}/{runs}{flag}")
            if case.get("avoid_any"):
                misread = sum(
                    any(t.lower() in primary_text(r).lower() for t in case["avoid_any"])
                    for r in rows
                )
                print(f"       └ 展示层被当成概率模型 {misread}/{runs}  (必须为 0)")
        else:
            wrong = sum(
                any(t in primary_text(r) for t in case.get("forbidden", [])) for r in rows
            )
            print(
                f"  {key:20s} [{case['field']}] 误用高并发术语 {wrong}/{runs}  "
                f"{'ok' if wrong == 0 else '套错术语'}"
            )

    real = [
        r
        for r in stored
        if CASES.get(r["key"]) and CASES[r["key"]]["expect_any"]
    ]
    named_total = sum(
        any(t.lower() in primary_text(r).lower() for t in CASES[r["key"]]["expect_any"])
        for r in real
    )
    print(f"\n有标准归类的题，说出名字 : {named_total}/{len(real)}")

    layered = [
        r
        for r in stored
        if CASES.get(r["key"]) and CASES[r["key"]].get("require_all")
    ]
    if layered:
        total = got = 0
        for row in layered:
            for terms in CASES[row["key"]]["require_all"].values():
                total += 1
                got += any(t in primary_text(row) for t in terms)
        print(f"多机制题，每层都答到     : {got}/{total}")
