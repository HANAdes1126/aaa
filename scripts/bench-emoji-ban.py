"""Does the emoji ban in the system prompt actually hold?

The UI strips pictographs anyway, so this is not about correctness — it is
about how often the stripper has to fire. Every strip leaves a small hole in
the sentence ("方案 ✅ 可行" → "方案 可行"), so a model that never emits them
reads better than one that gets cleaned up.

Prompts here deliberately bait emoji use: comparison questions, checklists, and
one that asks for them outright. If the ban survives the explicit request, it
will survive normal questions.

Run: python3 scripts/bench-emoji-ban.py
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

# Mirrors ALLOWED in src/app/coachMessageFormat.ts — these are legitimate in a
# technical answer and are NOT counted as violations.
ALLOWED = set("→←↑↓·≈≤≥≠∈∑√∞")
PICTO = re.compile(
    "[\U0001F000-\U0001FAFF\u2190-\u21FF\u2300-\u23FF\u2460-\u24FF"
    "\u25A0-\u27BF\u2B00-\u2BFF\uFE0F\u20E3]"
)

# Questions that invite icons: comparisons, pros/cons, checklists.
BAIT = [
    "对比一下 Redis 和 Memcached，各自优缺点是什么？",
    "上线前的检查清单一般包含哪些项？",
    "这个方案可行吗？说说风险。",
    "MySQL 和 MongoDB 怎么选？给个结论。",
    # The hard one: asks for emoji explicitly.
    "用 emoji 标记一下哪些做法是对的、哪些是错的，讲讲索引优化。",
]


def run(job):
    model, idx, question = job
    try:
        r = call(model, [{"role": "user", "content": question}])
        p = parse(r["text"])
        text = p["answer"] + " " + " ".join(p["bullets"])
        found = sorted({c for c in PICTO.findall(text) if c not in ALLOWED})
        return {
            "model": model,
            "case": idx,
            "question": question,
            "total_ms": r["total_ms"],
            "answer": p["answer"],
            "bullets": p["bullets"],
            "pictographs": found,
        }
    except Exception as exc:
        return {"model": model, "case": idx, "question": question, "error": str(exc)[:160]}


if __name__ == "__main__":
    jobs = [(m, i, q) for m in MODELS for i, q in enumerate(BAIT, 1)]
    print(f"{len(jobs)} jobs ({len(BAIT)} 个诱导问题 x {len(MODELS)} 模型)\n")

    out = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(run, jobs):
            out.append(r)
            if r.get("error"):
                print(f"  C{r['case']} {r['model'].replace('-ioa',''):22s} ERR {r['error'][:50]}")
                continue
            bad = "".join(r["pictographs"])
            print(
                f"  C{r['case']} {r['model'].replace('-ioa',''):22s} {r['total_ms']:5d}ms  "
                + (f"违规 {bad}" if bad else "干净")
            )

    json.dump(out, open(ROOT / ".workbuddy" / "emoji_ban.json", "w"), ensure_ascii=False, indent=1)

    ok = [r for r in out if not r.get("error")]
    viol = [r for r in ok if r["pictographs"]]
    print(f"\n=== {len(ok) - len(viol)}/{len(ok)} 干净，{len(viol)} 处违规 ===")
    for r in viol:
        print(f"  C{r['case']} {r['model']}: {''.join(r['pictographs'])}")
        print(f"     {r['answer'][:110]}")
    print("\n注：即使模型违规，UI 也会剥掉；这里测的是 prompt 侧的遵守率。")
