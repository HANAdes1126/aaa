"""A/B: when the candidate said something DIFFERENT from what the coach suggested.

Interviewer-only audio means the coach never hears the candidate. So the coach's
memory of its own previous answer can actively mislead it: the interviewer's
follow-up may reference an approach the candidate actually named, which never
reached the transcript.

  q1  "你们线上的限流是怎么做的？"
  a1  coach answer: gateway token bucket + Redis Lua + ZSET sliding window
      (candidate actually said "Sentinel" — unheard, absent from transcript)
  q2  "你刚说用的是 Sentinel 做单机 QPS 限流，那集群扩容时总量怎么保证？"

Two arms differ only in the note on <previous_coach_answers> and one guard clause:
  plain    current shipped wording
  hedged   adds "you cannot know whether the candidate actually said it; if the
           interviewer's wording implies a different approach, follow them"

A good answer talks about Sentinel. A derailed answer keeps explaining ZSET /
sliding windows, i.e. the coach's own unheard suggestion.
"""

import json
from importlib import util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent.parent
_spec = util.spec_from_file_location("bench_followup", ROOT / "scripts" / "bench-followup.py")
_bf = util.module_from_spec(_spec)
_spec.loader.exec_module(_bf)
MODELS, call, parse, SCENARIOS = _bf.MODELS, _bf.call, _bf.parse, _bf.SCENARIOS

HIST = json.load(open(ROOT / ".workbuddy" / "followup_results.json"))["history"]
Q1 = SCENARIOS["long_range"]["q1"]
A1 = HIST["long_range"]["a1"]["answer"]
Q2 = "你刚说用的是 Sentinel 做单机 QPS 限流，那集群扩容的时候，Sentinel 怎么保证总量不超？"

PLAIN_NOTE = (
    "what you already told the candidate earlier in this session, oldest first; "
    "the latest question may be a follow-up to one of these"
)
HEDGED_NOTE = (
    "what you already told the candidate earlier in this session, oldest first; "
    "the latest question may be a follow-up to one of these. NOTE: only the "
    "interviewer is being transcribed, so you CANNOT know whether the candidate "
    "actually said any of this. If the interviewer's own wording implies a "
    "different approach than yours, the interviewer is right — follow them."
)

PLAIN_RULE = (
    "If the latest question is a follow-up to something in the previous coach "
    "answers (它 / 这个 / 那个 / 第二种 / 刚才 / 为什么, or it simply continues the same "
    "topic), build on that answer and go one level deeper: never repeat it, never "
    "contradict it, and never restart from scratch. If the interviewer moved to a "
    "new topic, ignore the previous answers completely."
)
HEDGED_RULE = PLAIN_RULE + (
    " But if the interviewer's own words describe a different approach from your "
    "previous answer, the interviewer wins: answer about THEIR approach and drop "
    "yours, because the candidate may have said something entirely different."
)


def prompt_for(arm):
    note = PLAIN_NOTE if arm == "plain" else HEDGED_NOTE
    rule = PLAIN_RULE if arm == "plain" else HEDGED_RULE
    return "\n".join(
        [
            "Audio source: system",
            "",
            "Recent transcript:",
            Q1,
            Q2,
            "",
            f'<previous_coach_answers note="{note}">',
            f"Q: {Q1}",
            f"A: {A1}",
            "</previous_coach_answers>",
            "",
            "You are the right-side realtime interview coach. Give the candidate a "
            "complete answer they can deliver out loud.",
            rule,
            "Answer the interviewer's latest question.",
        ]
    )


def run(job):
    arm, model = job
    try:
        r = call(model, [{"role": "user", "content": prompt_for(arm)}])
        p = parse(r["text"])
        return {
            "arm": arm,
            "model": model,
            "total_ms": r["total_ms"],
            "answer": p["answer"],
            "chars": len(p["answer"]),
        }
    except Exception as exc:
        return {"arm": arm, "model": model, "error": str(exc)[:200]}


if __name__ == "__main__":
    jobs = [(arm, m) for arm in ("plain", "hedged") for m in MODELS]
    out = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(run, jobs):
            text = r.get("answer", "").lower()
            r["sentinel"] = "sentinel" in text
            # Derailed = still selling the coach's own unheard suggestion.
            r["derailed"] = any(k in text for k in ("zset", "滑动窗口", "令牌桶"))
            out.append(r)
            print(
                f"  {r['arm']:7s} {r['model'].replace('-ioa',''):22s} "
                f"{r.get('total_ms',0):5d}ms  sentinel={'Y' if r['sentinel'] else 'N'}  "
                f"derailed={'Y' if r['derailed'] else 'N'}"
            )
    json.dump(out, open(ROOT / ".workbuddy" / "followup_conflict.json", "w"), ensure_ascii=False, indent=1)
    print("\n" + json.dumps({"q1": Q1, "a1": A1, "q2": Q2}, ensure_ascii=False, indent=1))
