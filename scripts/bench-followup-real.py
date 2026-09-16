"""Stage 3: follow-ups through the REAL coach prompt shape.

The automatic coach path never sends the previous coach answer. Its prompt is
transcript + anchors + a recalled earlier turn (`<earlier_context>`). So whether
a follow-up works depends entirely on whether the CANDIDATE happened to speak
the answer out loud — because that is the only way it lands in the transcript.

Regimes:
  spoken          the candidate read the coach's answer aloud (upper bound)
  silent          the candidate just said "嗯，好的" (very common)
  memory          same as silent + `<previous_coach_answers>` (after the 2026-09-11 fix)
  system_only     audioSource=system: ONLY the interviewer is transcribed, so the
                  transcript is literally "Q1 / Q2" with nothing in between. Before
                  the fix the coach had no idea what it had already said.
  system_only_mem same, plus `<previous_coach_answers>`

`system_only*` is not the same as `silent`: in silent the model can at least see
that the candidate said nothing substantive. In system_only the two interviewer
lines sit directly adjacent and the candidate's voice never existed in the prompt.

Requires .workbuddy/followup_results.json from bench-followup.py (for A1).
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
MODELS, call, parse, SCENARIOS = _bf.MODELS, _bf.call, _bf.parse, _bf.SCENARIOS

HIST = json.load(open(ROOT / ".workbuddy" / "followup_results.json"))["history"]

# Only the two scenarios where context actually decides the answer.
KEYS = ["ellipsis", "long_range"]

FOLLOWUP_RULE = (
    "If the latest question is a follow-up to something in the previous coach "
    "answers (它 / 这个 / 那个 / 第二种 / 刚才 / 为什么, or it simply continues the "
    "same topic), build on that answer and go one level deeper: never repeat it, "
    "never contradict it, and never restart from scratch. If the interviewer moved "
    "to a new topic, ignore the previous answers completely."
)


def coach_prompt(q1, spoken_answer, q2, anchors="", previous=None, audio_source="system + mic"):
    """Mirrors buildAgentPrompt(): anchors, then recent transcript, then the
    optional earlier-context / previous-coach-answer blocks, then instructions."""
    lines = [f"Audio source: {audio_source}", ""]
    if anchors:
        lines += [anchors, ""]
    lines += ["Recent transcript:"]
    lines.append(q1)
    if spoken_answer:
        lines.append(spoken_answer)
    lines.append(q2)
    lines.append("")
    if previous:
        pq, pa = previous
        lines += [
            '<previous_coach_answers note="what you already told the candidate '
            'earlier in this session, oldest first; the latest question may be a '
            'follow-up to one of these">',
            f"Q: {pq}",
            f"A: {pa}",
            "</previous_coach_answers>",
            "",
        ]
    lines += [
        "You are the right-side realtime interview coach. Give the candidate a "
        "complete answer they can deliver out loud.",
    ]
    if previous:
        lines.append(FOLLOWUP_RULE)
    lines.append("Answer the interviewer's latest question.")
    return "\n".join(lines)


REGIMES = ("spoken", "silent", "memory", "system_only", "system_only_mem")


def build_jobs():
    jobs = []
    for key in KEYS:
        sc = SCENARIOS[key]
        a1 = HIST[key]["a1"]["answer"]
        for regime in REGIMES:
            sys_only = regime.startswith("system_only")
            if regime == "spoken":
                spoken = a1
            elif regime == "silent":
                spoken = "嗯，好的，我了解了。"
            else:
                # system_only: the candidate's voice is never transcribed at all.
                spoken = None
            previous = (sc["q1"], a1) if regime in ("memory", "system_only_mem") else None
            prompt = coach_prompt(
                sc["q1"],
                spoken,
                sc["q2"],
                previous=previous,
                audio_source="system" if sys_only else "system + mic",
            )
            for model in MODELS:
                jobs.append((key, regime, model, prompt))
    return jobs


def run(job):
    key, regime, model, prompt = job
    try:
        r = call(model, [{"role": "user", "content": prompt}])
        p = parse(r["text"])
        return {
            "scenario": key,
            "regime": regime,
            "model": model,
            "total_ms": r["total_ms"],
            "json": p["json"],
            "answer": p["answer"],
            "bullets": p["bullets"],
            "chars": len(p["answer"]),
        }
    except Exception as exc:
        return {"scenario": key, "regime": regime, "model": model, "error": str(exc)[:200]}


if __name__ == "__main__":
    jobs = build_jobs()
    out = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(run, jobs):
            out.append(r)
            sc = SCENARIOS[r["scenario"]]
            text = (r.get("answer", "") + " " + " ".join(r.get("bullets") or [])).lower()
            r["topic_hit"] = any(t.lower() in text for t in sc["expect_any_topic"])
            r["hits"] = [t for t in sc["expect"] if t.lower() in text]
            print(
                f"  {r['scenario']:11s} {r['regime']:7s} {r['model'].replace('-ioa',''):22s} "
                f"{r.get('total_ms', 0):5d}ms  {'HIT ' if r['topic_hit'] else 'MISS'}  "
                f"expect={len(r['hits'])}"
            )
    json.dump(out, open(ROOT / ".workbuddy" / "followup_real.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(out)} done -> .workbuddy/followup_real.json")
