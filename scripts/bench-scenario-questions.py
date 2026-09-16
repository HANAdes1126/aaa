"""Meetly: answer-quality + latency benchmark for the interview coach path.

This is the "回答问题" benchmark: what the app actually does when the
interviewer asks a question and the candidate needs a spoken answer.

Design notes:
  - Real gateway (copilot.tencent.com), real key, streaming TTFT/total — so
    the numbers are what the app experiences, not a synthetic proxy.
  - Question set is deliberately spread over the four buckets the coach
    sees in a real interview, because they stress different things:
      八股/组成   naming every component in sentence 1
      对比        picking the axes that matter
      场景/方法    methodology, tends to run long
      系统设计     structure without bullet lists
      行为         must classify as `behavioral` and refuse to fabricate
  - Each question runs twice for the three production candidates so we can
    see jitter, not just a lucky single sample.

Requires /tmp/interview_prompt.txt (Rust test `dump_interview_prompt_smoke`).
"""

import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SECRETS = os.path.expanduser("~/.meetly/secrets.json")
CFG = os.path.expanduser(
    "~/Library/Application Support/com.maidang.meetly/provider_config.json"
)

cfg = json.load(open(CFG))["llm"]
BASE_URL = cfg["base_url"]
SYSTEM = open("/tmp/interview_prompt.txt").read()


def find_key(obj):
    if isinstance(obj, str) and len(obj) > 12:
        return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(t in k.lower() for t in ("key", "token", "secret")):
                r = find_key(v)
                if r:
                    return r
    return None


API_KEY = find_key(json.load(open(SECRETS))) or os.environ.get("MEETLY_LLM_KEY")
if not API_KEY:
    sys.exit("no api key")

QUESTIONS = {
    "cache_avalanche": ("场景", "如果缓存突然大面积失效，请求全打到数据库，你会怎么处理？"),
    "tech_disagree": ("场景", "如果你和同事对技术方案有分歧，你会怎么处理？"),
    "scope_change": ("场景", "上线前三天需求突然变了，但发布时间不变，你会怎么办？"),
    "cpu_spike": ("场景", "线上服务 CPU 突然打满，你会怎么定位？"),
}

MODELS_ROUND1 = [
    "deepseek-v4.1-flash-ioa",
    "deepseek-v4-flash-ioa",
    "deepseek-v4-pro-ioa",
]
MODELS_ROUND2 = []
# Round 2 is about jitter, so it only repeats the realistic candidates.
MODELS_ROUND2 = [
    "deepseek-v4.1-flash-ioa",
    "deepseek-v4-flash-ioa",
    "deepseek-v4-pro-ioa",
]

EXPECTED_KIND = {
    "hard_problem": "behavioral",
    "why_quit": "behavioral",
}


def strip_fence(text):
    clean = text.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        if len(parts) >= 2:
            clean = parts[1]
            if clean.startswith("json"):
                clean = clean[4:]
    return clean.strip()


def quality(text):
    raw = text.strip()
    try:
        parsed = json.loads(strip_fence(raw))
    except Exception:
        parsed = None

    if not isinstance(parsed, dict):
        return {
            "json": False,
            "answer": raw[:800],
            "chars": len(raw),
            "bullet_markers": raw.count("·") + len(re.findall(r"^\s*[-*]\s", raw, re.M)),
            "md_heading": bool(re.search(r"^\s*#{1,6}\s", raw, re.M)),
            "note": "not JSON",
        }

    answer = str(parsed.get("answer", ""))
    bullets = parsed.get("bullets") or []
    kind = str(parsed.get("kind", "")).strip().lower()
    if "behav" in kind or "行为" in kind or "经历" in kind:
        kind = "behavioral"
    elif kind:
        kind = "knowledge"
    return {
        "json": True,
        "kind": kind or "knowledge",
        "answer": answer,
        "bullets": bullets,
        "chars": len(answer),
        "sentences": len([s for s in re.split(r"[。！？.!?]", answer) if s.strip()]),
        "bullet_markers": answer.count("·") + len(re.findall(r"^\s*[-*]\s", answer, re.M)),
        "md_heading": bool(re.search(r"^\s*#{1,6}\s", answer, re.M)),
        "n_bullets": len(bullets),
        "has_digit": bool(re.search(r"\d", answer)),
    }


def run(model, qkey, round_no):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUESTIONS[qkey][1]},
    ]
    body = {"model": model, "messages": messages, "temperature": 0.3, "stream": True}
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "text/event-stream",
        },
    )

    started = time.time()
    ttft = None
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.time()
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                delta = evt.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    chunks.append(delta["content"])
    except urllib.error.HTTPError as e:
        return {
            "model": model,
            "q": qkey,
            "round": round_no,
            "bucket": QUESTIONS[qkey][0],
            "error": f"HTTP {e.code}: {e.read()[:200].decode('utf-8','replace')}",
        }
    except Exception as e:
        return {
            "model": model,
            "q": qkey,
            "round": round_no,
            "bucket": QUESTIONS[qkey][0],
            "error": f"{type(e).__name__}: {e}",
        }

    total = time.time() - started
    text = "".join(chunks)
    ttft_ms = (ttft - started) * 1000 if ttft else total * 1000
    result = {
        "model": model,
        "q": qkey,
        "round": round_no,
        "bucket": QUESTIONS[qkey][0],
        "question": QUESTIONS[qkey][1],
        "expected_kind": EXPECTED_KIND.get(qkey, "knowledge"),
        "ttft_ms": round(ttft_ms),
        "total_ms": round(total * 1000),
        "gen_ms": round((total * 1000) - ttft_ms),
        "cps": round(len(text) / max(total, 0.001), 1),
    }
    result.update(quality(text))
    return result


jobs = []
for m in MODELS_ROUND1:
    for k in QUESTIONS:
        jobs.append((m, k, 1))
for m in MODELS_ROUND2:
    for k in QUESTIONS:
        jobs.append((m, k, 2))

print(
    f"answer benchmark: {len(jobs)} jobs "
    f"({len(MODELS_ROUND1)}x{len(QUESTIONS)} round1 + {len(MODELS_ROUND2)}x{len(QUESTIONS)} round2)"
    f" via {BASE_URL}\n"
)

results = []
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(run, *j) for j in jobs]
    for f in futures:
        r = f.result()
        results.append(r)
        if r.get("error"):
            print(f"  FAIL {r['model']:<26} [{r['q']}] {r['error'][:110]}")
        else:
            ok = "OK " if r.get("kind") == r["expected_kind"] else "KIND"
            print(
                f"  {ok}  {r['model']:<26} [{r['q']:<13}] "
                f"ttft={r['ttft_ms']:>6}ms total={r['total_ms']:>6}ms "
                f"chars={r['chars']:>4} kind={r.get('kind','')}"
            )

json.dump(results, open("/tmp/scenario_q_results.json", "w"), ensure_ascii=False, indent=2)
print("\nsaved -> /tmp/scenario_q_results.json")
