"""Meetly coach prompt: scenario-question benchmark (behavioural + situational).

Scenario questions behave differently from knowledge questions, so they need
their own question set. Two buckets, and they diverge sharply:

  behavioural  (技术难题 / 搞砸的事 / 主动推动) -- the model INVENTS the story,
                with fabricated specifics ("压测两千并发", "错了三万条数据").
                Verify any behavioural answer against this before trusting it.
  situational  (线上变慢 / 技术分歧 / 需求变更) -- methodology, no personal
                history needed. This is where the answer is actually usable.

Usage:
  python3 scripts/bench-scenarios.py

Requires /tmp/interview_prompt.txt (Rust test `dump_interview_prompt_smoke`).
Reads the gateway URL from provider_config.json and the key from
~/.meetly/secrets.json, so it measures what the app actually experiences.

Usage:
  1. Export the live prompt first (Rust test `dump_interview_prompt_smoke`
     writes /tmp/interview_prompt.txt), or point SYSTEM at any file.
  2. Edit MODELS / QUESTIONS below, then:
       python3 scripts/bench-llm-models.py

Reads the real gateway URL from provider_config.json and the key from
~/.meetly/secrets.json, so it measures what the app actually experiences.

Streams each (model, question) pair the same way the app does and reports
TTFT / total / chars-per-second, then scores the parsed answer for the
spoken-answer contract (single paragraph, no bullets, no markdown).
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
CURRENT = cfg["model"]
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
    "hard_problem": "说说你遇到过的一个技术难题，你是怎么解决的。",
    "online_slow": "如果线上服务突然变慢了，你会怎么排查？",
    "disagree": "如果你和同事对技术方案有分歧，你会怎么处理？",
    "failed": "说一次你搞砸了的经历，你从中学到了什么？",
    "push_change": "讲一个你主动推动过的技术改进。",
    "scope_change": "上线前三天需求突然变了，但发布时间不变，你会怎么办？",
}

MODELS = [
    "deepseek-v4-flash-ioa",
    "deepseek-v4-pro-ioa",
    "gpt-5.6-luna",
]


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
    """Score the spoken-answer contract: how speakable is this?"""
    raw = text.strip()
    parsed = None
    try:
        parsed = json.loads(strip_fence(raw))
    except Exception:
        pass

    if not isinstance(parsed, dict):
        return {
            "json": False,
            "answer": raw[:400],
            "chars": len(raw),
            "bullet_markers": raw.count("·") + len(re.findall(r"^\s*[-*]\s", raw, re.M)),
            "md_heading": bool(re.search(r"^\s*#{1,6}\s", raw, re.M)),
            "note": "not JSON",
        }

    answer = str(parsed.get("answer", ""))
    bullets = parsed.get("bullets") or []
    raw_kind = parsed.get("kind")
    return {
        "json": True,
        # Mirrors Rust `normalize_kind`: unknown or missing means "knowledge".
        "kind": "behavioral" if isinstance(raw_kind, str)
        and ("behav" in raw_kind.lower() or "行为" in raw_kind or "经历" in raw_kind)
        else "knowledge",
        "raw_kind": raw_kind,
        "answer": answer,
        "bullets": bullets,
        "chars": len(answer),
        "sentences": len([s for s in re.split(r"[。！？]", answer) if s.strip()]),
        "bullet_markers": answer.count("·") + len(re.findall(r"^\s*[-*]\s", answer, re.M)),
        "md_heading": bool(re.search(r"^\s*#{1,6}\s", answer, re.M)),
        "n_bullets": len(bullets),
    }


def run(model, qkey, question):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
        "stream": True,
    }
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
        return {"model": model, "q": qkey, "error": f"HTTP {e.code}: {e.read()[:200].decode('utf-8','replace')}"}
    except Exception as e:
        return {"model": model, "q": qkey, "error": f"{type(e).__name__}: {e}"}

    total = time.time() - started
    text = "".join(chunks)
    ttft_ms = (ttft - started) * 1000 if ttft else total * 1000
    result = {
        "model": model,
        "q": qkey,
        "ttft_ms": round(ttft_ms),
        "total_ms": round(total * 1000),
        "gen_ms": round((total * 1000) - ttft_ms),
        "chars": len(text),
        "cps": round(len(text) / max(total, 0.001), 1),
    }
    result.update(quality(text))
    return result


jobs = [(m, k, q) for m in MODELS for k, q in QUESTIONS.items()]
print(f"benchmarking {len(MODELS)} models x {len(QUESTIONS)} questions via {BASE_URL}")
print(f"current app model: {CURRENT}\n")

results = []
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(run, *j) for j in jobs]
    for f in futures:
        r = f.result()
        results.append(r)
        if r.get("error"):
            print(f"  FAIL {r['model']:<26} [{r['q']}] {r['error']}")
        else:
            print(
                f"  ok   {r['model']:<26} [{r['q']}] "
                f"ttft={r['ttft_ms']:>6}ms total={r['total_ms']:>6}ms "
                f"chars={r['chars']:>4} json={r['json']}"
            )

json.dump(results, open("/tmp/scenario_results.json", "w"), ensure_ascii=False, indent=2)
print("\nsaved -> /tmp/scenario_results.json")
