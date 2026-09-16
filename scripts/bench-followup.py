"""Meetly: follow-up question benchmark (追问能力 + 记忆利用对比).

Run: python3 scripts/bench-followup.py   (then scripts/bench-followup-real.py)

Key design: every scenario runs each model TWICE — with the earlier turn in
`messages`, and with the follow-up alone as a control. If both answers match,
the model ignored the history no matter how good the answer looks; the delta
between the two is the actual memory contribution.

Requires /tmp/interview_prompt.txt (Rust test `dump_interview_prompt_smoke`).

The question this answers is not "is the model good?" but "does it actually
USE the earlier turns?" To prove that, every scenario runs twice per model:

  with_context   messages = [Q1, A1, Q2]   -- what the app really sends
  no_context     messages = [Q2]           -- control group

If a model's two answers are the same, it ignored the history no matter how
good the answer looks. If they differ, the delta is the memory contribution.

A1 is generated once by a fixed model (v4.1-flash) and reused for every model,
so the history is identical across the comparison.

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

MODELS = [
    "deepseek-v4.1-flash-ioa",
    "deepseek-v4-flash-ioa",
    "deepseek-v4-pro-ioa",
    "gpt-5.6-luna",
]
# Which model authors the shared history, so every model sees the same A1.
HISTORY_MODEL = "deepseek-v4.1-flash-ioa"


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


# --- scenarios ---------------------------------------------------------------
# `expect` lists the terms a context-aware answer should contain; `avoid` lists
# terms that would show the model drifted to the wrong referent.
SCENARIOS = {
    "pronoun": {
        "label": "S1 代词指代（那它的时间复杂度是多少）",
        "q1": "请描述哈希映射（hash map）的底层结构。",
        "q2": "那它的时间复杂度是多少？",
        "expect": ["o(1)", "o(n)", "o(log"],
        "expect_any_topic": ["哈希", "hash", "冲突", "桶"],
        "avoid": [],
    },
    "ellipsis": {
        "label": "S2 省略指代（那第二种的重写机制是怎么工作的）",
        "q1": "Redis 的持久化方式有哪些？",
        "q2": "那第二种的重写机制具体是怎么工作的？",
        "expect": ["aof"],
        "expect_any_topic": ["重写", "rewrite", "bgrewriteaof", "子进程", "fork"],
        "avoid": ["rdb 是第一种"],
    },
    "reversal": {
        "label": "S3 反转追问（那为什么不全部用线程）",
        "q1": "进程和线程的区别是什么？",
        "q2": "那为什么不全部用线程？",
        "expect": ["崩溃", "共享", "竞争", "锁", "隔离", "地址空间"],
        "expect_any_topic": ["线程", "进程"],
        "avoid": [],
    },
    "correction": {
        "label": "S4 纠正性追问（我问的是 JDK 7 的实现）",
        "q1": "HashMap 的底层结构是什么？",
        "q2": "你说的不对，我问的是 JDK 7 的实现。",
        "expect": ["链表"],
        "expect_any_topic": ["jdk 7", "jdk7", "1.7", "头插", "数组"],
        "avoid": ["红黑树", "树化"],
    },
    "long_range": {
        "label": "S5 长距离回调（刚才那个方案在分布式下有什么问题）",
        "q1": "你们的服务是怎么做限流的？",
        "q2": "刚才那个方案在分布式部署下有什么问题？",
        "expect": ["限流"],
        "expect_any_topic": ["令牌", "redis", "lua", "原子", "计数"],
        "avoid": [],
    },
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


def call(model, messages, timeout=90):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}] + messages,
        "stream": True,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    started = time.time()
    ttft = None
    payload = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
            except Exception:
                continue
            if delta:
                if ttft is None:
                    ttft = int((time.time() - started) * 1000)
                payload.append(delta)
    text = "".join(payload)
    return {
        "text": text,
        "ttft_ms": ttft or 0,
        "total_ms": int((time.time() - started) * 1000),
    }


def parse(text):
    try:
        obj = json.loads(strip_fence(text))
        return {
            "json": True,
            "kind": str(obj.get("kind", "knowledge")).strip().lower(),
            "answer": str(obj.get("answer", "")),
            "bullets": obj.get("bullets") or [],
        }
    except Exception:
        return {"json": False, "kind": "knowledge", "answer": text.strip(), "bullets": []}


def as_assistant_message(suggestion):
    # Mirrors Rust `build_voice_ask_messages`: the history turn is the JSON of
    # the previous suggestion, not a reformatted paragraph.
    return json.dumps(suggestion, ensure_ascii=False)


# --- stage 1: author the shared history --------------------------------------
def author_history():
    jobs = []
    for key, sc in SCENARIOS.items():
        jobs.append((key, sc))
    out = {}

    def run(item):
        key, sc = item
        r = call(HISTORY_MODEL, [{"role": "user", "content": sc["q1"]}])
        p = parse(r["text"])
        return key, {
            "q1": sc["q1"],
            "a1_raw": r["text"],
            "a1": {
                "kind": p["kind"],
                "answer": p["answer"],
                "bullets": p["bullets"],
            },
            "a1_ms": r["total_ms"],
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        for key, val in pool.map(run, jobs):
            out[key] = val
            print(f"  history {key}: {len(val['a1']['answer'])} chars, {val['a1_ms']}ms")
    return out


# --- stage 2: follow-up with and without context -----------------------------
def run_followups(history):
    jobs = []
    for key, sc in SCENARIOS.items():
        hist = history[key]["a1"]
        hist_msg = as_assistant_message(hist)
        for model in MODELS:
            jobs.append(
                (
                    key,
                    model,
                    "with_context",
                    [
                        {"role": "user", "content": sc["q1"]},
                        {"role": "assistant", "content": hist_msg},
                        {"role": "user", "content": sc["q2"]},
                    ],
                )
            )
            jobs.append(
                (
                    key,
                    model,
                    "no_context",
                    [{"role": "user", "content": sc["q2"]}],
                )
            )

    results = []

    def run(job):
        key, model, mode, messages = job
        try:
            r = call(model, messages)
            p = parse(r["text"])
            return {
                "scenario": key,
                "model": model,
                "mode": mode,
                "ttft_ms": r["ttft_ms"],
                "total_ms": r["total_ms"],
                "json": p["json"],
                "kind": p["kind"],
                "answer": p["answer"],
                "bullets": p["bullets"],
                "chars": len(p["answer"]),
            }
        except Exception as exc:
            return {
                "scenario": key,
                "model": model,
                "mode": mode,
                "error": str(exc)[:200],
            }

    with ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(run, jobs):
            results.append(r)
            tag = "ERR" if r.get("error") else f"{r['total_ms']}ms"
            print(f"  {r['scenario']:12s} {r['model']:26s} {r['mode']:13s} {tag}")
    return results


def score(results, history):
    """Automatic signals. Deliberately coarse — the qualitative read matters more."""
    for r in results:
        if r.get("error"):
            continue
        sc = SCENARIOS[r["scenario"]]
        text = (r["answer"] + " " + " ".join(r["bullets"])).lower()
        r["hits"] = [t for t in sc["expect"] if t.lower() in text]
        r["topic_hit"] = any(t.lower() in text for t in sc["expect_any_topic"])
        r["avoid_hit"] = [t for t in sc["avoid"] if t.lower() in text]
        # Overlap with the earlier answer: high overlap = restating, not building.
        a1 = history[r["scenario"]]["a1"]["answer"]
        r["overlap"] = round(char_overlap(r["answer"], a1), 3)
    return results


def char_overlap(a, b, n=4):
    if not a or not b:
        return 0.0
    grams = lambda s: {s[i : i + n] for i in range(max(len(s) - n + 1, 1))}
    ga, gb = grams(a), grams(b)
    return len(ga & gb) / max(len(ga), 1)


if __name__ == "__main__":
    print("stage 1: authoring shared history with", HISTORY_MODEL)
    history = author_history()
    print("\nstage 2: follow-ups with and without context")
    results = run_followups(history)
    results = score(results, history)
    out = {"history": history, "results": results}
    json.dump(out, open("/tmp/followup_results.json", "w"), ensure_ascii=False, indent=1)
    ok = sum(1 for r in results if not r.get("error"))
    print(f"\n{ok}/{len(results)} done -> /tmp/followup_results.json")
