"""Ask the coach one question through the real shipped prompt.

Existing benchmark scripts run fixed case sets. This one takes an arbitrary
question so you can sanity-check "what would Meetly actually say if the
interviewer asked X" without editing a case table first.

The system prompt is read from /tmp/interview_prompt.txt, which is produced by
the Rust smoke test — never re-typed here, because a hand-copied prompt tests
something that does not ship:

    cargo test --manifest-path src-tauri/Cargo.toml \
        dump_interview_prompt_smoke -- --nocapture

Usage:
    python3 scripts/ask-one.py "面试官的问题"
    python3 scripts/ask-one.py "问题" --model deepseek-v4-pro-ioa
"""

import json
import os
import sys
import time
import urllib.request

BASE_URL = "https://copilot.tencent.com/v2/chat/completions"
API_KEY = json.load(open(os.path.expanduser("~/.meetly/secrets.json")))["llm_api_key"]
PROMPT_FILE = "/tmp/interview_prompt.txt"
DEFAULT_MODEL = "deepseek-v4.1-flash-ioa"


def call(model: str, question: str) -> dict:
    system = open(PROMPT_FILE, encoding="utf-8").read()
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
        "stream": True,
    }
    request = urllib.request.Request(
        BASE_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    started = time.time()
    ttft = None
    text = ""
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content")
            if delta:
                if ttft is None:
                    ttft = int((time.time() - started) * 1000)
                text += delta
    return {"raw": text, "ttft_ms": ttft, "total_ms": int((time.time() - started) * 1000)}


def parse(raw: str) -> dict:
    """Mirror the Rust side: JSON when possible, prose when the model rambles."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        value = json.loads(stripped)
        # Mirror the production parser's defensive ceiling. The prompt asks
        # design answers for 3-5 points, but sampled output occasionally emits
        # a sixth; the app truncates it before any UI sees the suggestion.
        bullets = value.get("bullets") or []
        return {
            "kind": value.get("kind", "knowledge"),
            "answer": value.get("answer", ""),
            "bullets": bullets[:5],
            "clarifying": value.get("clarifyingQuestion"),
        }
    except json.JSONDecodeError:
        return {"kind": "prose_fallback", "answer": raw, "bullets": [], "clarifying": None}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = DEFAULT_MODEL
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    if not args:
        sys.exit('usage: ask-one.py "面试官的问题" [--model NAME]')

    question = args[0]
    result = call(model, question)
    parsed = parse(result["raw"])

    print(f"model  : {model}")
    print(f"latency: {result['total_ms']}ms (首字 {result['ttft_ms']}ms)")
    print(f"kind   : {parsed['kind']}")
    print(f"chars  : {len(parsed['answer'])}")
    print("=" * 78)
    print(parsed["answer"])
    if parsed["bullets"]:
        label = "设计思路" if parsed["kind"] == "design" else "追问可补"
        print(f"\n--- {label} ---")
        for bullet in parsed["bullets"]:
            print(f"  - {bullet}")
    if parsed["clarifying"]:
        print(f"\n--- 反问 ---\n  {parsed['clarifying']}")
