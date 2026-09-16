"""Check whether a vision answer is being cut off mid-code.

A truncated coding answer is worse than no answer: the candidate pastes it,
it does not compile, and they burn interview time debugging our bug. The
9000-character answer logged at 22:11 was suspiciously round, so this replays
the exact screenshot that produced it and inspects the tail.

Truncation cannot be detected from the length alone — a legitimate two-solution
Java answer really is a few thousand characters. The signal is structural:
unbalanced code fences, unbalanced braces, or a final line that stops in the
middle of a statement.

Usage:
    python3 scripts/probe-vision-truncation.py [screenshot.jpg]
"""

import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://copilot.tencent.com/v2/chat/completions"
API_KEY = json.load(open(os.path.expanduser("~/.meetly/secrets.json")))["llm_api_key"]

# The vision prompt lives in Rust, so read it from the source instead of
# re-typing it here: a hand-copied prompt tests something that does not ship.
SRC = (ROOT / "src-tauri/src/app/screen_capture.rs").read_text(encoding="utf-8")
_match = re.search(
    r'pub const VISION_SYSTEM_PROMPT: &str = "(.*?)";\n', SRC, re.S
) or re.search(r'VISION_SYSTEM_PROMPT: &str = "(.*?)";\n', SRC, re.S)
if not _match:
    sys.exit("could not extract VISION_SYSTEM_PROMPT from screen_capture.rs")
SYSTEM = _match.group(1).replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")

QUESTION = "请解答截图中的题目"  # matches question_chars=12 in the log


def latest_sent_screenshot() -> Path:
    shots = sorted(
        Path(os.path.expanduser("~/.meetly/screenshots")).glob("sent-*.jpg"),
        key=lambda p: p.stat().st_mtime,
    )
    if not shots:
        sys.exit("no sent-*.jpg screenshots found")
    return shots[-1]


def call(model: str, image_b64: str) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        # Mirror the app: it never caps vision output.
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
    text = ""
    finish_reason = None
    with urllib.request.urlopen(request, timeout=300) as response:
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
            choice = (chunk.get("choices") or [{}])[0]
            text += (choice.get("delta") or {}).get("content") or ""
            # THE decisive field: "length" means the provider stopped us,
            # "stop" means the model finished on its own.
            finish_reason = choice.get("finish_reason") or finish_reason
    return {
        "text": text,
        "total_ms": int((time.time() - started) * 1000),
        "finish_reason": finish_reason,
    }


def analyse(text: str) -> dict:
    fences = text.count("```")
    # Only count braces inside code fences; prose may contain stray ones.
    code = "\n".join(
        re.findall(r"```(?:java)?\n(.*?)(?:```|\Z)", text, re.S)
    )
    tail = text.rstrip().splitlines()[-1] if text.strip() else ""
    return {
        "chars": len(text),
        "fences": fences,
        "fences_balanced": fences % 2 == 0,
        "braces_open": code.count("{"),
        "braces_close": code.count("}"),
        "braces_balanced": code.count("{") == code.count("}"),
        "tail": tail[-90:],
        # A healthy answer ends on a closing brace, a fence, or a full stop.
        "tail_looks_complete": bool(re.search(r"[}`。）\)]\s*$", text.rstrip())),
    }


if __name__ == "__main__":
    shot = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_sent_screenshot()
    image_b64 = base64.b64encode(shot.read_bytes()).decode()
    print(f"screenshot : {shot.name}  ({len(image_b64)} b64 chars)")
    print(f"prompt     : {len(SYSTEM)} chars extracted from screen_capture.rs")
    print()

    out = []
    for model in ("claude-opus-5", "deepseek-v4.1-flash-ioa"):
        print(f"--- {model} ---")
        try:
            result = call(model, image_b64)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {str(exc)[:140]}\n")
            continue
        stats = analyse(result["text"])
        verdict = (
            "被截断"
            if result["finish_reason"] == "length"
            or not stats["fences_balanced"]
            or not stats["braces_balanced"]
            else "完整"
        )
        print(f"  {result['total_ms']}ms  {stats['chars']} 字  finish_reason={result['finish_reason']}")
        print(
            f"  代码围栏 {stats['fences']} 个 "
            f"({'配对' if stats['fences_balanced'] else '不配对'})  "
            f"花括号 {stats['braces_open']}/{stats['braces_close']} "
            f"({'配对' if stats['braces_balanced'] else '不配对'})"
        )
        print(f"  结尾: ...{stats['tail']}")
        print(f"  判定: {verdict}\n")
        out.append({"model": model, **result, **stats, "verdict": verdict})

    target = ROOT / ".workbuddy" / "vision_truncation.json"
    json.dump(out, open(target, "w"), ensure_ascii=False, indent=1)
    print(f"-> {target.relative_to(ROOT)}")
