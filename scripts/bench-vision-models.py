"""Which models on this gateway can actually read a screenshot?

Before wiring a separate vision-model setting it is worth knowing whether the
choice is real. A model that is strong at text is not necessarily accepted on
the `image_url` content shape — some return 400, some silently answer without
looking at the image.

Sends the SAME real screenshot to each candidate and reports: accepted, latency,
and whether the answer actually references what is in the picture.

Run: python3 scripts/bench-vision-models.py
"""

import base64
import glob
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = os.path.expanduser("~/Library/Application Support/com.maidang.meetly/provider_config.json")
SECRETS = os.path.expanduser("~/.meetly/secrets.json")

cfg = json.load(open(CFG))["llm"]
BASE_URL = cfg["base_url"]


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


API_KEY = find_key(json.load(open(SECRETS)))
assert API_KEY, "no api key"

# Reuse a screenshot the app already sent, so this measures the real payload
# size and the real question, not a toy image.
shots = sorted(glob.glob(os.path.expanduser("~/.meetly/screenshots/sent-*.jpg")))
assert shots, "没有历史截图，先在 app 里截一次屏"
SHOT = shots[-1]
IMAGE = base64.b64encode(open(SHOT, "rb").read()).decode()

CANDIDATES = [
    "deepseek-v4.1-flash-ioa",
    "deepseek-v4-flash-ioa",
    "deepseek-v4-pro-ioa",
    "claude-opus-5",
    "claude-sonnet-5",
    "gpt-5.6-luna",
    "gemini-3-pro",
    "qwen3-vl-plus",
]

QUESTION = "帮我分析截图里的题目并给出答案。"
SYSTEM = "你是一个截图解题助手。先读题，再作答。代码要完整、缩进正确。"


def call(model):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{IMAGE}"}},
                ],
            },
        ],
        "stream": True,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    started = time.time()
    ttft = None
    parts = []
    with urllib.request.urlopen(req, timeout=180) as resp:
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
                parts.append(delta)
    return "".join(parts), ttft or 0, int((time.time() - started) * 1000)


def run(model):
    try:
        text, ttft, total = call(model)
        blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
        indented = any(re.search(r"^[ \t]+\S", b, re.M) for b in blocks)
        return {
            "model": model,
            "ok": True,
            "ttft_ms": ttft,
            "total_ms": total,
            "chars": len(text),
            "has_code": bool(blocks),
            "indented": indented,
            "answer": text,
        }
    except Exception as exc:
        return {"model": model, "ok": False, "error": str(exc)[:150]}


if __name__ == "__main__":
    print(f"截图: {os.path.basename(SHOT)}  ({len(IMAGE)//1024} KB base64)\n")
    out = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(run, CANDIDATES):
            out.append(r)
            if not r["ok"]:
                print(f"  {r['model']:26s} 不可用  {r['error'][:70]}")
            else:
                print(
                    f"  {r['model']:26s} {r['total_ms']:6d}ms (首字 {r['ttft_ms']:5d}) "
                    f"{r['chars']:5d}字  代码{'有' if r['has_code'] else '无'} "
                    f"缩进{'正常' if r['indented'] else '-'}"
                )

    json.dump(out, open(ROOT / ".workbuddy" / "vision_models.json", "w"), ensure_ascii=False, indent=1)
    good = [r for r in out if r["ok"]]
    print(f"\n可用 {len(good)}/{len(out)}")
    if good:
        fastest = min(good, key=lambda r: r["total_ms"])
        print(f"最快: {fastest['model']} {fastest['total_ms']}ms")
