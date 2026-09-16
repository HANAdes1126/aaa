"""Build a human-review page for the interviewer-only follow-up benchmark.

The automatic scorer only reports keyword hits, which repeatedly disagreed with
the qualitative read (it flagged "JDK 7 没有红黑树" as drift). So the real verdict
has to be human. This renders every one of the 64 answers as the interview
dialogue it belongs to — interviewer line, what the coach said last turn, the
follow-up, then before/after side by side — plus the exact prompt that produced
each answer, collapsed.

Run: python3 scripts/build-followup-review.py
Out: .workbuddy/followup-review.html
"""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = json.load(open("/tmp/followup_prompts.json"))
ROWS = json.load(open(ROOT / ".workbuddy" / "followup_system.json"))

MODEL_ORDER = [
    "deepseek-v4.1-flash-ioa",
    "deepseek-v4-flash-ioa",
    "deepseek-v4-pro-ioa",
    "gpt-5.6-luna",
]
MODEL_LABEL = {m: m.replace("-ioa", "") for m in MODEL_ORDER}


def esc(text):
    return html.escape(str(text or ""))


def find(key, arm, model):
    for r in ROWS:
        if r["key"] == key and r["arm"] == arm and r["model"] == model:
            return r
    return None


def answer_block(row):
    if row is None:
        return '<p class="err">（没有这条记录）</p>'
    if row.get("error"):
        return f'<p class="err">请求失败：{esc(row["error"])}</p>'
    bullets = ""
    if row.get("bullets"):
        items = "".join(f"<li>{esc(b)}</li>" for b in row["bullets"])
        bullets = f'<ul class="bul">{items}</ul>'
    return (
        f'<p class="ans">{esc(row["answer"])}</p>{bullets}'
        f'<p class="meta">{row["total_ms"]}ms · {row["chars"]} 字 · '
        f'kind={esc(row.get("kind"))} · JSON={"是" if row.get("json") else "否"}</p>'
    )


def prompt_details(case, arm):
    label = "改前（记忆为空）" if arm == "before" else "改后（带教练记忆）"
    info = case[arm]
    flags = (
        f'transcript {info["recentCount"]} 段 · 记忆 {info["coachTurns"]} 轮 · '
        f'召回 {"命中" if info["hasEarlier"] else "未命中"} · '
        f'记忆块 {"有" if info["hasMemoryBlock"] else "无"} · anchors {info["anchors"]} 条'
    )
    return (
        f"<details><summary>查看 {label} 送给模型的完整 prompt（{flags}）</summary>"
        f'<pre class="pp">{esc(info["text"])}</pre></details>'
    )


blocks = []
for case in CASES:
    rows_html = []
    for model in MODEL_ORDER:
        before = find(case["key"], "before", model)
        after = find(case["key"], "after", model)
        rows_html.append(
            f'<div class="mrow"><div class="mname">{esc(MODEL_LABEL[model])}</div>'
            f'<div class="pair">'
            f'<div class="col before"><div class="tag tb">改前</div>{answer_block(before)}</div>'
            f'<div class="col after"><div class="tag ta">改后</div>{answer_block(after)}</div>'
            f"</div></div>"
        )

    blocks.append(
        f"""<section class="case">
<h2>{esc(case['label'])}</h2>
<p class="note">{esc(case['notes'])}</p>
<div class="dialog">
  <div class="line"><span class="who itv">面试官</span><span class="say">{esc(case['q1'])}</span></div>
  <div class="line"><span class="who co">教练上轮</span><span class="say">{esc(case['a1'])}</span></div>
  <div class="line gap"><span class="who sil">候选人</span><span class="say muted">（只录对面模式，候选人说什么都进不了转录）</span></div>
  <div class="line"><span class="who itv">面试官</span><span class="say strong">{esc(case['q2'])}{'　（间隔 ' + str(case['gapMs'] // 60000) + ' 分钟，已超出 2 分钟实时窗口）' if case['gapMs'] > 120000 else ''}</span></div>
</div>
<p class="crit">人工判断要点：这个追问的答案应当接住「{esc(case['a1'][:40])}…」，
而不是另起一套方案，也不能把候选人已经说过的东西再建议一遍。</p>
{''.join(rows_html)}
<div class="prompts">{prompt_details(case, 'before')}{prompt_details(case, 'after')}</div>
</section>"""
    )

page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>追问实测完整记录 · 只录对面模式</title>
<style>
:root {{
  --bg: #faf9f7; --card: #fff; --line: #e5e2dc; --tx: #1c1c1a; --tx2: #63625d;
  --red: #a32d2d; --redbg: #fdf2f2; --grn: #0f6e56; --grnbg: #eef8f4;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 32px 24px 80px; background: var(--bg); color: var(--tx);
  font: 400 15px/1.7 -apple-system, "PingFang SC", "Helvetica Neue", sans-serif; }}
.wrap {{ max-width: 1180px; margin: 0 auto; }}
h1 {{ font-size: 24px; font-weight: 600; margin: 0 0 6px; }}
.sub {{ color: var(--tx2); font-size: 14px; margin: 0 0 8px; }}
.howto {{ background: #fff8e6; border: 1px solid #e8d9a8; border-radius: 10px;
  padding: 14px 18px; font-size: 14px; margin: 18px 0 28px; }}
.howto b {{ font-weight: 600; }}
.case {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 22px 26px; margin-bottom: 24px; }}
.case h2 {{ font-size: 18px; font-weight: 600; margin: 0 0 4px; }}
.note {{ color: var(--tx2); font-size: 13px; margin: 0 0 16px; }}
.dialog {{ background: #f6f5f2; border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }}
.line {{ display: flex; gap: 12px; margin-bottom: 8px; align-items: baseline; }}
.line:last-child {{ margin-bottom: 0; }}
.line.gap {{ opacity: .75; }}
.who {{ flex: 0 0 74px; font-size: 12px; padding: 2px 0; text-align: center;
  border-radius: 4px; font-weight: 500; }}
.itv {{ background: #e6f1fb; color: #185fa5; }}
.co {{ background: #eeedfe; color: #534ab7; }}
.sil {{ background: #f1efe8; color: #5f5e5a; }}
.say {{ flex: 1; font-size: 14px; }}
.say.strong {{ font-weight: 600; }}
.say.muted {{ color: var(--tx2); font-style: italic; }}
.crit {{ font-size: 13px; color: var(--tx2); border-left: 3px solid #d3d1c7;
  padding-left: 12px; margin: 0 0 20px; }}
.mrow {{ margin-bottom: 18px; }}
.mname {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px;
  color: var(--tx2); margin-bottom: 6px; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
@media (max-width: 820px) {{ .pair {{ grid-template-columns: 1fr; }} }}
.col {{ border-radius: 10px; padding: 12px 14px; border: 1px solid transparent; }}
.col.before {{ background: var(--redbg); border-color: #f0d5d5; }}
.col.after {{ background: var(--grnbg); border-color: #cfe8df; }}
.tag {{ font-size: 11px; font-weight: 600; letter-spacing: .04em; margin-bottom: 6px; }}
.tb {{ color: var(--red); }}
.ta {{ color: var(--grn); }}
.ans {{ margin: 0; font-size: 14px; }}
.bul {{ margin: 8px 0 0; padding-left: 20px; font-size: 13px; color: var(--tx2); }}
.meta {{ margin: 8px 0 0; font-size: 12px; color: var(--tx2);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.err {{ margin: 0; font-size: 13px; color: var(--red); }}
.prompts {{ margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }}
details {{ border-top: 1px solid var(--line); padding-top: 8px; }}
summary {{ cursor: pointer; font-size: 13px; color: var(--tx2); }}
.pp {{ background: #f6f5f2; border-radius: 8px; padding: 12px 14px; overflow-x: auto;
  font: 400 12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap; word-break: break-word; }}
</style></head><body><div class="wrap">
<h1>追问实测完整记录</h1>
<p class="sub">只录对面模式（audioSource = system）· {len(CASES)} 个场景 × {len(MODEL_ORDER)} 个模型 × 改前/改后 = {len(ROWS)} 条答案</p>
<p class="sub">prompt 由真实的 buildAgentPrompt 生成，与装机版本逐字节一致</p>
<div class="howto">
<b>怎么看：</b>每个场景先还原成面试对话——面试官问、教练上一轮答了什么、然后面试官追问。
关键在于<b>只录对面模式下候选人的声音永远进不了转录</b>，所以「教练上轮说了什么」是唯一的前文来源。<br>
<b>改前</b>（左，红）= 教练不记得自己说过什么；<b>改后</b>（右，绿）= 上一轮答案注入了 prompt。<br>
我的自动打分只数关键词，已经误判过好几次，所以哪边更能直接念出口，<b>请你自己判断</b>。
每个场景底部可以展开送给模型的完整 prompt。<br>
<b>注意：</b>C1 / C2 / C8 是我刻意留的基线场景（八股题模型自己就会答），改前改后本该没差别；
如果每个场景记忆都赢，那是在挑场景给功能捧场。
</div>
{''.join(blocks)}
</div></body></html>"""

target = ROOT / ".workbuddy" / "followup-review.html"
target.write_text(page, encoding="utf-8")
print(f"wrote {target} ({len(page)} bytes, {len(ROWS)} answers across {len(CASES)} cases)")
