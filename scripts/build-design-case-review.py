"""Render design classification benchmark results as a human-review page."""

import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / ".workbuddy" / "design_vs_coding.json"
OUTPUT_DIR = ROOT / "outputs"
HTML_TARGET = OUTPUT_DIR / "design-case-review.html"
JSON_TARGET = OUTPUT_DIR / "design-case-results.json"

PERSISTENCE_TERMS = (
    "数据库", "mysql", "postgres", "redis", "表", "持久化", "落库",
    "对象存储", "存储", "状态记录", "kv", "唯一键", "流水",
)
DATABASE_TERMS = ("数据库", "mysql", "postgres", "建表", "落库", "持久化")


def primary_text(row):
    return "\n".join([row.get("answer", ""), *row.get("bullets", [])]).lower()


def flags(row):
    if row.get("error"):
        return [("bad", "请求失败")]
    result = []
    if row.get("kind") == row.get("want"):
        result.append(("ok", "分类正确"))
    else:
        result.append(("bad", f"分类应为 {row.get('want')}"))
    if row.get("want") == "design":
        raw_count = row.get("bullet_count", 0)
        count = min(raw_count, 5)
        result.append(("ok" if 3 <= count <= 5 else "bad", f"应用显示 {count} 条思路"))
        if raw_count > 5:
            result.append(("warn", f"模型原始 {raw_count} 条，解析器截为 5 条"))
        if row.get("has_code"):
            result.append(("bad", "误输出代码"))
        text = primary_text(row)
        if row.get("state") == "required":
            found = any(term in text for term in PERSISTENCE_TERMS)
            result.append(("ok" if found else "warn", "提到状态存储" if found else "需人工检查状态存储"))
        elif row.get("state") in {"none", "local_only"}:
            positive_text = text
            for negated in ("不落库", "无需数据库", "不需要数据库", "不依赖数据库"):
                positive_text = positive_text.replace(negated, "")
            overfit = any(term in positive_text for term in DATABASE_TERMS)
            if overfit:
                result.append(("warn", "可能机械加入数据库"))
            else:
                result.append(("ok", "未强塞数据库"))
    return result


def render_row(row, index):
    chips = "".join(
        f'<span class="chip {kind}">{html.escape(label)}</span>'
        for kind, label in flags(row)
    )
    if row.get("error"):
        body = f'<div class="error">{html.escape(row["error"])}</div>'
    else:
        # Render what the actual app exposes after Rust parsing, not an
        # impossible sixth raw item that production defensively truncates.
        bullets = "".join(
            f"<li>{html.escape(item)}</li>" for item in row.get("bullets", [])[:5]
        )
        bullet_block = f'<ol class="bullets">{bullets}</ol>' if bullets else ""
        body = (
            f'<div class="overview"><span>一句总览</span>{html.escape(row.get("answer", ""))}</div>'
            f'{bullet_block}'
        )
    latency = row.get("total_ms")
    latency_text = f"{latency / 1000:.2f}s" if isinstance(latency, (int, float)) else "-"
    return f"""
    <article class="case" data-kind="{html.escape(row.get('kind', 'error'))}" data-domain="{html.escape(row.get('domain', ''))}">
      <header>
        <div>
          <div class="eyebrow">#{index:02d} · 第 {row.get('sample', 1)} 次 · {html.escape(row.get('domain', '未分类'))} · {html.escape(row.get('key', ''))}</div>
          <h2>{html.escape(row.get('question', ''))}</h2>
        </div>
        <div class="meta"><b>{html.escape(row.get('kind', 'error'))}</b><span>{latency_text}</span></div>
      </header>
      <div class="chips">{chips}</div>
      {body}
    </article>
    """


def main():
    rows = json.loads(SOURCE.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_rows = []
    for row in rows:
        exported = dict(row)
        exported["raw_bullet_count"] = len(row.get("bullets", []))
        exported["app_bullets"] = row.get("bullets", [])[:5]
        export_rows.append(exported)
    JSON_TARGET.write_text(json.dumps(export_rows, ensure_ascii=False, indent=2))

    successes = [row for row in rows if not row.get("error")]
    design = [row for row in successes if row.get("want") == "design"]
    correct = sum(row.get("kind") == row.get("want") for row in successes)
    shaped = sum(
        row.get("kind") == "design" and 3 <= min(row.get("bullet_count", 0), 5) <= 5
        for row in design
    )
    coded = sum(row.get("has_code", False) for row in design)
    errors = len(rows) - len(successes)
    kinds = Counter(row.get("kind", "error") for row in rows)
    avg = sum(row.get("total_ms", 0) for row in successes) / max(len(successes), 1)
    unique_cases = len({row.get("key") for row in rows})
    samples_per_case = Counter(row.get("key") for row in rows)

    case_order = {}
    for row in rows:
        case_order.setdefault(row.get("key"), len(case_order))
    display_rows = sorted(
        rows, key=lambda row: (case_order[row.get("key")], row.get("sample", 1))
    )
    cards = "".join(render_row(row, i) for i, row in enumerate(display_rows, 1))
    kind_summary = " · ".join(f"{key} {value}" for key, value in sorted(kinds.items()))
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meetly 场景题扩展测试</title>
<style>
:root {{ color-scheme: light dark; --bg:#f3f5f8; --panel:#fff; --text:#19202a; --muted:#667085; --line:#dfe4ea; --accent:#3457d5; --soft:#edf2ff; --ok:#147d50; --warn:#a15c00; --bad:#b42318; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#101318; --panel:#181d24; --text:#edf2f7; --muted:#a3adba; --line:#303743; --accent:#87a2ff; --soft:#202a49; --ok:#54d69b; --warn:#f3b45d; --bad:#ff8b82; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; }}
main {{ width:min(1080px, calc(100% - 32px)); margin:36px auto 80px; }}
h1 {{ margin:0 0 8px; font-size:30px; }} .subtitle {{ color:var(--muted); margin:0 0 22px; line-height:1.7; }}
.stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:18px; }}
.stat,.case {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; }}
.stat {{ padding:15px; }} .stat b {{ display:block; font-size:24px; margin-bottom:4px; }} .stat span {{ color:var(--muted); font-size:13px; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:8px; margin:18px 0; }} button {{ border:1px solid var(--line); background:var(--panel); color:var(--text); border-radius:999px; padding:7px 12px; cursor:pointer; }} button.active {{ background:var(--accent); color:white; border-color:var(--accent); }}
.case {{ padding:20px; margin:12px 0; }} header {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; }}
h2 {{ font-size:17px; line-height:1.55; margin:4px 0 0; }} .eyebrow {{ color:var(--muted); font-size:12px; letter-spacing:.02em; }}
.meta {{ text-align:right; min-width:90px; }} .meta b {{ display:block; color:var(--accent); }} .meta span {{ color:var(--muted); font-size:12px; }}
.chips {{ display:flex; gap:7px; flex-wrap:wrap; margin:14px 0; }} .chip {{ font-size:12px; padding:4px 8px; border-radius:999px; background:var(--soft); }} .chip.ok {{ color:var(--ok); }} .chip.warn {{ color:var(--warn); }} .chip.bad {{ color:var(--bad); }}
.overview {{ line-height:1.75; padding:13px 15px; background:var(--soft); border-radius:10px; }} .overview span {{ display:block; color:var(--muted); font-size:12px; margin-bottom:3px; }}
.bullets {{ margin:14px 0 0; padding-left:25px; }} .bullets li {{ margin:8px 0; line-height:1.65; padding-left:4px; }} .error {{ color:var(--bad); }}
.note {{ color:var(--muted); font-size:13px; line-height:1.6; margin-top:12px; }}
@media (max-width:720px) {{ .stats {{ grid-template-columns:repeat(2,1fr); }} header {{ display:block; }} .meta {{ text-align:left; margin-top:8px; }} }}
</style>
</head>
<body><main>
<h1>Meetly 场景题扩展测试</h1>
<p class="subtitle">模型：deepseek-v4.1-flash-ioa · {unique_cases} 个问题 · 共 {len(rows)} 次回答（每题 {min(samples_per_case.values())}–{max(samples_per_case.values())} 次）。自动标签只检查分类和输出形态，回答好不好请直接看原文。</p>
<section class="stats">
  <div class="stat"><b>{correct}/{len(successes)}</b><span>分类正确</span></div>
  <div class="stat"><b>{shaped}/{len(design)}</b><span>设计题为 3–5 条</span></div>
  <div class="stat"><b>{coded}</b><span>设计题误输出代码</span></div>
  <div class="stat"><b>{avg/1000:.2f}s</b><span>平均完整回答耗时</span></div>
</section>
<p class="note">分类分布：{html.escape(kind_summary)}；请求失败：{errors}。其中“提到状态存储”是宽松关键词检查，不等于质量判定；“可能机械加入数据库”也需要人工结合题目判断。</p>
<div class="toolbar">
  <button class="active" data-filter="all">全部</button><button data-filter="design">design</button><button data-filter="knowledge">knowledge</button><button data-filter="coding">coding</button><button data-filter="behavioral">behavioral</button>
</div>
<section id="cases">{cards}</section>
</main>
<script>
for (const button of document.querySelectorAll('button[data-filter]')) {{
  button.addEventListener('click', () => {{
    document.querySelectorAll('button').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    const filter = button.dataset.filter;
    document.querySelectorAll('.case').forEach((card) => {{
      card.style.display = filter === 'all' || card.dataset.kind === filter ? '' : 'none';
    }});
  }});
}}
</script>
</body></html>"""
    HTML_TARGET.write_text(document)
    print(HTML_TARGET)
    print(JSON_TARGET)


if __name__ == "__main__":
    main()
