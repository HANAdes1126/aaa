"""A/B the design-answer reasoning framework using real exported prompts."""

import html
import importlib.util
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEFORE_PROMPT = Path("/tmp/interview_prompt_before_xhs_framework.txt")
AFTER_PROMPT = Path("/tmp/interview_prompt.txt")
MODEL = "deepseek-v4.1-flash-ioa"
RUNS = 3
BASE_URL = "https://copilot.tencent.com/v2/chat/completions"
API_KEY = json.load(open(os.path.expanduser("~/.meetly/secrets.json")))["llm_api_key"]

_spec = importlib.util.spec_from_file_location("ask_one", Path(__file__).parent / "ask-one.py")
_ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ask)

CASES = [
    {
        "key": "signin_mysql_only",
        "want": "design",
        "q": "如果是要设计一个签到系统，不能用Redis、消息队列，只能用MySQL，而且还需要实现每日签到排行榜功能，应该怎么设计？",
        "signals": {
            "硬约束": ["只用mysql", "mysql", "唯一索引", "唯一键"],
            "排行榜语义": ["签到时间", "先后", "rank", "名次", "排名", "序号"],
            "并发幂等": ["幂等", "唯一索引", "唯一键", "事务", "原子", "行锁"],
            "关键链路": ["事务", "插入", "更新", "写回", "返回"],
            "取舍瓶颈": ["热点", "瓶颈", "代价", "取舍", "冲突", "吞吐"],
        },
        "forbidden_positive": ["redis", "消息队列", "mq", "本地缓存"],
    },
    {
        "key": "lottery_constraints",
        "want": "design",
        "q": "给你1000个人抢100个奖品，且每个中奖概率是30%，请问你怎么设计数据结构、表和接口",
        "signals": {
            "库存与概率": ["库存", "概率"],
            "并发幂等": ["原子", "锁", "唯一", "幂等", "事务"],
            "关键链路": ["接口", "请求", "扣减", "记录", "返回"],
            "取舍瓶颈": ["不足", "兜底", "瓶颈", "热点", "取舍", "代价"],
        },
    },
    {
        "key": "gacha_state",
        "want": "design",
        "q": "九宫格周围8个奖品稀有概率不同，每个用户每抽一次幸运值都会提高，这个系统怎么设计",
        "signals": {
            "动态概率": ["权重", "概率"],
            "状态生命周期": ["幸运值", "清零", "重置", "上限", "回退"],
            "持久状态": ["用户", "表", "存", "落库", "记录"],
            "关键链路": ["抽", "计算", "更新", "记录", "返回"],
        },
    },
    {
        "key": "payment_callbacks",
        "want": "design",
        "q": "第三方支付回调可能重复、乱序甚至丢失，你怎么设计订单更新和补偿机制",
        "signals": {
            "挑战定位": ["重复", "乱序", "丢失"],
            "幂等状态机": ["幂等", "唯一", "状态机", "条件更新"],
            "补偿链路": ["对账", "主动查询", "补偿", "重试"],
            "取舍边界": ["最终一致", "代价", "边界", "失败", "告警"],
        },
    },
    {
        "key": "chunk_upload",
        "want": "design",
        "q": "设计一个大文件分片上传功能，要求支持断点续传、秒传和失败重试",
        "signals": {
            "会话模型": ["会话", "upload_id", "上传id"],
            "三项约束": ["断点", "秒传", "重试"],
            "关键链路": ["创建", "分片", "合并", "校验"],
            "失败边界": ["幂等", "失败", "过期", "清理", "状态机"],
        },
    },
    {
        "key": "local_lru",
        "want": "design",
        "q": "设计一个进程内的 LRU 缓存，还要支持过期时间和并发访问，不要求写代码",
        "signals": {
            "核心结构": ["哈希", "map", "链表"],
            "过期机制": ["ttl", "过期", "expire"],
            "并发策略": ["锁", "并发", "分段", "concurrent"],
            "取舍边界": ["竞争", "吞吐", "近似", "代价", "不一致"],
        },
        "forbidden_positive": ["mysql", "数据库", "redis", "消息队列", "mq"],
    },
    {
        "key": "virtual_list",
        "want": "design",
        "q": "前端要流畅展示十万条高度不固定的数据列表，你会怎么设计这个组件",
        "signals": {
            "问题归类": ["虚拟", "可视区"],
            "不定高": ["测量", "高度", "缓存"],
            "定位链路": ["前缀和", "二分", "scroll", "偏移"],
            "取舍边界": ["抖动", "估算", "校正", "性能", "代价"],
        },
        "forbidden_positive": ["mysql", "数据库", "redis", "消息队列", "mq"],
    },
    {
        "key": "coding_override",
        "want": "coding",
        "q": "设计一个 LRU 缓存，并用 Java 完整实现 get 和 put",
        "signals": {},
    },
    {
        "key": "knowledge_control",
        "want": "knowledge",
        "q": "说说 MySQL 聚簇索引和非聚簇索引的区别",
        "signals": {},
    },
]

NEGATIONS = ("不能用", "不用", "不使用", "无需", "不需要", "没有用", "不依赖", "禁止", "去掉", "移除")


def call(prompt: str, question: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
        "stream": True,
    }
    request = urllib.request.Request(
        BASE_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    )
    started = time.time()
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
                text += delta
    return {"raw": text, "total_ms": int((time.time() - started) * 1000)}


def primary(parsed: dict) -> str:
    text = parsed.get("answer", "")
    if parsed.get("kind") == "design":
        text += "\n" + "\n".join(parsed.get("bullets", []))
    return text.lower()


def positive_use(text: str, term: str) -> bool:
    start = 0
    boundaries = "。！？；\n"
    while True:
        at = text.find(term, start)
        if at < 0:
            return False
        sentence_start = max(text.rfind(mark, 0, at) for mark in boundaries) + 1
        sentence_end_candidates = [text.find(mark, at) for mark in boundaries]
        sentence_end_candidates = [end for end in sentence_end_candidates if end >= 0]
        sentence_end = min(sentence_end_candidates, default=len(text))
        sentence = text[sentence_start:sentence_end]
        prefix = text[max(sentence_start, at - 24):at]
        if not any(negation in prefix or negation in sentence for negation in NEGATIONS):
            return True
        start = at + len(term)


def score(case: dict, parsed: dict) -> dict:
    text = primary(parsed)
    signal_hits = {
        name: any(term.lower() in text for term in terms)
        for name, terms in case.get("signals", {}).items()
    }
    forbidden = [
        term for term in case.get("forbidden_positive", [])
        if positive_use(text, term.lower())
    ]
    return {
        "kind_ok": parsed.get("kind") == case["want"],
        "shape_ok": parsed.get("kind") != "design" or 3 <= len(parsed.get("bullets", [])) <= 5,
        "signal_hits": signal_hits,
        "signal_count": sum(signal_hits.values()),
        "signal_total": len(signal_hits),
        "forbidden_positive": forbidden,
    }


def self_check():
    assert not positive_use("不能用redis，只能用mysql", "redis")
    assert not positive_use("方案不依赖消息队列", "消息队列")
    assert positive_use("后续可以加入redis优化", "redis")
    assert positive_use("使用本地缓存提高性能", "本地缓存")


def render(rows: list[dict], target: Path):
    groups = {}
    for row in rows:
        groups.setdefault((row["key"], row["run"]), {})[row["arm"]] = row

    cards = []
    for index, ((key, run), pair) in enumerate(groups.items(), 1):
        case = next(item for item in CASES if item["key"] == key)
        columns = []
        for arm in ("before", "after"):
            row = pair[arm]
            parsed = row["parsed"]
            metrics = row["metrics"]
            bullets = "".join(f"<li>{html.escape(item)}</li>" for item in parsed.get("bullets", []))
            hits = ", ".join(name for name, hit in metrics["signal_hits"].items() if hit) or "无"
            forbidden = ", ".join(metrics["forbidden_positive"]) or "无"
            columns.append(f"""
            <section class="arm">
              <h3>{arm}</h3>
              <p class="meta">kind={html.escape(parsed.get('kind',''))} · {row['total_ms']/1000:.2f}s · 信号 {metrics['signal_count']}/{metrics['signal_total']} · 禁用依赖={html.escape(forbidden)}</p>
              <p class="answer">{html.escape(parsed.get('answer',''))}</p>
              <ol>{bullets}</ol>
              <p class="signals">命中：{html.escape(hits)}</p>
            </section>""")
        cards.append(f"""
        <article><header><span>#{index:02d} · 第 {run} 次 · {html.escape(key)}</span><h2>{html.escape(case['q'])}</h2></header><div class="pair">{''.join(columns)}</div></article>
        """)

    target.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>场景题框架 A/B</title><style>
:root{{color-scheme:light dark;--bg:#f4f5f7;--panel:#fff;--text:#1f2933;--muted:#687381;--line:#dfe3e8;--soft:#eef3ff}}@media(prefers-color-scheme:dark){{:root{{--bg:#11151a;--panel:#1a2027;--text:#edf1f5;--muted:#9aa5b1;--line:#313944;--soft:#202b43}}}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{width:min(1240px,calc(100% - 28px));margin:30px auto 70px}}h1{{font-size:28px;margin:0 0 8px}}.intro,.meta,.signals,header span{{color:var(--muted)}}article{{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:14px 0;overflow:hidden}}header{{padding:16px 18px;border-bottom:1px solid var(--line)}}h2{{font-size:16px;line-height:1.55;margin:5px 0 0}}.pair{{display:grid;grid-template-columns:1fr 1fr}}.arm{{padding:16px 18px;min-width:0}}.arm+ .arm{{border-left:1px solid var(--line)}}h3{{margin:0 0 7px;font-size:14px}}.meta,.signals{{font-size:12px;line-height:1.5}}.answer{{background:var(--soft);border-radius:9px;padding:11px 13px;line-height:1.65}}ol{{padding-left:22px}}li{{margin:7px 0;line-height:1.6}}@media(max-width:760px){{.pair{{grid-template-columns:1fr}}.arm+.arm{{border-left:0;border-top:1px solid var(--line)}}}}
</style></head><body><main><h1>场景题框架 A/B</h1><p class="intro">同一模型、同一问题、每个版本各 3 次。before 是改动前真实 prompt，after 是加入“需求约束 → 核心挑战 → 子问题决策 → 关键链路 → 取舍边界”后的真实 prompt。</p>{''.join(cards)}</main></body></html>""")


def main():
    self_check()
    before = BEFORE_PROMPT.read_text()
    after = AFTER_PROMPT.read_text()
    assert "identify the one or two challenges" not in before
    assert "identify the one or two challenges" in after
    output_dir = ROOT / "outputs"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    json_target = output_dir / "design-framework-ab.json"
    html_target = output_dir / "design-framework-ab.html"
    rows = []
    for case in CASES:
        for run in range(1, RUNS + 1):
            for arm, prompt in (("before", before), ("after", after)):
                result = call(prompt, case["q"])
                parsed = _ask.parse(result["raw"])
                metrics = score(case, parsed)
                rows.append({
                    "key": case["key"], "question": case["q"], "want": case["want"],
                    "run": run, "arm": arm, "total_ms": result["total_ms"],
                    "parsed": parsed, "metrics": metrics,
                })
                json_target.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
                print(f"{case['key']:22s} run={run} {arm:6s} kind={parsed.get('kind')} signals={metrics['signal_count']}/{metrics['signal_total']} forbidden={metrics['forbidden_positive']}")
    json_target.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    render(rows, html_target)
    for arm in ("before", "after"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        design_rows = [row for row in arm_rows if row["want"] == "design"]
        signals = sum(row["metrics"]["signal_count"] for row in design_rows)
        total = sum(row["metrics"]["signal_total"] for row in design_rows)
        forbidden = sum(bool(row["metrics"]["forbidden_positive"]) for row in design_rows)
        print(f"{arm}: kind={sum(r['metrics']['kind_ok'] for r in arm_rows)}/{len(arm_rows)} shape={sum(r['metrics']['shape_ok'] for r in design_rows)}/{len(design_rows)} signals={signals}/{total} forbidden={forbidden}/{len(design_rows)} latency_avg={statistics.mean(r['total_ms'] for r in arm_rows):.0f}ms")
    print(json_target)
    print(html_target)


if __name__ == "__main__":
    main()
