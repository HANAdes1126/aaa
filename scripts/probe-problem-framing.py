"""Check that "name the known problem first" generalises, and knows when to stop.

A design answer that opens by naming the standard problem it reduces to reads
as transferable judgement; one that recites mechanisms reads as memorisation.
The prompt rule for this is easy to get wrong in two opposite ways, so both are
measured here:

  over-trigger   a question with no standard class gets a name forced onto it.
                 Worse than saying nothing: the interviewer will follow up on
                 whatever label was used.
  under-trigger  a textbook variant is described from scratch anyway.

Coverage spans backend, frontend, data and algorithms on purpose — a rule that
only fires for distributed-systems vocabulary is a patch for one question, not
a rule. `expect_any` lists acceptable names per case; `no_class` cases must
match none of them and are the ones that catch an over-eager prompt.

Usage:
    python3 scripts/probe-problem-framing.py --runs 3
    python3 scripts/probe-problem-framing.py --case gacha_pity --runs 8
"""

import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ask_one", Path(__file__).resolve().parent / "ask-one.py"
)
_ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ask)

ROOT = Path(__file__).resolve().parent.parent

CASES = [
    {
        "key": "lottery",
        "field": "后端",
        "q": "给你1000个人抢100个奖品，且每个中奖概率是30%，"
        "请问你怎么去设计数据结构，表格，和接口啥的",
        # "超卖" is the商品 framing; nothing is being sold here, so the model
        # reaches for 超发 / 库存扣减 / 秒杀 instead and those are equally
        # correct names. Listing only one phrasing measures vocabulary rather
        # than whether the class was named at all — this scorer has already
        # produced two false negatives that way.
        "expect_any": [
            "超卖",
            "超发",
            "秒杀",
            "抢购",
            "库存竞争",
            "库存扣减",
            "先到先得",
            "限量",
        ],
    },
    {
        "key": "feed_read",
        "field": "后端",
        "q": "设计一个微博的信息流，大V有千万粉丝，存储和接口怎么设计",
        "expect_any": ["读扩散", "写扩散", "推拉", "扇出", "fanout"],
    },
    {
        "key": "hot_key",
        "field": "后端",
        "q": "有个商品详情接口突然被刷爆了，缓存里那条数据还恰好过期了，你怎么设计",
        "expect_any": ["击穿", "穿透", "雪崩", "热点"],
    },
    {
        "key": "frontend_list",
        "field": "前端",
        "q": "前端要渲染十万条数据的列表，你怎么设计",
        "expect_any": ["虚拟", "虚拟滚动", "长列表", "分片", "时间切片"],
    },
    {
        "key": "dedup_stream",
        "field": "数据",
        "q": "每天十亿条日志，要统计今天有多少个不同的用户访问过，你怎么设计",
        "expect_any": ["基数", "hyperloglog", "布隆", "去重", "uv"],
    },
    {
        "key": "topk",
        "field": "算法",
        "q": "海量数据里找出出现次数最多的一百个词，你的数据结构怎么设计",
        "expect_any": ["top k", "topk", "堆", "分治", "分桶"],
    },
    # Naming the class is necessary but not sufficient. When a question stacks
    # several mechanisms, a confident class name can hide a dropped layer: the
    # answer sounds authoritative while silently ignoring part of what was
    # asked, which is exactly the kind of gap an interviewer probes next.
    # `require_all` makes each stated mechanism its own assertion. `avoid_any`
    # catches an actual misread without requiring the answer to spend scarce
    # spoken time explicitly explaining every piece of UI that it already
    # handled correctly.
    {
        "key": "gacha_pity",
        "field": "后端",
        "q": "如果是9宫格，中间一个按钮，周围8个奖品，稀有概率不一样，"
        "然后每个用户每抽一次幸运值都会提高，请问你怎么去设置",
        "expect_any": ["加权", "权重", "保底", "概率区间"],
        # The three actual mechanisms are weighted rarity, accumulating luck,
        # and what ends that accumulation. Merely mentioning a 3x3 UI does not
        # make "explain that UI is presentation" a fourth spoken requirement:
        # the first two reviewed answers already made the weight model decide
        # the result and were correct without spending a sentence on animation.
        "require_all": {
            "按稀有度分权重": ["权重", "加权", "概率区间", "稀有度"],
            "幸运值随抽次累积": ["幸运值", "累加", "递增", "抽数", "次数"],
            # Every synonym here has actually been produced by the model in a
            # real run — 清空 was missing at first and scored a correct answer
            # as a dropped layer. A concept has many phrasings; a term list
            # pinned to the one that came to mind measures vocabulary, not
            # substance. Add, never narrow.
            "命中后重置": [
                "清零",
                "重置",
                "归零",
                "扣减",
                "重新累计",
                "清空",
                "清掉",
                "复位",
                "重新计",
                "归位",
            ],
        },
        # This is the real UI/model failure: making position or equal-sized
        # cells determine probability. Absence of an explicit UI disclaimer is
        # not a failure when a proper weighted model already decides the draw.
        "avoid_any": ["每格等概率", "八等份", "12.5%", "按格子位置", "位置决定"],
    },
    # --- naming must stay honest ---
    #
    # These two started out as "no standard class exists" cases, which was a
    # bad premise: the model answered 维度聚合统计 and 配置化投放, and both are
    # real, standard designs — not invented labels. Almost any backend CRUD
    # question turns out to be a variant of something, so "no class exists" is
    # nearly impossible to construct honestly.
    #
    # What actually needs guarding is a name that is WRONG, so these now check
    # that a mundane question does not get dressed up in high-concurrency
    # vocabulary it has no business using. An internal ticket report is not
    # 秒杀; a config page is not 削峰填谷.
    {
        "key": "plain_report",
        "field": "平凡题",
        "q": "我们内部有个工单系统，现在想加一个按部门统计平均处理时长的功能，表怎么设计",
        "expect_any": [],
        "forbidden": ["秒杀", "超卖", "削峰", "分库分表", "一致性哈希", "缓存击穿"],
    },
    {
        "key": "plain_config",
        "field": "平凡题",
        "q": "给运营做一个活动文案配置页，后端的表和接口你怎么设计",
        "expect_any": [],
        "forbidden": ["秒杀", "超卖", "削峰", "分库分表", "一致性哈希", "缓存击穿"],
    },
]

# Phrases that mean "this is a known problem class". Kept for reporting only:
# their presence is not itself a defect, since naming the class is the point.
CLASS_MARKERS = [
    "经典", "本质上就是", "其实就是", "标准的", "典型的",
    "这就是", "属于", "老问题", "教科书",
]


def first_sentence(text: str) -> str:
    for sep in ("。", "；", "\n"):
        if sep in text:
            return text.split(sep, 1)[0]
    return text[:80]


if __name__ == "__main__":
    runs = 3
    model = _ask.DEFAULT_MODEL
    only_case = None
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    if "--case" in sys.argv:
        only_case = sys.argv[sys.argv.index("--case") + 1]

    selected_cases = [case for case in CASES if only_case in (None, case["key"])]
    if not selected_cases:
        raise SystemExit(f"unknown case: {only_case}")

    print(f"{model}  {len(selected_cases)} 题 x {runs} 次\n")
    out = []
    for case in selected_cases:
        named = 0
        early = 0
        wrong = 0
        misread = 0
        layer_hits = {name: 0 for name in case.get("require_all", {})}
        for _ in range(runs):
            result = _ask.call(model, case["q"])
            parsed = _ask.parse(result["raw"])
            answer = parsed["answer"]
            # In `design`, bullets are the primary visible outline. In every
            # other kind they are optional follow-up material and do not count
            # as answering the question.
            primary = answer
            if parsed["kind"] == "design":
                primary += "\n" + "\n".join(parsed["bullets"])
            lowered = primary.lower()
            hit = any(t.lower() in lowered for t in case["expect_any"])
            bad = any(t in primary for t in case.get("forbidden", []))
            bad_reading = any(t.lower() in lowered for t in case.get("avoid_any", []))
            named += hit
            wrong += bad
            misread += bad_reading
            # Design bullets are the visible answer outline; other kinds keep
            # bullets as reserve material, so `primary` above deliberately
            # includes them only for design.
            layers = {
                name: any(t in primary for t in terms)
                for name, terms in case.get("require_all", {}).items()
            }
            for name, ok in layers.items():
                layer_hits[name] += ok
            if hit and any(
                t.lower() in first_sentence(answer).lower() for t in case["expect_any"]
            ):
                early += 1
            out.append(
                {
                    "key": case["key"],
                    "field": case["field"],
                    "plain": not case["expect_any"],
                    "named": hit,
                    "wrong_label": bad,
                    "misread": bad_reading,
                    "kind": parsed["kind"],
                    "layers": layers,
                    "bullet_count": len(parsed["bullets"]),
                    "bullets": parsed["bullets"],
                    "chars": len(answer),
                    "total_ms": result["total_ms"],
                    "answer": answer,
                }
            )
        if case["expect_any"]:
            print(
                f"  {case['key']:20s} [{case['field']}] 命中 {named}/{runs}  "
                f"其中开头就说 {early}/{runs}"
            )
            for name, count in layer_hits.items():
                flag = "" if count == runs else "   <-- 漏层"
                print(f"       └ {name:14s} {count}/{runs}{flag}")
            if case.get("avoid_any"):
                print(f"       └ 展示层被当成概率模型 {misread}/{runs}  (必须为 0)")
        else:
            verdict = "ok" if wrong == 0 else "套错术语"
            print(f"  {case['key']:20s} [{case['field']}] 误用高并发术语 {wrong}/{runs}  {verdict}")

    print()
    real = [r for r in out if not r["plain"]]
    plain = [r for r in out if r["plain"]]
    print(f"有标准归类的题，说出名字   : {sum(r['named'] for r in real)}/{len(real)}")
    print(f"平凡题被套上高并发术语     : {sum(r['wrong_label'] for r in plain)}/{len(plain)}  (必须为 0)")
    layered = [r for r in out if r["layers"]]
    if layered:
        total = sum(len(r["layers"]) for r in layered)
        got = sum(sum(r["layers"].values()) for r in layered)
        print(f"多机制题，每层都答到       : {got}/{total}")

    suffix = f"_{only_case}" if only_case else ""
    target = ROOT / ".workbuddy" / f"problem_framing{suffix}.json"
    json.dump(out, open(target, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {target.relative_to(ROOT)}")
