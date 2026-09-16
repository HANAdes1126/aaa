"""Measure four-way classification and the design-outline output shape.

The coach decides `kind` per request. `design` must produce one framing sentence
plus 3-5 visible points; `coding` unlocks a fenced code block. A single-machine
Java class is wrong for 设计一个抽奖系统, while a paragraph without an outline
hides the data, flow, state, and boundary decisions the interviewer wants.

`kind` is sampled, not deduced, so a single run proves nothing — hence N
samples per question. Questions are split into three groups with a known
correct answer so the script can also catch an over-correction that starts
withholding code from real 手撕 questions.

Usage:
    python3 scripts/probe-design-vs-coding.py --runs 3
"""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ask_one", Path(__file__).resolve().parent / "ask-one.py"
)
_ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ask)

ROOT = Path(__file__).resolve().parent.parent

CASES = [
    # --- backend/system design: persistent state is part of the problem ---
    {
        "key": "design_lottery",
        "want": "design",
        "domain": "后端",
        "state": "required",
        "q": "给你1000个人抢100个奖品，且每个中奖概率是30%，"
        "请问你怎么去设计数据结构，表格，和接口啥的",
    },
    {
        "key": "design_shorturl",
        "want": "design",
        "domain": "后端",
        "state": "required",
        "q": "设计一个短链服务，说说你的数据结构和表怎么设计",
    },
    {
        "key": "design_feed",
        "want": "design",
        "domain": "后端",
        "state": "required",
        "q": "如果让你设计一个朋友圈的信息流，存储和接口你会怎么设计",
    },
    {
        "key": "design_gacha_pity",
        "want": "design",
        "domain": "后端",
        "state": "required",
        "q": "如果是9宫格，中间一个按钮，周围8个奖品，稀有概率不一样，"
        "然后每个用户每抽一次幸运值都会提高，请问你怎么去设置",
    },
    {
        "key": "design_order_timeout",
        "want": "design",
        "domain": "后端",
        "state": "required",
        "q": "订单创建后30分钟没支付就自动取消，这个功能你怎么设计",
    },
    {
        "key": "design_seat_hold",
        "want": "design",
        "domain": "后端",
        "state": "required",
        "q": "设计电影院选座，用户锁座5分钟，超时自动释放，还要避免同一个座位卖给两个人",
    },
    {
        "key": "design_coupon_claim",
        "want": "design",
        "domain": "后端",
        "state": "required",
        "q": "一万张优惠券同一时间开抢，每个用户只能领一张，你会怎么设计",
    },
    {
        "key": "design_chunk_upload",
        "want": "design",
        "domain": "存储",
        "state": "required",
        "q": "设计一个大文件分片上传功能，要求支持断点续传、秒传和失败重试",
    },
    {
        "key": "design_chat_unread",
        "want": "design",
        "domain": "即时通信",
        "state": "required",
        "q": "聊天系统的未读数怎么设计，要求手机和电脑多端登录后最终一致",
    },
    {
        "key": "design_payment_callback",
        "want": "design",
        "domain": "支付",
        "state": "required",
        "q": "第三方支付回调可能重复、乱序甚至丢失，你怎么设计订单更新和补偿机制",
    },
    {
        "key": "design_leaderboard",
        "want": "design",
        "domain": "后端",
        "state": "required",
        "q": "设计一个每天重置的游戏排行榜，需要查前100名和任意用户当前排名",
    },
    {
        "key": "design_rate_limit",
        "want": "design",
        "domain": "网关",
        "state": "required",
        "q": "多实例服务要限制每个用户每分钟最多请求100次，这个限流功能怎么设计",
    },
    {
        "key": "design_feature_flag",
        "want": "design",
        "domain": "发布系统",
        "state": "required",
        "q": "设计一个灰度开关系统，能按用户百分比放量、指定白名单，并支持一键回滚",
    },
    {
        "key": "design_search_suggest",
        "want": "design",
        "domain": "搜索",
        "state": "required",
        "q": "搜索框输入前缀时要在10毫秒内返回热门联想词，这个功能怎么设计",
    },
    {
        "key": "design_comment_thread",
        "want": "design",
        "domain": "内容",
        "state": "required",
        "q": "设计评论和二级回复功能，还要支持热度排序、删除和分页",
    },
    {
        "key": "design_job_dag",
        "want": "design",
        "domain": "调度",
        "state": "required",
        "q": "设计一个有任务依赖关系的批处理调度系统，失败可以重试但不能重复执行成功任务",
    },
    {
        "key": "design_album_share",
        "want": "design",
        "domain": "权限",
        "state": "required",
        "q": "设计相册分享功能，链接可以设置过期时间和访问密码，用户还能随时撤销分享",
    },
    {
        "key": "design_signin_mysql_only",
        "want": "design",
        "domain": "后端",
        "state": "required",
        "q": "设计一个签到系统，不能用Redis、消息队列，只能用MySQL，"
        "而且还需要实现每日签到排行榜功能",
    },
    # --- design questions where a database must not be inserted mechanically ---
    {
        "key": "design_local_lru_ttl",
        "want": "design",
        "domain": "本地组件",
        "state": "local_only",
        "q": "设计一个进程内的 LRU 缓存，还要支持过期时间和并发访问，不要求你写代码",
    },
    {
        "key": "design_virtual_list",
        "want": "design",
        "domain": "前端",
        "state": "none",
        "q": "前端要流畅展示十万条高度不固定的数据列表，你会怎么设计这个组件",
    },
    {
        "key": "design_id_generator",
        "want": "design",
        "domain": "基础设施",
        "state": "optional",
        "q": "设计一个分布式唯一 ID 生成器，要求趋势递增、高可用，而且不能依赖单机自增数据库",
    },
    # --- genuine coding questions: code is required ---
    {
        "key": "coding_quicksort",
        "want": "coding",
        "domain": "算法",
        "state": "none",
        "q": "来手写一个快速排序吧，用 Java",
    },
    {
        "key": "coding_lru",
        "want": "coding",
        "domain": "算法",
        "state": "none",
        "q": "实现一个 LRU 缓存，说一下思路然后写出来",
    },
    {
        "key": "coding_design_lru",
        "want": "coding",
        "domain": "算法",
        "state": "none",
        "q": "设计一个 LRU 缓存，并用 Java 完整实现 get 和 put",
    },
    # --- standard answers: a scenario word alone must not force design ---
    {
        "key": "concept_index",
        "want": "knowledge",
        "domain": "数据库",
        "state": "none",
        "q": "说说 MySQL 的聚簇索引和非聚簇索引有什么区别",
    },
    {
        "key": "method_cpu",
        "want": "knowledge",
        "domain": "排障",
        "state": "none",
        "q": "线上 Java 服务 CPU 突然到 100%，你会怎么排查",
    },
    {
        "key": "method_cache_breakdown",
        "want": "knowledge",
        "domain": "缓存",
        "state": "none",
        "q": "缓存击穿是什么，常见解决办法有哪些",
    },
    {
        "key": "method_slow_api",
        "want": "knowledge",
        "domain": "排障",
        "state": "none",
        "q": "一个接口突然变慢，你一般按什么步骤定位问题",
    },
    # --- personal-history questions: do not fabricate a candidate story ---
    {
        "key": "behavioral_incident",
        "want": "behavioral",
        "domain": "行为题",
        "state": "none",
        "q": "说说你处理过最严重的一次线上事故，以及你是怎么复盘的",
    },
    {
        "key": "behavioral_design",
        "want": "behavioral",
        "domain": "行为题",
        "state": "none",
        "q": "讲一个你主导过的系统设计项目，你遇到的最大分歧是什么",
    },
]


def has_code(text: str) -> bool:
    return "```" in text


if __name__ == "__main__":
    runs = 3
    model = _ask.DEFAULT_MODEL
    only_case = None
    append = "--append" in sys.argv
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    if "--case" in sys.argv:
        only_case = sys.argv[sys.argv.index("--case") + 1]

    selected_cases = [case for case in CASES if only_case in (None, case["key"])]
    if not selected_cases:
        raise SystemExit(f"unknown case: {only_case}")

    suffix = f"_{only_case}" if only_case else ""
    target = ROOT / ".workbuddy" / f"design_vs_coding{suffix}.json"
    out = json.loads(target.read_text()) if append and target.exists() else []
    if append:
        print(f"续跑：已载入 {len(out)} 条历史结果")
    print(f"{model}  {len(selected_cases)} 题 x {runs} 次 = {len(selected_cases) * runs} 次\n")
    for case in selected_cases:
        kinds = Counter()
        coded = 0
        prior_samples = [
            row.get("sample", 0) for row in out if row.get("key") == case["key"]
        ]
        sample_offset = max(prior_samples, default=0)
        for sample in range(sample_offset + 1, sample_offset + runs + 1):
            try:
                result = _ask.call(model, case["q"])
                parsed = _ask.parse(result["raw"])
            except Exception as exc:
                out.append(
                    {
                        "key": case["key"],
                        "want": case["want"],
                        "domain": case["domain"],
                        "state": case["state"],
                        "question": case["q"],
                        "sample": sample,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            kinds[parsed["kind"]] += 1
            if has_code(parsed["answer"]):
                coded += 1
            out.append(
                {
                    "key": case["key"],
                    "want": case["want"],
                    "domain": case["domain"],
                    "state": case["state"],
                    "question": case["q"],
                    "sample": sample,
                    "kind": parsed["kind"],
                    "has_code": has_code(parsed["answer"]),
                    "bullet_count": len(parsed["bullets"]),
                    "bullets": parsed["bullets"],
                    "chars": len(parsed["answer"]),
                    "total_ms": result["total_ms"],
                    "answer": parsed["answer"],
                }
            )
        correct = kinds[case["want"]]
        flag = "" if correct == runs else "   <-- 不稳定"
        outline_counts = [
            row["bullet_count"] for row in out[-runs:]
            if row.get("kind") == "design"
        ]
        outline = (
            f"  思路数={min(outline_counts)}-{max(outline_counts)}"
            if outline_counts else ""
        )
        print(
            f"  {case['key']:18s} 期望={case['want']:9s} "
            f"实际={dict(kinds)}  带代码 {coded}/{runs}{outline}{flag}"
        )

    print()
    design = [r for r in out if r["key"].startswith("design_")]
    coding = [r for r in out if r["key"].startswith("coding_")]
    print(f"设计题正确分类 design : {sum(r.get('kind') == 'design' for r in design)}/{len(design)}")
    print(f"设计题实际甩了代码    : {sum(r.get('has_code', False) for r in design)}/{len(design)}")
    print(f"设计题给出 3-5 条思路 : {sum(3 <= r.get('bullet_count', -1) <= 5 for r in design)}/{len(design)}")
    print(f"手撕题正确给代码      : {sum(r.get('has_code', False) for r in coding)}/{len(coding)}")

    json.dump(out, open(target, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {target.relative_to(ROOT)}")
