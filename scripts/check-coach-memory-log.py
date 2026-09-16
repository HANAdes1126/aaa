"""Read ~/.meetly/debug.log and report whether coach memory actually fired.

Written because eyeballing an 11k-line log for `coachTurns=` is slow and easy to
get wrong — and because "the feature is in the bundle" is not the same claim as
"the feature ran on this machine with real audio". This only reports what the
log proves.

Run: python3 scripts/check-coach-memory-log.py            # last session
     python3 scripts/check-coach-memory-log.py --all      # every session
"""

import os
import re
import sys
from datetime import datetime

LOG = os.path.expanduser("~/.meetly/debug.log")

# Two generations of this log line exist. Before 2026-09-11 the coach had no
# memory of its own answers, so there was no `coachTurns` field:
#   [session-memory] wake=… anchors=N[…] earlier=hit|miss durable=N recent=N
# Matching only the new shape makes every historical line invisible and the
# report then claims "transcripts but no session-memory — wake never fired",
# which is a false alarm about a feature that simply did not exist yet.
MEM_RE = re.compile(
    r"\[session-memory\] wake=(?P<wake>\S+) anchors=(?P<anchors>\d+)\[(?P<alist>[^\]]*)\] "
    r"earlier=(?P<earlier>\w+) (?:coachTurns=(?P<turns>\d+) )?"
    r"durable=(?P<durable>\d+) recent=(?P<recent>\d+)"
)
TS_RE = re.compile(r"^(\d{13}) ")


def stamp(line):
    m = TS_RE.match(line)
    if not m:
        return ""
    return datetime.fromtimestamp(int(m.group(1)) / 1000).strftime("%H:%M:%S")


def main():
    if not os.path.exists(LOG):
        sys.exit(f"没有找到日志：{LOG}")

    lines = open(LOG, encoding="utf-8", errors="ignore").read().splitlines()
    starts = [i for i, l in enumerate(lines) if "[native] app start" in l]
    scope = lines if "--all" in sys.argv else lines[starts[-1] if starts else 0 :]
    label = "全部历史" if "--all" in sys.argv else "本次会话"

    print(f"=== {label}（{len(scope)} 行）===")
    if scope:
        print(f"起始 {stamp(scope[0])}　最后 {stamp(scope[-1])}")

    # Did an interview actually happen? Without this, a zero coachTurns count
    # means "nothing was tested", not "the feature is broken" — a distinction
    # worth making loudly.
    transcripts = [l for l in scope if "[demand-trace]" in l]
    answers = [l for l in scope if "[agent] coach message" in l]
    bridged = [l for l in scope if "voice-ask answer bridged" in l]
    folded = [l for l in scope if "manual answer folded into coach memory" in l]

    print(f"\n转录片段 {len(transcripts)}　教练回答 {len(answers)}　"
          f"语音回灌 {len(bridged)}　手动回灌 {len(folded)}")

    hits = []
    for l in scope:
        m = MEM_RE.search(l)
        if m:
            hits.append((stamp(l), m.groupdict()))

    if not hits:
        print("\n没有 [session-memory] 记录。")
        if not transcripts:
            print("原因：这次没有开始过会话，没有任何转录 —— 不是功能问题，是还没测。")
        else:
            print("注意：有转录但没有 session-memory，说明唤醒没触发（问句没被识别？）。")
        return

    # A line without `coachTurns` predates the feature. Counting those as
    # "memory was empty" would understate the feature; calling them broken
    # would be a false alarm. They are simply older builds.
    legacy = [(t, g) for t, g in hits if g["turns"] is None]
    current = [(t, g) for t, g in hits if g["turns"] is not None]

    if legacy and not current:
        print(f"\n找到 {len(legacy)} 条 [session-memory]，但全部来自**旧版本**"
              "（日志里没有 coachTurns 字段）。")
        print("这不是功能故障 —— 这些记录产生时教练记忆还没上线。")
        print("需要用当前版本跑一场会话，才会出现带 coachTurns 的记录。")
        return

    print(f"\n=== 教练每次回答时看到的记忆（{len(current)} 次）===")
    print(f"{'时间':10s} {'唤醒':16s} {'记忆轮':>6s} {'召回':>5s} {'锚点':>4s} {'窗内':>4s}  锚点内容")
    for t, g in current:
        print(f"{t:10s} {g['wake']:16s} {g['turns']:>6s} {g['earlier']:>5s} "
              f"{g['anchors']:>4s} {g['recent']:>4s}  {g['alist'][:60]}")

    if legacy:
        print(f"\n（另有 {len(legacy)} 条旧版本记录，无 coachTurns 字段，已略过）")

    withmem = [g for _, g in current if int(g["turns"]) > 0]
    print(f"\n带记忆的回答：{len(withmem)}/{len(current)}")
    if withmem:
        print("记忆已注入 —— 教练回答时看得到自己上一轮说过什么。")
    else:
        print("第一轮 coachTurns=0 是正常的（还没答过）。")
        print("若追问时仍为 0，把这几行发出来，需要查 pushCoachTurn 有没有被调用。")


if __name__ == "__main__":
    main()
