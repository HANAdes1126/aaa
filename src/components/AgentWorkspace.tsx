import {
  Asterisk,
  Camera,
  Check,
  ChevronDown,
  AudioLines,
  CircleDot,
  Eye,
  EyeOff,
  FileClock,
  FileText,
  Globe2,
  Keyboard,
  Mic,
  MicOff,
  PanelTopClose,
  Plus,
  Settings,
  Square,
  TerminalSquare,
  TriangleAlert,
} from "lucide-react";
import { useLayoutEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AgentChatTurn, CoachMessage, CoachToolTrace } from "../app/types";
import { DESIGN_KIND, stripPictographs } from "../app/coachMessageFormat";
import type { useMeetlyState } from "../app/useMeetlyState";
import { SettingsContent } from "../SettingsApp";

type WorkspaceView = "agent" | "fn" | "dictation" | "meetings" | "logs" | "settings";

type AgentWorkspaceProps = {
  askScreenshot: (question?: string) => Promise<void>;
  clearConversation: () => void;
  closePanel: () => void;
  ctx: ReturnType<typeof useMeetlyState>;
  initialView?: WorkspaceView;
  openFilePicker: () => void;
  startIslandDrag: (event: MouseEvent<HTMLElement>) => Promise<void>;
  toggleSession: () => void;
  toggleStealth: () => Promise<void>;
};

const NAV_ITEMS: Array<{
  icon: typeof Asterisk;
  id: WorkspaceView;
  label: string;
  meta?: string;
}> = [
  { id: "agent", label: "助手", icon: Asterisk },
  { id: "fn", label: "语音提问", icon: AudioLines, meta: "Fn" },
  { id: "dictation", label: "语音输入", icon: Keyboard, meta: "Fn + Space" },
  { id: "meetings", label: "会议记录", icon: FileClock },
  { id: "logs", label: "运行日志", icon: TerminalSquare },
  { id: "settings", label: "设置", icon: Settings },
];

export function AgentWorkspace({
  askScreenshot,
  clearConversation,
  closePanel,
  ctx,
  initialView = "agent",
  openFilePicker,
  startIslandDrag,
  toggleSession,
  toggleStealth,
}: AgentWorkspaceProps) {
  const [view, setView] = useState<WorkspaceView>(initialView);
  // While either pipeline is generating, + becomes "stop and start over" —
  // the button has to stay clickable, that's the only way out of a stall.
  const isBusy = ctx.isAsking || ctx.isCoachThinking;

  return (
    <section className="meetly-workspace" aria-label="Meetly 工作台">
      <aside className="workspace-rail">
        <div className="workspace-brand" title="拖动工作台" onMouseDown={startIslandDrag}>
          <span className="workspace-brand-mark">m</span>
        </div>

        <nav className="workspace-nav" aria-label="工作台导航">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={view === item.id ? "workspace-nav-item is-active" : "workspace-nav-item"}
                aria-current={view === item.id ? "page" : undefined}
                title={item.meta ? `${item.label} · ${item.meta}` : item.label}
                onClick={() => setView(item.id)}
              >
                <Icon />
                <span>{item.label}</span>
                {item.meta && <kbd>{item.meta}</kbd>}
              </button>
            );
          })}
        </nav>

        <div className="workspace-rail-footer">
          <button
            className="workspace-session-button"
            title={ctx.state === "listening" ? "结束会话" : "开始会话"}
            aria-label={ctx.state === "listening" ? "结束会话" : "开始会话"}
            onClick={toggleSession}
          >
            {ctx.state === "listening" ? <MicOff /> : <Mic />}
            <span>{ctx.state === "listening" ? "结束会话" : "开始会话"}</span>
            <span className={ctx.state === "listening" ? "session-dot is-live" : "session-dot"} />
          </button>
          {ctx.sessionKind === "remote" && (
            <button
              className="workspace-record-toggle"
              title="切换录音通道（只录对面 / 对面 + 我的声音）"
              aria-label="切换录音通道"
              onClick={() => ctx.setRecordUserMic((value) => !value)}
            >
              <span className="workspace-record-label">录音</span>
              <span className="workspace-record-value">{ctx.recordUserMic ? "对面+我" : "对面"}</span>
              <ChevronDown />
            </button>
          )}
        </div>
      </aside>

      <div className="workspace-stage">
        <header className="workspace-header">
          <div className="workspace-title">
            <h1>{NAV_ITEMS.find((item) => item.id === view)?.label}</h1>
            {view === "agent" && <span>{ctx.state === "listening" ? "监听中" : "就绪"}</span>}
          </div>
          <div className="workspace-header-actions">
            {view === "agent" && (
              <>
                <button
                  className="workspace-header-button"
                  title={isBusy ? "停止当前回答并新建会话" : "新对话"}
                  aria-label={isBusy ? "停止当前回答并新建会话" : "新对话"}
                  onClick={clearConversation}
                >
                  {isBusy ? <Square /> : <Plus />}
                </button>
                <button className="workspace-header-button" title="上传资料" aria-label="上传资料" onClick={openFilePicker}>
                  <FileText />
                </button>
                <button
                  className="workspace-header-button"
                  title="截图提问"
                  aria-label="截图提问"
                  disabled={ctx.isAsking}
                  onClick={() => void askScreenshot()}
                >
                  <Camera />
                </button>
              </>
            )}
            <button
              className={ctx.isStealthOn ? "workspace-header-button is-active" : "workspace-header-button"}
              title={ctx.isStealthOn ? "当前对屏幕共享不可见" : "当前对屏幕共享可见"}
              aria-label={ctx.isStealthOn ? "切换为可见" : "切换为不可见"}
              aria-pressed={ctx.isStealthOn}
              onClick={() => void toggleStealth()}
            >
              {ctx.isStealthOn ? <EyeOff /> : <Eye />}
            </button>
            <button className="workspace-collapse-button" title="收起工作台" aria-label="收起工作台" onClick={closePanel}>
              <PanelTopClose />
            </button>
          </div>
        </header>

        {view === "agent" ? (
          <div className={ctx.state === "listening" ? "agent-layout" : "agent-layout is-solo"}>
            <div className="agent-column">
              <AgentTimeline
                chatTurns={ctx.agentChatTurns}
                coachDraft={ctx.coachDraft}
                coachMessages={ctx.coachMessages}
                isAsking={ctx.isAsking}
              />
            </div>
            {ctx.state === "listening" && <TranscriptRail ctx={ctx} />}
          </div>
        ) : view === "settings" ? (
          <div className="workspace-settings">
            <SettingsContent
              compact
              onOnboardingCompleted={() => setView("agent")}
            />
          </div>
        ) : (
          <WorkspaceLedger view={view} ctx={ctx} />
        )}
      </div>
    </section>
  );
}

function AgentTimeline({
  chatTurns,
  coachDraft,
  coachMessages,
  isAsking,
}: {
  chatTurns: AgentChatTurn[];
  coachDraft: CoachMessage | null;
  coachMessages: CoachMessage[];
  isAsking: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const entries = useMemo(() => [
    ...coachMessages.map((message) => ({ kind: "proactive" as const, createdAt: message.createdAt, message })),
    ...chatTurns.map((turn) => ({ kind: "chat" as const, createdAt: turn.createdAt, turn })),
  ].sort((left, right) => left.createdAt - right.createdAt), [chatTurns, coachMessages]);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [coachDraft?.text, coachDraft?.toolTraces, entries, isAsking]);

  return (
    <div ref={scrollRef} className="agent-timeline" data-testid="agent-timeline">
      {entries.length === 0 && !coachDraft && (
        <div className="agent-empty-state">
          <span className="agent-empty-glyph"><Asterisk /></span>
          <p>暂无对话</p>
          <span>Agent 会在这里保持同一条上下文。</span>
        </div>
      )}

      {entries.map((entry) => entry.kind === "proactive" ? (
        <AgentMessage
          key={`coach-${entry.message.id}`}
          label="主动"
          text={entry.message.text}
          time={entry.message.createdAt}
          toolTraces={entry.message.toolTraces}
        />
      ) : (
        <div key={entry.turn.id} className="agent-turn">
          <div className="agent-user-message">
            <span>{entry.turn.question}</span>
            <time>{formatTime(entry.turn.createdAt)}</time>
          </div>
          <AgentToolTraceList traces={entry.turn.toolTraces} />
          {entry.turn.error ? (
            <div className="agent-error-message">{entry.turn.error}</div>
          ) : entry.turn.suggestion ? (
            <AgentMessage
              label="回应"
              text={formatSuggestion(entry.turn.suggestion)}
              time={entry.turn.createdAt}
            />
          ) : (
            <div className="agent-thinking-row">
              <span /><span /><span />
              <p>正在处理这条请求</p>
            </div>
          )}
        </div>
      ))}

      {coachDraft && (
        <div className="agent-live-message">
          <div className="agent-message-meta">
            <span><CircleDot />主动</span>
            <span>生成中</span>
          </div>
          <AgentToolTraceList traces={coachDraft.toolTraces} />
          <div className="agent-markdown">
            {coachDraft.text || "正在判断当前时刻最有价值的下一步。"}
          </div>
        </div>
      )}
    </div>
  );
}

function AgentMessage({
  label,
  text,
  time,
  toolTraces = [],
}: {
  label: string;
  text: string;
  time: number;
  toolTraces?: CoachToolTrace[];
}) {
  return (
    <article className="agent-message">
      <div className="agent-message-meta">
        <span><Asterisk />{label}</span>
        <time>{formatTime(time)}</time>
      </div>
      <AgentToolTraceList traces={toolTraces} />
      <div className="agent-markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      </div>
    </article>
  );
}

function AgentToolTraceList({ traces }: { traces: CoachToolTrace[] }) {
  if (traces.length === 0) return null;

  return (
    <div className="agent-tool-list" aria-label="助手工具">
      {traces.map((trace) => {
        const StatusIcon = trace.status === "running"
          ? CircleDot
          : trace.status === "error"
            ? TriangleAlert
            : Check;
        return (
          <div key={trace.id} className={`agent-tool-row is-${trace.status}`}>
            <div className="agent-tool-summary">
              <Globe2 />
              <div>
                <span>{trace.label}</span>
                {trace.query && <p>{trace.query}</p>}
              </div>
              <span className="agent-tool-status">
                <StatusIcon />
                {toolTraceStatus(trace.status)}
              </span>
            </div>
            {trace.content && (
              <details className="agent-tool-result">
                <summary>
                  <ChevronDown />
                  {trace.status === "error" ? "查看错误" : "查看来源"}
                </summary>
                <ToolResultContent content={trace.content} isError={trace.status === "error"} />
              </details>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ToolResultContent({ content, isError }: { content: string; isError: boolean }) {
  if (isError) return <p className="agent-tool-error">{content}</p>;
  const sources = content.split("\n\n").map((block) => {
    const [title, url] = block.split("\n");
    return { title, url };
  });

  return (
    <div className="agent-tool-sources">
      {sources.map((source, index) => source.url?.startsWith("http") ? (
        <a key={`${source.url}-${index}`} href={source.url} target="_blank" rel="noreferrer">
          <span>{source.title || source.url}</span>
          <small>{source.url}</small>
        </a>
      ) : (
        <p key={`${source.title}-${index}`}>{source.title}</p>
      ))}
    </div>
  );
}

function toolTraceStatus(status: CoachToolTrace["status"]) {
  if (status === "running") return "搜索中";
  if (status === "error") return "失败";
  return "已完成";
}

function TranscriptRail({ ctx }: { ctx: ReturnType<typeof useMeetlyState> }) {
  // Compact one-line strip: only the latest transcript (live partial if present,
  // otherwise the newest final segment). History list, header, waveform and
  // footer all removed — the panel previously stole vertical space from the
  // coach answer above and crowded the rail.
  const partial = ctx.partialTranscript;
  const latest = ctx.transcriptHistory.length > 0
    ? ctx.transcriptHistory[ctx.transcriptHistory.length - 1]
    : null;

  let body: React.ReactNode;
  if (partial) {
    body = (
      <div className="transcript-line is-live">
        <span>现在</span>
        <p title={partial.text}>{partial.text}</p>
      </div>
    );
  } else if (latest) {
    const speaker = latest.speaker === "user" ? "我" : "对方";
    body = (
      <div className="transcript-line">
        <span>{speaker}</span>
        <p title={latest.text}>{latest.text}</p>
      </div>
    );
  } else {
    const idleLabel = ctx.state === "listening" ? "正在听" : "已暂停";
    body = (
      <div className="transcript-line transcript-line--empty">
        <span>{idleLabel}</span>
        <p>—</p>
      </div>
    );
  }

  return <aside className="transcript-rail">{body}</aside>;
}

function WorkspaceLedger({ view, ctx }: { view: Exclude<WorkspaceView, "agent">; ctx: ReturnType<typeof useMeetlyState> }) {
  const content = getLedgerContent(view, ctx);
  return (
    <div className="workspace-ledger">
      <div className="ledger-summary">
        <span>{content.kicker}</span>
        <strong>{content.total}</strong>
        <p>{content.caption}</p>
      </div>
      <div className="ledger-list">
        {content.rows.map((row, index) => (
          <article key={`${view}-${index}`} className="ledger-row">
            <time>{row.time}</time>
            <div>
              <h2>{row.title}</h2>
              <p>{row.body}</p>
            </div>
            <span className={row.tone === "live" ? "ledger-status is-live" : "ledger-status"}>{row.status}</span>
          </article>
        ))}
      </div>
    </div>
  );
}

function getLedgerContent(view: Exclude<WorkspaceView, "agent">, ctx: ReturnType<typeof useMeetlyState>) {
  if (view === "fn") {
    return {
      kicker: "语音提问",
      total: "03",
      caption: "今日语音提问",
      rows: [
        { time: "14:32", title: "这个结论有什么漏洞？", body: "基于当前选中文本与屏幕上下文", status: "已回答" },
        { time: "11:08", title: "把这段话解释得更直接", body: "连续对话 · 2 轮", status: "已回答" },
        { time: "09:41", title: "下一步我应该问什么？", body: "会议上下文", status: "已回答" },
      ],
    };
  }
  if (view === "dictation") {
    return {
      kicker: "语音输入",
      total: "18",
      caption: "今日听写片段",
      rows: [
        { time: "15:04", title: "产品评审结论", body: "这一版先收敛 Agent 主线，转录作为辅助信息。", status: "已粘贴" },
        { time: "13:17", title: "项目更新", body: "已经完成桌面端语音链路的端到端验证。", status: "已粘贴" },
        { time: "10:26", title: "快速记录", body: "下周确认会议复盘页的信息结构。", status: "已粘贴" },
      ],
    };
  }
  if (view === "meetings") {
    const active = ctx.state === "listening";
    return {
      kicker: "会议",
      total: String(active ? 13 : 12).padStart(2, "0"),
      caption: "本月会议",
      rows: [
        {
          time: active ? "现在" : "今天",
          title: ctx.meetingGoal || "产品方向讨论",
          body: `${ctx.transcriptHistory.length} 条转录 · ${ctx.coachMessages.length + ctx.agentChatTurns.length} 条 Agent 记录`,
          status: active ? "进行中" : "已结束",
          tone: active ? "live" : undefined,
        },
        { time: "周四", title: "Meetly 体验复盘", body: "42 分钟 · 6 个行动项", status: "已归档" },
        { time: "周二", title: "桌面端语音链路评审", body: "31 分钟 · 3 个行动项", status: "已归档" },
      ],
    };
  }
  return {
    kicker: "运行时",
    total: "24h",
    caption: "本地运行窗口",
    rows: [
      { time: "刚刚", title: "Agent runtime", body: "用户输入通道优先级：100", status: "正常", tone: "live" },
      { time: "2 分前", title: "Audio capture", body: ctx.state === "listening" ? "系统音频与麦克风采集中" : "当前没有活跃采集", status: ctx.state === "listening" ? "运行中" : "空闲" },
      { time: "8 分前", title: "Voice overlay", body: "Fn 与 Fn + Space 使用独立浮层", status: "正常" },
      { time: "今天", title: "Provider health", body: "STT 与 LLM 配置可用", status: "正常" },
    ],
  };
}

function formatSuggestion(suggestion: AgentChatTurn["suggestion"] & {}) {
  if (!suggestion) return "";
  return [
    stripPictographs(suggestion.answer),
    suggestion.bullets.length && suggestion.kind === DESIGN_KIND ? "设计思路" : null,
    suggestion.bullets.length
      ? suggestion.bullets.map((bullet) => `- ${stripPictographs(bullet)}`).join("\n")
      : null,
    suggestion.clarifyingQuestion ? `> ${stripPictographs(suggestion.clarifyingQuestion)}` : null,
  ].filter(Boolean).join("\n\n");
}

function formatTime(timestamp: number) {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false }).format(timestamp);
}
