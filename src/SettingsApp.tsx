import { invoke } from "@tauri-apps/api/core";
import { useCallback, useEffect, useState } from "react";
import { AppearanceSection } from "./settings/AppearanceSection";
import { OnboardingPanel, type OnboardingStatus } from "./settings/OnboardingPanel";
import { UpdateSection } from "./settings/UpdateSection";
import type {
  DictationOutputResult,
  DictationSettings,
  DictationStatus,
} from "./app/dictation/types";
import { DEFAULT_DICTATION_SETTINGS } from "./app/dictation/types";

type ProviderKind = "stt" | "llm";
type ProviderId = "openai_compatible" | "xiaomi_mimo";

type ProviderConfig = {
  providerId: ProviderId;
  baseUrl: string;
  model: string;
};

type ProviderDescriptor = {
  id: ProviderId;
  displayName: string;
  description: string;
  defaultBaseUrl: string;
  defaultModel: string;
};

type DiagnosticResult = {
  success: boolean;
  message: string;
};

type WebSearchSettings = {
  enabled: boolean;
  provider: "exa";
  hasApiKey: boolean;
};

type AudioRunState = "idle" | "listening" | "setup_required" | "error";

type AudioStatus = {
  state: AudioRunState;
  platform: string;
  inputDevice: string | null;
  outputDevice: string | null;
  sampleRate: number | null;
  level: number;
  setupRequired: boolean;
  message: string | null;
};

const FIELD = "ui-field";
const LABEL = "mb-1 block text-xs font-medium text-white/60";
const PRIMARY_BUTTON = "ui-primary-button";
const SECONDARY_BUTTON = "ui-secondary-button";

function useProviderSection(kind: ProviderKind) {
  const [providerId, setProviderId] = useState<ProviderId>("openai_compatible");
  const [providerOptions, setProviderOptions] = useState<ProviderDescriptor[]>([]);
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [hasStoredKey, setHasStoredKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<DiagnosticResult | null>(null);

  const load = useCallback(async () => {
    try {
      const [config, options] = await Promise.all([
        invoke<ProviderConfig>("get_provider_config", { kind }),
        invoke<ProviderDescriptor[]>("list_provider_options", { kind }),
      ]);
      setProviderId(config.providerId);
      setProviderOptions(options);
      setBaseUrl(config.baseUrl);
      setModel(config.model);
      const stored = await invoke<boolean>("has_api_key", { kind });
      setHasStoredKey(stored);
    } catch (error) {
      console.error(`Failed to load ${kind} config:`, error);
    }
  }, [kind]);

  useEffect(() => {
    void load();
  }, [load]);

  const save = useCallback(async () => {
    setIsSaving(true);
    setSaveMessage(null);
    try {
      await invoke("save_provider_config", {
        kind,
        providerId,
        baseUrl,
        model,
        apiKey,
      });
      if (apiKey.trim()) {
        setHasStoredKey(true);
        setApiKey("");
      }
      setSaveMessage("已保存。");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setSaveMessage(`保存失败：${message}`);
    } finally {
      setIsSaving(false);
    }
  }, [kind, providerId, baseUrl, model, apiKey]);

  const selectProvider = (nextProviderId: ProviderId) => {
    setProviderId(nextProviderId);
    const descriptor = providerOptions.find((option) => option.id === nextProviderId);
    if (!descriptor) return;
    setBaseUrl(descriptor.defaultBaseUrl);
    setModel(descriptor.defaultModel);
    setTestResult(null);
  };

  const test = useCallback(async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const command = kind === "stt" ? "test_stt_config" : "test_llm_config";
      const result = await invoke<DiagnosticResult>(command);
      setTestResult(result);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setTestResult({ success: false, message });
    } finally {
      setIsTesting(false);
    }
  }, [kind]);

  return {
    providerId,
    providerOptions,
    selectProvider,
    baseUrl,
    setBaseUrl,
    model,
    setModel,
    apiKey,
    setApiKey,
    hasStoredKey,
    isSaving,
    isTesting,
    saveMessage,
    testResult,
    save,
    test,
  };
}

export function ProviderSection({
  title,
  description,
  kind,
  onSaved,
}: {
  title: string;
  description: string;
  kind: ProviderKind;
  onSaved?: () => void;
}) {
  const section = useProviderSection(kind);
  const save = async () => {
    await section.save();
    onSaved?.();
  };

  return (
    <section className="settings-section">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="section-title">{title}</h2>
          <p className="mt-1 mb-0 max-w-[560px] text-xs leading-relaxed text-white/44">{description}</p>
        </div>
        <span className={`mt-0.5 text-[11px] ${section.hasStoredKey ? "text-[#b9c6cc]" : "text-white/32"}`}>
          {section.hasStoredKey ? "已配置" : "未配置"}
        </span>
      </div>

      <div className="grid grid-cols-[minmax(0,1.4fr)_minmax(180px,0.8fr)] gap-3">
        <label className="col-span-2">
          <span className={LABEL}>服务商</span>
          <select
            className={FIELD}
            value={section.providerId}
            onChange={(event) => section.selectProvider(event.target.value as ProviderId)}
          >
            {section.providerOptions.map((option) => (
              <option key={option.id} value={option.id}>{option.displayName}</option>
            ))}
          </select>
          <span className="mt-1 block text-[11px] leading-relaxed text-white/38">
            {section.providerOptions.find((option) => option.id === section.providerId)?.description}
          </span>
        </label>
        <label>
          <span className={LABEL}>接口地址</span>
          <input
            className={FIELD}
            value={section.baseUrl}
            onChange={(event) => section.setBaseUrl(event.target.value)}
            placeholder="https://api.siliconflow.cn/v1/..."
          />
        </label>
        <label>
          <span className={LABEL}>模型</span>
          <input
            className={FIELD}
            value={section.model}
            onChange={(event) => section.setModel(event.target.value)}
          />
        </label>
      </div>

      <div className="mt-3 mb-4">
        <label className={LABEL}>
          API 密钥 {section.hasStoredKey && <span className="text-white/40">（已保存 — 留空则不修改）</span>}
        </label>
        <input
          className={FIELD}
          type="password"
          value={section.apiKey}
          onChange={(event) => section.setApiKey(event.target.value)}
          placeholder={section.hasStoredKey ? "••••••••" : "sk-..."}
        />
      </div>

      <div className="flex items-center gap-2">
        <button className={PRIMARY_BUTTON} disabled={section.isSaving} onClick={() => void save()}>
          {section.isSaving ? "保存中…" : "保存"}
        </button>
        <button className={SECONDARY_BUTTON} disabled={section.isTesting} onClick={() => void section.test()}>
          {section.isTesting ? "测试中…" : "测试连接"}
        </button>
      </div>

      {section.saveMessage && (
        <p className="mt-2 text-xs text-white/60">{section.saveMessage}</p>
      )}
      {section.testResult && (
        <p className={`mt-2 text-xs ${section.testResult.success ? "text-[#b9c6cc]" : "text-[#ff5c70]"}`}>
          {section.testResult.message}
        </p>
      )}
    </section>
  );
}

function WebSearchSection() {
  const [settings, setSettings] = useState<WebSearchSettings>({
    enabled: false,
    provider: "exa",
    hasApiKey: false,
  });
  const [apiKey, setApiKey] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<DiagnosticResult | null>(null);

  const load = useCallback(async () => {
    try {
      setSettings(await invoke<WebSearchSettings>("get_web_search_config"));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async () => {
    setIsSaving(true);
    setMessage(null);
    setTestResult(null);
    try {
      const next = await invoke<WebSearchSettings>("save_web_search_config", {
        enabled: settings.enabled,
        provider: settings.provider,
        apiKey,
      });
      setSettings(next);
      setApiKey("");
      setMessage("联网搜索设置已保存。");
    } catch (error) {
      setMessage(`保存失败：${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsSaving(false);
    }
  };

  const test = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      setTestResult(await invoke<DiagnosticResult>("test_web_search_config"));
    } catch (error) {
      setTestResult({
        success: false,
        message: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <section className="settings-section">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="section-title">联网搜索</h2>
          <p className="mt-1 mb-0 max-w-[560px] text-xs leading-relaxed text-white/44">
            默认关闭。开启后，Coach 与 Fn Agent 可分别按需搜索公开网页。
          </p>
        </div>
        <span className={`mt-0.5 text-[11px] ${settings.enabled ? "text-[#b9c6cc]" : "text-white/32"}`}>
          {settings.enabled ? "已开启" : "默认关闭"}
        </span>
      </div>

      <label className="settings-row mb-3 justify-between text-sm">
        <span>
          <span className="block text-[13px] font-medium">Exa 联网搜索</span>
          <span className="block text-xs text-white/45">关闭后，两个 Agent 都不会向 Exa 发送搜索请求。</span>
        </span>
        <input
          className="ui-switch"
          type="checkbox"
          checked={settings.enabled}
          onChange={(event) => setSettings((current) => ({ ...current, enabled: event.target.checked }))}
        />
      </label>

      <div className="mb-3">
        <label>
          <span className={LABEL}>搜索服务商</span>
          <select className={FIELD} value={settings.provider} disabled>
            <option value="exa">Exa</option>
          </select>
        </label>
      </div>

      <div className="mb-4">
        <label className={LABEL}>
          Exa API 密钥 {settings.hasApiKey && <span className="text-white/40">（已保存 — 留空则不修改）</span>}
        </label>
        <input
          className={FIELD}
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          placeholder={settings.hasApiKey ? "••••••••" : "exa-..."}
        />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button className={PRIMARY_BUTTON} disabled={isSaving} onClick={() => void save()}>
          {isSaving ? "保存中…" : "保存搜索设置"}
        </button>
        <button
          className={SECONDARY_BUTTON}
          disabled={isTesting || !settings.hasApiKey}
          onClick={() => void test()}
        >
          {isTesting ? "测试中…" : "测试 Exa"}
        </button>
      </div>

      {message && <p className="mt-2 mb-0 text-xs text-white/60">{message}</p>}
      {testResult && (
        <p className={`mt-2 mb-0 text-xs ${testResult.success ? "text-[#b9c6cc]" : "text-[#ff5c70]"}`}>
          {testResult.message}
        </p>
      )}
    </section>
  );
}

export function DiagnosticsSection() {
  const [audioStatus, setAudioStatus] = useState<AudioStatus | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const load = useCallback(async () => {
    setMessage(null);
    try {
      const status = await invoke<AudioStatus>("get_audio_status");
      setAudioStatus(status);
    } catch (error) {
      const nextMessage = error instanceof Error ? error.message : String(error);
      setMessage(nextMessage);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <section className="settings-section">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="section-title">诊断信息</h2>
          <p className="mt-1 mb-0 text-xs text-white/50">
            音频与运行状态，用于本地排查问题。
          </p>
        </div>
        <button className={SECONDARY_BUTTON} onClick={() => void load()}>
          刷新
        </button>
      </div>

      <div>
        <DiagnosticItem label="Tauri 外壳" value="就绪" state="ok" />
        <DiagnosticItem
          label="音频状态"
          value={audioStatus ? audioStatus.state : "检查中…"}
          state={audioStatus?.setupRequired ? "pending" : "ok"}
        />
        <DiagnosticItem
          label="输出设备"
          value={audioStatus?.outputDevice ?? "未找到"}
          state={audioStatus?.outputDevice ? "ok" : "pending"}
        />
        <DiagnosticItem
          label="输入设备"
          value={audioStatus?.inputDevice ?? "未找到"}
          state={audioStatus?.inputDevice ? "ok" : "pending"}
        />
        <DiagnosticItem
          label="采样率"
          value={audioStatus?.sampleRate ? `${audioStatus.sampleRate} Hz` : "未启用"}
          state={audioStatus?.sampleRate ? "ok" : "pending"}
        />
        <DiagnosticItem
          label="音量电平"
          value={audioStatus ? audioStatus.level.toFixed(3) : "未知"}
          state={audioStatus?.state === "listening" ? "ok" : "pending"}
        />
        <DiagnosticItem
          label="平台"
          value={audioStatus?.platform ?? "未知"}
          state={audioStatus ? "ok" : "pending"}
        />
        {audioStatus?.message && (
          <DiagnosticItem label="音频信息" value={audioStatus.message} state="pending" />
        )}
        {message && <DiagnosticItem label="诊断错误" value={message} state="pending" />}
      </div>
    </section>
  );
}

function DictationSection() {
  const [settings, setSettings] = useState<DictationSettings>(DEFAULT_DICTATION_SETTINGS);
  const [status, setStatus] = useState<DictationStatus | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);

  const load = useCallback(async () => {
    try {
      const next = await invoke<DictationStatus>("get_dictation_status");
      setStatus(next);
      setSettings(next.settings);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async () => {
    setIsSaving(true);
    setMessage(null);
    try {
      const next = await invoke<DictationStatus>("save_dictation_settings", { settings });
      setStatus(next);
      setSettings(next.settings);
      setMessage("语音输入设置已保存。");
    } catch (error) {
      setMessage(`保存失败：${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsSaving(false);
    }
  };

  const testPaste = async () => {
    setIsTesting(true);
    setMessage(null);
    try {
      const result = await invoke<DictationOutputResult>("test_dictation_paste");
      setMessage(result.outcome === "pasted" ? "粘贴测试完成。" : result.message);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <section className="settings-section">
      <h2 className="section-title">语音快捷键</h2>
      <p className="mt-1 mb-4 text-xs leading-relaxed text-white/44">
        控制语音输入、自动粘贴与模型整理。
      </p>

      <label className="settings-row mb-3 justify-between text-sm">
        <span>
          <span className="block text-[13px] font-medium">启用语音快捷键</span>
          <span className="block text-xs text-white/45">语音输入与语音提问独立于会议和教练运行。</span>
        </span>
        <input
          className="ui-switch"
          type="checkbox"
          checked={settings.enabled}
          onChange={(event) => setSettings((current) => ({ ...current, enabled: event.target.checked }))}
        />
      </label>

      <div className="mb-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label>
          <span className={LABEL}>主快捷键</span>
          <input
            className={FIELD}
            value={settings.shortcut}
            onChange={(event) => setSettings((current) => ({ ...current, shortcut: event.target.value }))}
            placeholder="Fn+Space"
          />
        </label>
        <label>
          <span className={LABEL}>备用快捷键</span>
          <input
            className={FIELD}
            value={settings.fallbackShortcut}
            onChange={(event) => setSettings((current) => ({ ...current, fallbackShortcut: event.target.value }))}
            placeholder="Alt+Space"
          />
        </label>
      </div>

      <div className="mb-4">
        <DictationToggle
          label="AI 润色"
          description="去除口头禅、修正标点，不会添加新内容。"
          checked={settings.aiPolishEnabled}
          onChange={(checked) => setSettings((current) => ({ ...current, aiPolishEnabled: checked }))}
        />
        <DictationToggle
          label="自动粘贴"
          description="粘贴到开始语音输入时处于前台的应用。"
          checked={settings.autoPasteEnabled}
          onChange={(checked) => setSettings((current) => ({ ...current, autoPasteEnabled: checked }))}
        />
        <DictationToggle
          label="结果保留在剪贴板"
          description="自动粘贴失败时，仍可从剪贴板取回文本。"
          checked={settings.keepResultInClipboard}
          onChange={(checked) => setSettings((current) => ({ ...current, keepResultInClipboard: checked }))}
        />
      </div>

      <div className="mb-3">
        <DiagnosticItem
          label="麦克风权限"
          value={formatMicrophonePermission(status?.microphonePermission)}
          state={status?.microphonePermission === "authorized" ? "ok" : "pending"}
        />
        <DiagnosticItem
          label="键盘权限"
          value={status?.accessibilityGranted ? "已授权" : "Fn 快捷键与自动粘贴需要此权限"}
          state={status?.accessibilityGranted ? "ok" : "pending"}
        />
        <DiagnosticItem
          label="快捷键后端"
          value={status?.shortcutBackend ?? "检查中…"}
          state={status?.shortcutBackend && status.shortcutBackend !== "unavailable" ? "ok" : "pending"}
        />
      </div>

      {status?.shortcutError && <p className="mb-3 text-xs text-[#ff9ba8]">{status.shortcutError}</p>}

      <div className="flex flex-wrap items-center gap-2">
        <button className={PRIMARY_BUTTON} disabled={isSaving} onClick={() => void save()}>
          {isSaving ? "保存中…" : "保存语音设置"}
        </button>
        {!status?.accessibilityGranted && (
          <button
            className={SECONDARY_BUTTON}
            onClick={() =>
              void invoke("request_dictation_accessibility")
                .then(load)
                .catch((error) => setMessage(error instanceof Error ? error.message : String(error)))
            }
          >
            打开键盘权限设置
          </button>
        )}
        <button className={SECONDARY_BUTTON} disabled={isTesting} onClick={() => void testPaste()}>
          {isTesting ? "测试中…" : "测试粘贴"}
        </button>
      </div>
      {message && <p className="mt-2 mb-0 text-xs text-white/60">{message}</p>}
    </section>
  );
}

function formatMicrophonePermission(permission: DictationStatus["microphonePermission"] | undefined) {
  switch (permission) {
    case "authorized":
      return "已授权";
    case "not_determined":
      return "首次录音时询问";
    case "denied":
      return "已在系统设置中拒绝";
    case "restricted":
      return "受 macOS 限制";
    case "unknown":
      return "未知";
    default:
      return "检查中…";
  }
}

function DictationToggle({
  checked,
  description,
  label,
  onChange,
}: {
  checked: boolean;
  description: string;
  label: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="settings-row justify-between">
      <span>
        <span className="block text-[13px] font-medium">{label}</span>
        <span className="block text-xs text-white/45">{description}</span>
      </span>
      <input className="ui-switch" type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function DiagnosticItem({
  label,
  value,
  state,
}: {
  label: string;
  value: string;
  state: "ok" | "pending";
}) {
  return (
    <div className="settings-row">
      <span
        className={`h-4 w-[2px] shrink-0 ${state === "ok" ? "bg-[#9cafb8]" : "bg-white/22"}`}
      />
      <p className="m-0 min-w-[150px] text-[12px] font-medium text-white/62">{label}</p>
      <span className="min-w-0 flex-1 truncate text-right text-[12px] leading-normal text-white/46">{value}</span>
    </div>
  );
}

export function SettingsContent({
  compact = false,
  onOnboardingCompleted,
}: {
  compact?: boolean;
  onOnboardingCompleted?: () => void;
}) {
  const [onboardingStatus, setOnboardingStatus] = useState<OnboardingStatus | null>(null);

  const loadOnboardingStatus = useCallback(async () => {
    try {
      const status = await invoke<OnboardingStatus>("get_onboarding_status");
      setOnboardingStatus(status);
    } catch (error) {
      console.error("加载引导状态失败：", error);
    }
  }, []);

  useEffect(() => {
    void loadOnboardingStatus();
  }, [loadOnboardingStatus]);

  return (
    <div className={compact ? "" : "h-screen w-screen overflow-y-auto bg-[#151718] px-6 py-5"}>
      {!compact && (
        <div className="mb-5">
          <p className="section-label">工作区</p>
          <h1 className="m-0 mt-1 text-base font-semibold text-white/90">设置</h1>
          <p className="mt-1 mb-0 text-xs text-white/42">模型、语音输入、运行状态与应用更新。</p>
        </div>
      )}

      {!onboardingStatus?.completed && (
        <OnboardingPanel
          status={onboardingStatus}
          onCompleted={() => {
            void loadOnboardingStatus();
            onOnboardingCompleted?.();
          }}
        />
      )}

      <div className="settings-stack">
        <AppearanceSection />
        <ProviderSection
          title="语音转文字"
          description="转写麦克风与系统音频；不同 provider 使用各自独立的协议和音频能力。"
          kind="stt"
          onSaved={() => void loadOnboardingStatus()}
        />
        <ProviderSection
          title="助手（LLM）"
          description="生成 Ask、主动建议和语音整理；ASR 与 LLM provider 独立选择。"
          kind="llm"
          onSaved={() => void loadOnboardingStatus()}
        />
        <WebSearchSection />
        <DictationSection />
        <DiagnosticsSection />
        <UpdateSection />
        <FooterActions />
      </div>
    </div>
  );
}

export function SettingsApp() {
  return <SettingsContent />;
}

function FooterActions() {
  // Always the real quit. It used to accept a callback, and the workspace passed
  // its "collapse to island" handler — so the button that says "退出 Meetly"
  // merely folded the panel away.
  const quit = () => {
    void invoke("quit_app");
  };

  return (
    <div className="py-4">
      <button className={SECONDARY_BUTTON} onClick={quit}>
        退出 Meetly
      </button>
    </div>
  );
}
