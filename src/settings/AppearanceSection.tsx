import { useCallback, useMemo } from "react";
import {
  APPEARANCE_LIMITS,
  DEFAULT_APPEARANCE,
  GHOST_ROOT_CLASS,
  ghostPreviewStyle,
  useAppearance,
  type AppearanceSettings,
} from "../app/appearance";

type RangeKey = "ink" | "strong" | "base" | "soft";
type SizeKey = "uiScale" | "islandWidth" | "panelWidth" | "panelHeight";

const GHOST_SLIDERS: {
  key: RangeKey;
  label: string;
  hint: string;
  format: (value: number) => string;
}[] = [
  { key: "ink", label: "文字灰度", hint: "数值越高越亮、越接近普通文字", format: (value) => String(Math.round(value)) },
  { key: "strong", label: "强调文字", hint: "标题、AI 建议正文", format: (value) => value.toFixed(2) },
  { key: "base", label: "正文文字", hint: "转写、列表、常规说明", format: (value) => value.toFixed(2) },
  { key: "soft", label: "次要文字", hint: "时间戳、元信息、弱化说明", format: (value) => value.toFixed(2) },
];

const SIZE_SLIDERS: { key: SizeKey; label: string; hint: string }[] = [
  {
    key: "uiScale",
    label: "整体缩放",
    hint: "同时放大文字与窗口，适配高分屏或需要更贴近屏幕尺寸的场合",
  },
  { key: "islandWidth", label: "折叠条宽度", hint: "未展开时悬浮岛的横向宽度" },
  { key: "panelWidth", label: "面板宽度", hint: "展开后的工作台宽度" },
  { key: "panelHeight", label: "面板高度", hint: "展开后的工作台高度" },
];

const LIMITS: Record<RangeKey | SizeKey, { min: number; max: number; step: number }> = {
  ink: APPEARANCE_LIMITS.ink,
  strong: APPEARANCE_LIMITS.opacity,
  base: APPEARANCE_LIMITS.opacity,
  soft: APPEARANCE_LIMITS.opacity,
  uiScale: APPEARANCE_LIMITS.uiScale,
  islandWidth: APPEARANCE_LIMITS.islandWidth,
  panelWidth: APPEARANCE_LIMITS.panelWidth,
  panelHeight: APPEARANCE_LIMITS.panelHeight,
};

function formatSize(key: SizeKey, value: number): string {
  if (key === "uiScale") return `${Math.round(value * 100)}%`;
  return `${Math.round(value)} px`;
}

function Slider({
  hint,
  label,
  max,
  min,
  onChange,
  step,
  value,
  valueLabel,
}: {
  hint: string;
  label: string;
  max: number;
  min: number;
  onChange: (value: number) => void;
  step: number;
  value: number;
  valueLabel: string;
}) {
  return (
    <div className="ghost-slider-row">
      <div className="ghost-slider-head">
        <span className="text-[12px] font-medium">{label}</span>
        <span className="ghost-slider-value">{valueLabel}</span>
      </div>
      <input
        className="ui-range"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span className="block text-[11px] leading-relaxed text-white/38">{hint}</span>
    </div>
  );
}

export function AppearanceSection() {
  const { settings, isReady, update } = useAppearance();

  // These sliders *are* the ghost look, so moving one while ghost mode is off
  // would look like a broken control. Turn it on instead: the user asked for
  // the effect, show it to them.
  const setGhost = useCallback(
    (key: RangeKey, value: number) =>
      update(
        settings.ghostEnabled
          ? ({ [key]: value } as Partial<AppearanceSettings>)
          : ({ [key]: value, ghostEnabled: true } as Partial<AppearanceSettings>),
      ),
    [settings.ghostEnabled, update],
  );

  const setSize = useCallback(
    (key: SizeKey, value: number) => update({ [key]: value } as Partial<AppearanceSettings>),
    [update],
  );

  const toggleGhost = useCallback(() => {
    update({ ghostEnabled: !settings.ghostEnabled });
  }, [settings.ghostEnabled, update]);

  // Scoped resets: the ghost button sits inside a block that is disabled while
  // ghost mode is off, so it must not be the only way to restore the sizes.
  const resetGhost = useCallback(() => {
    update({
      ink: DEFAULT_APPEARANCE.ink,
      strong: DEFAULT_APPEARANCE.strong,
      base: DEFAULT_APPEARANCE.base,
      soft: DEFAULT_APPEARANCE.soft,
    });
  }, [update]);

  const resetSizes = useCallback(() => {
    update({
      uiScale: DEFAULT_APPEARANCE.uiScale,
      islandWidth: DEFAULT_APPEARANCE.islandWidth,
      panelWidth: DEFAULT_APPEARANCE.panelWidth,
      panelHeight: DEFAULT_APPEARANCE.panelHeight,
    });
  }, [update]);

  const previewStyle = useMemo(() => ghostPreviewStyle(settings), [settings]);

  const ghostIsDefault = GHOST_SLIDERS.every(({ key }) => settings[key] === DEFAULT_APPEARANCE[key]);
  const sizesAreDefault = SIZE_SLIDERS.every(({ key }) => settings[key] === DEFAULT_APPEARANCE[key]);

  return (
    <section className="settings-section">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h2 className="section-title">外观与隐身</h2>
          <p className="mt-1 mb-0 max-w-[560px] text-xs leading-relaxed text-white/44">
            隐身模式会去掉所有面板底色、边框和阴影，并把文字压成中灰。所有滑杆实时生效于悬浮岛、工作台与语音浮窗，
            无需重启应用。
          </p>
        </div>
        <span className={`mt-0.5 text-[11px] ${settings.ghostEnabled ? "text-[#b9c6cc]" : "text-white/32"}`}>
          {settings.ghostEnabled ? "隐身中" : "常规"}
        </span>
      </div>

      <label className="settings-row justify-between">
        <span>
          <span className="block text-[13px] font-medium">隐身模式</span>
          <span className="block text-xs text-white/45">
            关闭后恢复原始的毛玻璃面板与配色。悬浮岛上的眼睛图标可随时切换。
          </span>
        </span>
        <input className="ui-switch" type="checkbox" checked={settings.ghostEnabled} onChange={toggleGhost} />
      </label>

      <div className="mt-3">
        <p className="mt-0 mb-2 text-[11px] leading-relaxed text-white/38">
          拖动即刻作用于当前界面。隐身模式关闭时拖动任意一项会自动开启，否则改动无从显现。
        </p>
        {GHOST_SLIDERS.map(({ key, label, hint, format }) => (
          <Slider
            key={key}
            hint={hint}
            label={label}
            max={LIMITS[key].max}
            min={LIMITS[key].min}
            onChange={(value) => setGhost(key, value)}
            step={LIMITS[key].step}
            value={settings[key]}
            valueLabel={format(settings[key])}
          />
        ))}

        <div className="mt-3 flex items-center gap-2">
          <button className="ui-secondary-button" onClick={resetGhost} disabled={ghostIsDefault}>
            恢复隐身默认值
          </button>
          {!isReady && <span className="text-[11px] text-white/38">正在读取已保存的设置…</span>}
        </div>
      </div>

      <div className="mt-5">
        <span className="mb-1 block text-xs font-medium text-white/60">界面尺寸</span>
        <p className="mt-0 mb-2 text-[11px] leading-relaxed text-white/38">
          拖动即可改变悬浮岛与工作台的实际窗口大小，位置保持不变。超过当前屏幕可容纳范围时会自动收敛到屏幕内，不会把面板拖出可视区。
        </p>
        {SIZE_SLIDERS.map(({ key, label, hint }) => (
          <Slider
            key={key}
            hint={hint}
            label={label}
            max={LIMITS[key].max}
            min={LIMITS[key].min}
            onChange={(value) => setSize(key, value)}
            step={LIMITS[key].step}
            value={settings[key]}
            valueLabel={formatSize(key, settings[key])}
          />
        ))}
        <div className="mt-3 flex items-center gap-2">
          <button className="ui-secondary-button" onClick={resetSizes} disabled={sizesAreDefault}>
            恢复默认尺寸
          </button>
        </div>
      </div>

      <div className="mt-4">
        <span className="mb-1.5 block text-xs font-medium text-white/60">实时预览</span>
        <div className="ghost-preview-stage">
          <div className={GHOST_ROOT_CLASS} style={previewStyle}>
            <div className="app-panel bg-[rgb(19_21_22_/_0.86)] p-3">
              <div className="mb-1.5 flex items-center gap-2">
                <span className="session-dot" />
                <span className="section-label">Copilot · 刚才</span>
              </div>
              <p className="agent-markdown m-0 text-[13px] leading-relaxed">
                对方提到预算要下季度才批，可以先问清决策链路，再决定是否给阶梯报价。
              </p>
              <div className="mt-2.5 flex gap-2">
                <button className="ui-primary-button">采纳</button>
                <button className="ui-secondary-button">忽略</button>
              </div>
            </div>
          </div>
          <span className="ghost-preview-caption">
            预览固定以隐身样式呈现，便于对照调参；背景为模拟桌面，用于判断实际隐蔽程度
          </span>
        </div>
      </div>
    </section>
  );
}
