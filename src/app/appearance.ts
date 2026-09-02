import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { debugLog, isTauriRuntime, safeInvoke } from "./platform";

/**
 * The class the ghost stylesheet keys off.
 *
 * `applyAppearance` is the only place allowed to put it on an element, and
 * `<html>` is the only element it goes on. A component adding a second copy
 * driven by its own state is how the island ended up ignoring every slider the
 * settings panel wrote — the two copies read different sources and disagreed.
 */
export const GHOST_ROOT_CLASS = "meetly-ghost";

/**
 * Single source of truth for how the overlay looks and how big it is.
 *
 * These settings live on disk (Rust side) instead of in `localStorage`:
 * every Tauri window is its own webview, and the island never received the
 * `storage` events the settings window fired — which is why tuning used to
 * show up in the preview only. Rust persists the value and broadcasts
 * `appearance_changed`, so all windows update the moment a slider moves.
 */

export type AppearanceSettings = {
  ghostEnabled: boolean;
  /** Grey level of ghost text, 0–255, rendered as an `R G B` triplet. */
  ink: number;
  strong: number;
  base: number;
  soft: number;
  /** Scales the whole UI and the windows that host it. */
  uiScale: number;
  /** Collapsed island width in CSS pixels, before `uiScale`. */
  islandWidth: number;
  panelWidth: number;
  panelHeight: number;
};

export const APPEARANCE_EVENT = "appearance_changed";
export const APPEARANCE_STORAGE_KEY = "meetly.appearance";

export const DEFAULT_APPEARANCE: AppearanceSettings = {
  ghostEnabled: true,
  ink: 130,
  strong: 0.82,
  base: 0.66,
  soft: 0.46,
  uiScale: 1,
  islandWidth: 600,
  panelWidth: 920,
  panelHeight: 600,
};

export const APPEARANCE_LIMITS = {
  ink: { min: 40, max: 235, step: 1 },
  opacity: { min: 0.05, max: 1, step: 0.01 },
  uiScale: { min: 0.7, max: 1.6, step: 0.05 },
  islandWidth: { min: 320, max: 1400, step: 10 },
  panelWidth: { min: 560, max: 1600, step: 10 },
  panelHeight: { min: 320, max: 1100, step: 10 },
} as const;

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

/** Mirrors the clamping Rust applies, so previews never lie about the result. */
export function sanitizeAppearance(value: Partial<AppearanceSettings> | null): AppearanceSettings {
  const source = value ?? {};
  const number = (raw: unknown, range: { min: number; max: number }, fallback: number) =>
    clamp(typeof raw === "number" ? raw : fallback, range.min, range.max);

  return {
    ghostEnabled: source.ghostEnabled !== false,
    ink: number(source.ink, APPEARANCE_LIMITS.ink, DEFAULT_APPEARANCE.ink),
    strong: number(source.strong, APPEARANCE_LIMITS.opacity, DEFAULT_APPEARANCE.strong),
    base: number(source.base, APPEARANCE_LIMITS.opacity, DEFAULT_APPEARANCE.base),
    soft: number(source.soft, APPEARANCE_LIMITS.opacity, DEFAULT_APPEARANCE.soft),
    uiScale: number(source.uiScale, APPEARANCE_LIMITS.uiScale, DEFAULT_APPEARANCE.uiScale),
    islandWidth: number(
      source.islandWidth,
      APPEARANCE_LIMITS.islandWidth,
      DEFAULT_APPEARANCE.islandWidth,
    ),
    panelWidth: number(
      source.panelWidth,
      APPEARANCE_LIMITS.panelWidth,
      DEFAULT_APPEARANCE.panelWidth,
    ),
    panelHeight: number(
      source.panelHeight,
      APPEARANCE_LIMITS.panelHeight,
      DEFAULT_APPEARANCE.panelHeight,
    ),
  };
}

function readLocal(): AppearanceSettings | null {
  try {
    const raw = window.localStorage.getItem(APPEARANCE_STORAGE_KEY);
    return raw ? sanitizeAppearance(JSON.parse(raw) as Partial<AppearanceSettings>) : null;
  } catch {
    return null;
  }
}

function writeLocal(settings: AppearanceSettings): void {
  try {
    window.localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    /* Storage is best-effort; the settings still apply for the current session. */
  }
}

/**
 * Last known value, shared by every hook in this window. `patchAppearance`
 * needs a full payload to send, and reading it from here avoids an extra
 * round trip on every slider tick.
 */
let cached: AppearanceSettings | null = null;

export async function loadAppearance(): Promise<AppearanceSettings> {
  if (isTauriRuntime()) {
    try {
      const settings = await safeInvoke<AppearanceSettings>("get_appearance");
      if (settings) {
        cached = sanitizeAppearance(settings);
        writeLocal(cached);
        return cached;
      }
    } catch {
      /* Fall through to the local copy when the command is unavailable. */
    }
  }

  cached = readLocal() ?? sanitizeAppearance(null);
  return cached;
}

export async function saveAppearance(
  settings: AppearanceSettings,
): Promise<AppearanceSettings> {
  const next = sanitizeAppearance(settings);
  cached = next;
  writeLocal(next);

  if (!isTauriRuntime()) {
    return next;
  }

  try {
    const saved = await safeInvoke<AppearanceSettings>("save_appearance", { settings: next });
    if (saved) {
      cached = sanitizeAppearance(saved);
    }
  } catch (error) {
    console.error("Failed to save appearance:", error);
  }

  return cached;
}

/** Applies a partial change on top of the last known settings. */
export async function patchAppearance(
  patch: Partial<AppearanceSettings>,
): Promise<AppearanceSettings> {
  const base = cached ?? (await loadAppearance());
  return saveAppearance({ ...base, ...patch });
}

/**
 * Publishes the settings onto the document: the ghost class, the four ghost
 * CSS variables and the UI zoom.
 *
 * Zoom goes on `<html>` rather than a wrapper because the layout is built
 * from `100vh`/`100vw`; CSS zoom rescales the initial containing block, so
 * the panel keeps filling the window exactly at any scale.
 */
export function applyAppearance(
  settings: AppearanceSettings,
  options?: { zoom?: boolean },
): void {
  const root = document.documentElement;
  if (!root) return;

  root.classList.toggle(GHOST_ROOT_CLASS, settings.ghostEnabled);

  const ink = Math.round(settings.ink);
  root.style.setProperty("--ghost-ink", `${ink} ${ink} ${ink}`);
  root.style.setProperty("--ghost-strong", settings.strong.toFixed(2));
  root.style.setProperty("--ghost-base", settings.base.toFixed(2));
  root.style.setProperty("--ghost-soft", settings.soft.toFixed(2));

  void reportRender(windowLabel(), settings);

  // Inside Tauri the Rust side owns zoom (`WKWebView.pageZoom`, applied to the
  // window and its layout together). CSS zoom is only the fallback for a plain
  // browser session, where no native window exists.
  if (!isTauriRuntime() && options?.zoom !== false) {
    root.style.zoom = Math.abs(settings.uiScale - 1) < 0.001 ? "" : String(settings.uiScale);
  }
}

/** Which window this webview is: the log is the only way to see the overlays. */
function windowLabel(): string {
  const path = window.location.pathname;
  if (path.includes("voice-overlay")) return "voice";
  if (path.includes("settings")) return "settings";
  return "island";
}

/**
 * Stand-in for real body copy.
 *
 * The live panels only exist once a session is open, and the settings preview
 * is deliberately exempt from the ghost rules, so while the settings surface is
 * up neither one can say what an actual transcript line resolves to — which is
 * exactly the case that hid the original bug. This probes the cascade itself:
 * same classes, same ancestor, real computed colour, whatever the view is
 * showing. Appended, read and removed inside one tick, so it never paints.
 */
function probeColor(outerClass: string, tag: string): string {
  const holder = document.createElement("div");
  holder.className = outerClass;
  const inner = document.createElement(tag);
  inner.textContent = "probe";
  holder.appendChild(inner);

  // Directly under <html>: it inherits the ghost class like a real panel, and
  // it stays outside .workspace-settings, which restores its own light colours.
  document.documentElement.appendChild(holder);
  try {
    return getComputedStyle(inner).color;
  } finally {
    holder.remove();
  }
}

/**
 * The overlay windows are content-protected, so no screenshot can show whether
 * a slider actually changed anything. Writing the values the browser reports
 * back — not the ones we sent — is the only objective evidence.
 */
function reportRender(label: string, settings: AppearanceSettings): void {
  if (!isTauriRuntime()) return;

  const root = document.documentElement;
  const styles = getComputedStyle(root);
  const read = (name: string) => styles.getPropertyValue(name).trim() || "?";

  const sample = (selector: string, { insidePreview }: { insidePreview: boolean }) => {
    const found = Array.from(document.querySelectorAll(selector)).find((element) =>
      insidePreview
        ? Boolean(element.closest(".ghost-preview-stage"))
        : !element.closest(".ghost-preview-stage"),
    );
    return found ? getComputedStyle(found).color : "no-target";
  };

  // The live panels and the settings preview are reported separately: the
  // preview is the only thing visible while the settings surface is open, and
  // it wears the ghost class on purpose, so lumping it in with the real panels
  // once reported a healthy overlay no matter what the panels were doing.
  const live = sample(".agent-markdown, .transcript-line, .meetly-island", { insidePreview: false });
  const preview = sample(".agent-markdown", { insidePreview: true });
  const shell = document.querySelector(".workspace-settings");

  // `body` is the tier the agent answers render in, `meta` the timestamps and
  // hints. Together they prove the sliders reach the real text rules and not
  // just the preview, whatever is on screen.
  const body = probeColor("agent-markdown", "p");
  const meta = probeColor("agent-message-meta", "span");

  debugLog(
    `[render:${label}] ghost-class=${root.classList.contains(GHOST_ROOT_CLASS)} ` +
      `ink=${read("--ghost-ink")} base=${read("--ghost-base")} ` +
      `strong=${read("--ghost-strong")} soft=${read("--ghost-soft")} ` +
      `| asked ink=${Math.round(settings.ink)} ` +
      `| body=${body} meta=${meta} ` +
      `| live=${live} preview=${preview} ` +
      `settings-shell=${shell ? getComputedStyle(shell).backgroundColor : "no-shell"}`,
  );
}

/** Slider-friendly inline style for the ghost preview surfaces. */
export function ghostPreviewStyle(settings: AppearanceSettings): CSSProperties {
  const ink = Math.round(settings.ink);
  return {
    "--ghost-ink": `${ink} ${ink} ${ink}`,
    "--ghost-strong": settings.strong.toFixed(2),
    "--ghost-base": settings.base.toFixed(2),
    "--ghost-soft": settings.soft.toFixed(2),
  } as CSSProperties;
}

export type AppearanceStore = {
  settings: AppearanceSettings;
  isReady: boolean;
  update: (patch: Partial<AppearanceSettings>) => void;
};

/**
 * Reads the settings on mount, keeps them in sync with every other window via
 * the `appearance_changed` event, and throttles writes so dragging a slider
 * does not flood the IPC channel.
 */
export function useAppearance(): AppearanceStore {
  const [settings, setSettings] = useState<AppearanceSettings>(
    () => cached ?? readLocal() ?? DEFAULT_APPEARANCE,
  );
  const [isReady, setIsReady] = useState(false);
  const latest = useRef(settings);
  const pending = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sent = useRef<string | null>(null);

  latest.current = settings;

  useEffect(() => {
    void loadAppearance().then((next) => {
      setSettings(next);
      setIsReady(true);
    });
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;

    let unlisten: UnlistenFn | null = null;
    void listen<AppearanceSettings>(APPEARANCE_EVENT, (event) => {
      const next = sanitizeAppearance(event.payload);
      debugLog(`[event:${windowLabel()}] got ghost=${next.ghostEnabled} ink=${Math.round(next.ink)}`);
      setSettings((current) =>
        JSON.stringify(current) === JSON.stringify(next) ? current : next,
      );
    }).then((fn) => {
      unlisten = fn;
    });

    return () => {
      unlisten?.();
    };
  }, []);

  const commit = useCallback((next: AppearanceSettings) => {
    cached = next;
    // First change lands immediately (snappy feedback), the rest are
    // coalesced so a drag produces a handful of saves instead of hundreds.
    if (pending.current) return;

    sent.current = JSON.stringify(next);
    void saveAppearance(next);

    pending.current = setTimeout(() => {
      pending.current = null;

      // The trailing write only exists to flush the tail of a drag. Re-sending
      // an unchanged payload would be pure waste: every save makes the Rust
      // side re-measure and resize the island, so a single slider tick used to
      // move the window twice.
      const snapshot = JSON.stringify(latest.current);
      if (snapshot === sent.current) return;

      sent.current = snapshot;
      void saveAppearance(latest.current);
    }, 120);
  }, []);

  const update = useCallback(
    (patch: Partial<AppearanceSettings>) => {
      // Derived from the ref, not inside a state updater: updaters must stay
      // pure, and React may run them twice.
      const next = sanitizeAppearance({ ...latest.current, ...patch });
      latest.current = next;
      setSettings(next);
      commit(next);
    },
    [commit],
  );

  useEffect(
    () => () => {
      if (pending.current) clearTimeout(pending.current);
    },
    [],
  );

  return { settings, isReady, update };
}

/**
 * `useAppearance` plus the DOM side effects an overlay window needs. The
 * settings window uses the plain hook: it previews values instead of wearing
 * them.
 */
export function useOverlayAppearance(): AppearanceStore {
  const store = useAppearance();

  useEffect(() => {
    applyAppearance(store.settings);
  }, [store.settings]);

  return store;
}
