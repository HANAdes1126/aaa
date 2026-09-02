import { useCallback, useEffect, useMemo, type MouseEvent } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { loadAppearance, patchAppearance } from "./appearance";
import { isTauriRuntime, safeInvoke } from "./platform";
import type { OpenPanel } from "./types";
import type { MeetlyState } from "./useMeetlyState";

export function useWindowActions(ctx: MeetlyState) {
  // Ghost mode defaults on, so push the persisted value to the native window on
  // mount instead of waiting for the first toggle. `setIsStealthOn` is a stable
  // state setter, so this runs once per window rather than on every render.
  //
  // Later changes arrive as the `appearance_changed` event the Rust side
  // broadcasts — the old `storage` listener never fired, because each Tauri
  // window keeps its own `localStorage`.
  const setIsStealthOn = ctx.setIsStealthOn;

  useEffect(() => {
    void loadAppearance()
      .then((settings) => {
        setIsStealthOn(settings.ghostEnabled);
        return safeInvoke("set_stealth", { enabled: settings.ghostEnabled });
      })
      .catch(() => {
        /* The native window does not exist in a plain browser dev session. */
      });
  }, [setIsStealthOn]);

  const startIslandDrag = useCallback(async (event: MouseEvent<HTMLElement>) => {
    if (event.button !== 0 || !isTauriRuntime()) {
      return;
    }

    event.preventDefault();

    try {
      await getCurrentWindow().startDragging();
    } catch (error) {
      console.error("Failed to start island drag:", error);
    }
  }, []);

  const resizeIsland = useCallback(async (expanded: boolean) => {
    await safeInvoke("set_island_height", { height: expanded ? 600 : 54 });
  }, []);

  const setPanel = useCallback(async (panel: OpenPanel) => {
    ctx.setOpenPanel(panel);
    await resizeIsland(panel !== null);
  }, [ctx, resizeIsland]);

  const toggleHidden = useCallback(async () => {
    ctx.setIsHidden((current) => !current);
    await safeInvoke("set_island_visible", { visible: ctx.isHidden });
  }, [ctx]);

  const toggleStealth = useCallback(async () => {
    const next = !ctx.isStealthOn;
    ctx.setIsStealthOn(next);

    try {
      await safeInvoke("set_stealth", { enabled: next });
    } catch (error) {
      console.error("Failed to toggle stealth mode:", error);
    }

    // Persist the toggle so the settings window and the voice overlay pick it
    // up immediately, instead of waiting for their next focus event.
    try {
      await patchAppearance({ ghostEnabled: next });
    } catch (error) {
      console.error("Failed to persist stealth mode:", error);
    }
  }, [ctx]);

  const openSettings = useCallback(async () => {
    await setPanel("settings");
  }, [setPanel]);

  const status = useMemo(() => {
    if (ctx.state === "error") {
      return {
        icon: <AlertCircle className="h-3.5 w-3.5 text-[#ff5c70]" />,
        label: "需要处理",
        className: "text-[#ff5c70]",
      };
    }

    if (ctx.state === "thinking" || ctx.state === "transcribing") {
      return {
        icon: <Loader2 className="h-3.5 w-3.5 animate-spin" />,
        label: ctx.state === "thinking" ? "准备中" : "转写中",
        className: "text-white/70",
      };
    }

    if (ctx.state === "listening") {
      return {
        icon: (
          <span className="inline-block h-2 w-2 shrink-0 rounded-full bg-[#38d879] shadow-[0_0_0_0_rgb(56_216_121_/_0.42)] [animation:listening-dot-pulse_1.4s_infinite]" />
        ),
        label: ctx.sessionKind === "remote" ? "远程会议中" : "现场会议中",
        className: "text-[#38d879]",
      };
    }

    return {
      icon: null,
      label: "",
      className: "",
    };
  }, [ctx.sessionKind, ctx.state]);

  return {
    openSettings,
    resizeIsland,
    setPanel,
    startIslandDrag,
    status,
    toggleHidden,
    toggleStealth,
  };
}

export type WindowActions = ReturnType<typeof useWindowActions>;
