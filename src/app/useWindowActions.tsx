import { useCallback, useEffect, useMemo, type MouseEvent } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { loadAppearance, patchAppearance } from "./appearance";
import { debugLog, isTauriRuntime, safeInvoke } from "./platform";
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

  // Ghost mode stops the island from stealing the foreground app, which also
  // means an un-activated window never receives keystrokes — every text field
  // in the panel would be dead weight. Typing is the one case where taking
  // focus is both unavoidable and expected, so it is granted on demand from a
  // single delegated listener instead of wiring each field by hand.
  //
  // `pointerdown` fires before the caret lands and works even when the window
  // is not activated; `focusin` covers reaching a field with Tab. A short
  // window keeps the pair from invoking twice for the same interaction.
  const stealthOn = ctx.isStealthOn;

  useEffect(() => {
    if (!stealthOn || !isTauriRuntime()) {
      return;
    }

    let lastActivationAt = 0;

    const activateForTyping = (target: EventTarget | null) => {
      const element = target as HTMLElement | null;
      if (!element) {
        return;
      }

      const wantsKeyboard =
        element.tagName === "INPUT" ||
        element.tagName === "TEXTAREA" ||
        element.isContentEditable;

      if (!wantsKeyboard) {
        return;
      }

      const now = Date.now();
      if (now - lastActivationAt < 400) {
        return;
      }
      lastActivationAt = now;

      debugLog(`[island] typing target=${element.tagName} -> activate_island`);
      void safeInvoke("activate_island").catch((error) => {
        console.error("Failed to activate island for typing:", error);
      });
    };

    // The caret leaving the fields is the cue that typing is over, so the
    // foreground goes back to the meeting app. Moving between two fields keeps
    // it (otherwise every Tab would bounce the shared window back and forth).
    const handleFocusOut = (event: FocusEvent) => {
      const next = event.relatedTarget as HTMLElement | null;
      const stillTyping =
        !!next &&
        (next.tagName === "INPUT" ||
          next.tagName === "TEXTAREA" ||
          next.isContentEditable);

      if (stillTyping) {
        return;
      }

      debugLog("[island] caret left the fields -> release_island_focus");
      void safeInvoke("release_island_focus").catch((error) => {
        console.error("Failed to release island focus:", error);
      });
    };

    const handlePointerDown = (event: Event) => activateForTyping(event.target);
    const handleFocusIn = (event: Event) => activateForTyping(event.target);

    document.addEventListener("pointerdown", handlePointerDown, true);
    document.addEventListener("focusin", handleFocusIn);
    document.addEventListener("focusout", handleFocusOut);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown, true);
      document.removeEventListener("focusin", handleFocusIn);
      document.removeEventListener("focusout", handleFocusOut);
    };
  }, [stealthOn]);

  // Dragging moves the window from the Rust side rather than through the
  // native `startDragging()`. On Windows that call hands the move loop to the
  // shell, and entering the loop activates the window — one drag would undo
  // ghost mode for the rest of the session. The Rust command follows the
  // cursor with `SWP_NOACTIVATE` instead, so the meeting app keeps the
  // foreground. `startDragging` stays as the fallback for a runtime that does
  // not have the command (an older installed build).
  const startIslandDrag = useCallback(async (event: MouseEvent<HTMLElement>) => {
    if (event.button !== 0 || !isTauriRuntime()) {
      return;
    }

    event.preventDefault();

    debugLog("[island] drag start -> drag_island");

    try {
      await safeInvoke("drag_island");
    } catch (error) {
      console.error("Failed to drag island without activating:", error);

      try {
        await getCurrentWindow().startDragging();
      } catch (fallbackError) {
        console.error("Failed to start island drag:", fallbackError);
      }
    }
  }, []);

  const resizeIsland = useCallback(async (expanded: boolean) => {
    await safeInvoke("set_island_height", { height: expanded ? 600 : 54 });
  }, []);

  const setPanel = useCallback(
    async (panel: OpenPanel) => {
      ctx.setOpenPanel(panel);
      await resizeIsland(panel !== null);

      // Collapsing tears the fields down, which can skip `focusout` entirely —
      // release here too so a closed panel never keeps the foreground.
      if (panel === null && stealthOn) {
        await safeInvoke("release_island_focus").catch((error) => {
          console.error("Failed to release island focus:", error);
        });
      }
    },
    [ctx, resizeIsland, stealthOn]
  );

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
