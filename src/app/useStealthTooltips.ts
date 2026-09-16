import { useEffect } from "react";

/**
 * Hides native `title` tooltips while the overlay is in stealth mode.
 *
 * Content protection (`set_content_protected`, i.e. `NSWindow.sharingType` /
 * `WDA_EXCLUDEFROMCAPTURE`) keeps the window out of screen capture, but the
 * tooltip the browser pops up on hover is not covered by it: hovering the gear
 * still shows a readable "设置" bubble to everyone on the call, which defeats
 * the point of turning stealth on.
 *
 * There is no CSS switch for native tooltips, so the attribute itself has to
 * go. Values are stashed on the element and restored when stealth is turned
 * off — `aria-label` is left untouched, so screen readers keep working.
 *
 * A single initial pass is not enough: React rewrites `title` whenever a
 * prop behind it changes (a state message, a filename), so a MutationObserver
 * re-applies the suppression on every later write.
 */

const STASH_ATTRIBUTE = "data-meetly-stealth-title";

function suppressTitles() {
  document.querySelectorAll<HTMLElement>("[title]").forEach((element) => {
    element.setAttribute(STASH_ATTRIBUTE, element.getAttribute("title") ?? "");
    element.removeAttribute("title");
  });
}

function restoreTitles() {
  document
    .querySelectorAll<HTMLElement>(`[${STASH_ATTRIBUTE}]`)
    .forEach((element) => {
      element.setAttribute("title", element.getAttribute(STASH_ATTRIBUTE) ?? "");
      element.removeAttribute(STASH_ATTRIBUTE);
    });
}

/**
 * Mount in the floating windows only. The settings window is a normal window
 * and is never content-protected, so its tooltips stay.
 */
export function useStealthTooltips(enabled: boolean) {
  useEffect(() => {
    if (!enabled) return;

    suppressTitles();

    let frame = 0;
    const schedule = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        suppressTitles();
      });
    };

    const observer = new MutationObserver(schedule);
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["title"],
    });

    return () => {
      observer.disconnect();
      if (frame) cancelAnimationFrame(frame);
      restoreTitles();
    };
  }, [enabled]);
}
