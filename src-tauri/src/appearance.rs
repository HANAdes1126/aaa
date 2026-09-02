//! Window appearance settings shared by every webview.
//!
//! Ghost mode started life as a localStorage flag, but each Tauri window is a
//! separate WKWebView and does not reliably share `localStorage` or deliver
//! `storage` events across windows — so a slider dragged in Settings never
//! reached the floating island. These settings live on disk and are broadcast
//! with a Tauri event, which every window receives.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use tauri::{AppHandle, Emitter, Manager};

pub const APPEARANCE_EVENT: &str = "appearance_changed";

const INK_RANGE: (f64, f64) = (40.0, 235.0);
const OPACITY_RANGE: (f64, f64) = (0.05, 1.0);
const UI_SCALE_RANGE: (f64, f64) = (0.7, 1.6);
const ISLAND_WIDTH_RANGE: (f64, f64) = (320.0, 1400.0);
const PANEL_WIDTH_RANGE: (f64, f64) = (560.0, 1600.0);
const PANEL_HEIGHT_RANGE: (f64, f64) = (320.0, 1100.0);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppearanceSettings {
    pub ghost_enabled: bool,
    /// Grey level of ghost text (0–255), rendered as an `R G B` triplet.
    pub ink: f64,
    pub strong: f64,
    pub base: f64,
    pub soft: f64,
    /// Scales the whole overlay UI and the windows that host it.
    pub ui_scale: f64,
    /// Collapsed island width, in CSS pixels before `ui_scale`.
    pub island_width: f64,
    pub panel_width: f64,
    pub panel_height: f64,
}

impl Default for AppearanceSettings {
    fn default() -> Self {
        Self {
            ghost_enabled: true,
            ink: 130.0,
            strong: 0.82,
            base: 0.66,
            soft: 0.46,
            ui_scale: 1.0,
            island_width: 600.0,
            panel_width: 920.0,
            panel_height: 600.0,
        }
    }
}

impl AppearanceSettings {
    /// Clamps every field so a corrupted file or a hostile IPC payload can
    /// never produce an invisible, zero-sized or off-screen window.
    pub fn sanitized(self) -> Self {
        Self {
            ghost_enabled: self.ghost_enabled,
            ink: clamp(self.ink, INK_RANGE),
            strong: clamp(self.strong, OPACITY_RANGE),
            base: clamp(self.base, OPACITY_RANGE),
            soft: clamp(self.soft, OPACITY_RANGE),
            ui_scale: clamp(self.ui_scale, UI_SCALE_RANGE),
            island_width: clamp(self.island_width, ISLAND_WIDTH_RANGE),
            panel_width: clamp(self.panel_width, PANEL_WIDTH_RANGE),
            panel_height: clamp(self.panel_height, PANEL_HEIGHT_RANGE),
        }
    }
}

fn clamp(value: f64, (min, max): (f64, f64)) -> f64 {
    if value.is_finite() {
        value.max(min).min(max)
    } else {
        min
    }
}

fn appearance_path() -> Result<PathBuf> {
    Ok(crate::app_state::state_dir()?.join("appearance.json"))
}

/// Reads the settings, falling back to defaults for missing or invalid files.
pub fn load() -> AppearanceSettings {
    let Ok(path) = appearance_path() else {
        return AppearanceSettings::default();
    };
    if !path.exists() {
        return AppearanceSettings::default();
    }

    let Ok(bytes) = fs::read(&path) else {
        return AppearanceSettings::default();
    };

    serde_json::from_slice::<AppearanceSettings>(&bytes)
        .map(|settings| settings.sanitized())
        .unwrap_or_default()
}

fn store(settings: &AppearanceSettings) -> Result<()> {
    let bytes =
        serde_json::to_vec_pretty(settings).context("Failed to serialize appearance settings")?;
    fs::write(appearance_path()?, bytes).context("Failed to write appearance settings")
}

#[tauri::command]
pub fn get_appearance() -> Result<AppearanceSettings, String> {
    Ok(load())
}

#[tauri::command]
pub fn save_appearance(
    app: AppHandle,
    settings: AppearanceSettings,
) -> Result<AppearanceSettings, String> {
    let next = settings.sanitized();
    store(&next).map_err(|error| error.to_string())?;

    // The overlay windows are content-protected, so there is no screenshot to
    // check when a setting "does nothing" — this line is the only evidence.
    let _ = crate::debug_log::append(&format!(
        "[appearance] saved ghost={} ink={} strong={:.2} base={:.2} soft={:.2} scale={:.2} island={} panel={}x{}",
        next.ghost_enabled,
        next.ink,
        next.strong,
        next.base,
        next.soft,
        next.ui_scale,
        next.island_width,
        next.panel_width,
        next.panel_height,
    ));

    // Resize right away so dragging a slider moves the real window, not just
    // the preview. Content protection and page zoom follow the settings for
    // every overlay window, not only the one that owns the toggle.
    crate::window::apply_island_metrics(&app);
    crate::window::apply_overlay_metrics(&app);
    apply_content_protection(&app, next.ghost_enabled);
    apply_zoom(&app, next.ui_scale);

    app.emit(APPEARANCE_EVENT, next)
        .map_err(|error| error.to_string())?;

    Ok(next)
}

fn apply_content_protection(app: &AppHandle, enabled: bool) {
    for label in OVERLAY_WINDOWS {
        if let Some(window) = app.get_webview_window(label) {
            let _ = window.set_content_protected(enabled);
        }
    }
}

/// The floating windows — the settings window keeps a fixed, readable size.
const OVERLAY_WINDOWS: [&str; 2] = ["island", "voice-overlay"];

/**
 * Scales the page itself (`WKWebView.pageZoom` on macOS), so text, spacing and
 * every CSS pixel grow together. The windows are resized by the same factor,
 * which keeps `100vh` layouts filling the window exactly.
 */
pub fn apply_zoom(app: &AppHandle, scale: f64) {
    for label in OVERLAY_WINDOWS {
        if let Some(window) = app.get_webview_window(label) {
            let _ = window.set_zoom(scale);
        }
    }
}

/// Applies everything a freshly created window needs.
pub fn apply_to_windows(app: &AppHandle) {
    let settings = load();
    apply_content_protection(app, settings.ghost_enabled);
    apply_zoom(app, settings.ui_scale);
}

#[cfg(test)]
mod tests {
    use super::{clamp, AppearanceSettings};

    #[test]
    fn out_of_range_values_are_clamped() {
        let settings = AppearanceSettings {
            ink: 900.0,
            strong: 5.0,
            base: -1.0,
            soft: f64::NAN,
            ui_scale: 0.0,
            island_width: 10.0,
            panel_width: 99999.0,
            panel_height: 0.0,
            ..AppearanceSettings::default()
        }
        .sanitized();

        assert_eq!(settings.ink, 235.0);
        assert_eq!(settings.strong, 1.0);
        assert_eq!(settings.base, 0.05);
        assert_eq!(settings.soft, 0.05);
        assert_eq!(settings.ui_scale, 0.7);
        assert_eq!(settings.island_width, 320.0);
        assert_eq!(settings.panel_width, 1600.0);
        assert_eq!(settings.panel_height, 320.0);
    }

    #[test]
    fn non_finite_values_fall_back_to_the_minimum() {
        assert_eq!(clamp(f64::INFINITY, (1.0, 2.0)), 1.0);
    }
}
