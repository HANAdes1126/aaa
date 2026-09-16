#[cfg(target_os = "windows")]
use windows::Win32::Foundation::HWND;

#[derive(Debug, Clone)]
pub struct TargetSnapshot {
    pub pid: i32,
    pub app_name: Option<String>,
    #[cfg(target_os = "macos")]
    focused_element: Option<FocusedElementSnapshot>,
    #[cfg(target_os = "windows")]
    focused: Option<FocusedWindowSnapshot>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct FocusedElementSnapshot(accessibility::AXUIElement);

/// The Windows counterpart of `FocusedElementSnapshot`.
///
/// Only the raw window handle is kept, and it is stored as an `isize` rather
/// than an `HWND` so the snapshot stays `Send` without an extra unsafe impl.
/// The HWND is resolved back at restore time, which also avoids holding a
/// window handle across an arbitrary delay.
#[cfg(target_os = "windows")]
#[derive(Debug, Clone, Copy)]
struct FocusedWindowSnapshot {
    /// `GUITHREADINFO.hwndFocus`: the control that actually owned the keyboard,
    /// not its top-level window. Restoring the top-level window instead would
    /// land the caret on the frame rather than back in the text box.
    focus: isize,
}

#[cfg(target_os = "windows")]
impl FocusedWindowSnapshot {
    fn hwnd(self) -> HWND {
        HWND(self.focus as *mut core::ffi::c_void)
    }
}

// AXUIElement is a retained Core Foundation reference. macOS Accessibility
// messaging supports using the reference from the output worker thread.
#[cfg(target_os = "macos")]
unsafe impl Send for FocusedElementSnapshot {}

#[cfg(target_os = "macos")]
unsafe impl Sync for FocusedElementSnapshot {}

#[cfg(target_os = "macos")]
pub fn capture() -> Option<TargetSnapshot> {
    let (pid, app_name) = capture_app_identity()?;
    let focused_element = capture_focused_element(pid);
    Some(TargetSnapshot {
        pid,
        app_name,
        focused_element,
    })
}

#[cfg(target_os = "macos")]
pub fn capture_app_identity() -> Option<(i32, Option<String>)> {
    use objc2_app_kit::NSWorkspace;

    let workspace = NSWorkspace::sharedWorkspace();
    let app = workspace.frontmostApplication()?;
    let pid = app.processIdentifier();
    let app_name = app.localizedName().map(|name| name.to_string());
    Some((pid, app_name))
}

#[cfg(target_os = "macos")]
fn capture_focused_element(pid: i32) -> Option<FocusedElementSnapshot> {
    use accessibility::{AXAttribute, AXUIElement};
    use core_foundation::{base::CFType, string::CFString};

    if !handy_keys::check_accessibility() {
        return None;
    }

    let application = AXUIElement::application(pid);
    application.set_messaging_timeout(0.25).ok()?;
    let attribute = AXAttribute::new(&CFString::from_static_string("AXFocusedUIElement"));
    let value: CFType = application.attribute(&attribute).ok()?;
    value.downcast::<AXUIElement>().map(FocusedElementSnapshot)
}

#[cfg(target_os = "windows")]
pub fn capture() -> Option<TargetSnapshot> {
    let hwnd = foreground_window()?;
    let pid = window_pid(hwnd);
    Some(TargetSnapshot {
        pid,
        app_name: process_image_name(pid),
        focused: Some(FocusedWindowSnapshot {
            focus: focused_hwnd(hwnd).unwrap_or(hwnd.0 as isize),
        }),
    })
}

#[cfg(target_os = "windows")]
pub fn capture_app_identity() -> Option<(i32, Option<String>)> {
    let pid = window_pid(foreground_window()?);
    Some((pid, process_image_name(pid)))
}

/// The window the user was working in, skipping Meetly's own windows.
///
/// The overlay can hold the foreground when the shortcut fires, and capturing
/// ourselves would make the whole "hand the answer back to the app you were
/// in" flow a no-op. When that happens, walk down the Z-order to the first
/// window that is not ours.
#[cfg(target_os = "windows")]
fn foreground_window() -> Option<HWND> {
    use windows::Win32::System::Threading::GetCurrentProcessId;
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindow, GW_HWNDNEXT};

    let self_pid = unsafe { GetCurrentProcessId() };
    let mut candidate = unsafe { GetForegroundWindow() };
    let mut hops = 0;
    while !candidate.is_invalid() && hops < 16 {
        if window_pid(candidate) != self_pid as i32 {
            return Some(candidate);
        }
        let next = unsafe { GetWindow(candidate, GW_HWNDNEXT) };
        candidate = next.unwrap_or(HWND(std::ptr::null_mut()));
        hops += 1;
    }
    None
}

#[cfg(target_os = "windows")]
fn window_pid(hwnd: HWND) -> i32 {
    use windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId;

    let mut pid = 0u32;
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    pid as i32
}

/// The control holding the keyboard inside `hwnd`'s thread.
///
/// `GetFocus` only reports windows owned by the calling thread, so the
/// cross-thread query has to go through `GetGUIThreadInfo`.
#[cfg(target_os = "windows")]
fn focused_hwnd(hwnd: HWND) -> Option<isize> {
    use windows::Win32::UI::WindowsAndMessaging::GUITHREADINFO;
    use windows::Win32::UI::WindowsAndMessaging::{GetGUIThreadInfo, GetWindowThreadProcessId};

    let thread = unsafe { GetWindowThreadProcessId(hwnd, None) };
    if thread == 0 {
        return None;
    }
    let mut info = GUITHREADINFO::default();
    info.cbSize = std::mem::size_of::<GUITHREADINFO>() as u32;
    if unsafe { GetGUIThreadInfo(thread, &mut info) }.is_err() {
        return None;
    }
    if info.hwndFocus.is_invalid() {
        return None;
    }
    Some(info.hwndFocus.0 as isize)
}

/// Executable name of `pid`, used only for the "answered in X" label.
#[cfg(target_os = "windows")]
fn process_image_name(pid: i32) -> Option<String> {
    use windows::core::PWSTR;
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };

    if pid <= 0 {
        return None;
    }
    // PROCESS_QUERY_LIMITED_INFORMATION is enough for the image name and does
    // not require the debug privilege that PROCESS_ALL_ACCESS would.
    let handle =
        unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid as u32) }.ok()?;
    let mut buffer = [0u16; 1024];
    let mut length = buffer.len() as u32;
    let read = unsafe {
        QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            PWSTR(buffer.as_mut_ptr()),
            &mut length,
        )
    };
    let _ = unsafe { CloseHandle(handle) };
    if read.is_err() || length == 0 {
        return None;
    }
    let path = String::from_utf16_lossy(&buffer[..(length as usize).min(buffer.len())]);
    std::path::Path::new(&path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn capture() -> Option<TargetSnapshot> {
    None
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn capture_app_identity() -> Option<(i32, Option<String>)> {
    None
}

#[cfg(target_os = "macos")]
pub async fn activate(app: &tauri::AppHandle, target: &TargetSnapshot) -> bool {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let target = target.clone();
    if app
        .run_on_main_thread(move || {
            let _ = tx.send(activate_on_main_thread(&target));
        })
        .is_err()
    {
        return false;
    }
    rx.await.unwrap_or(false)
}

#[cfg(target_os = "macos")]
pub fn restore_focus(target: &TargetSnapshot) -> Result<(), String> {
    use accessibility::AXAttribute;
    use core_foundation::boolean::CFBoolean;

    let focused = target
        .focused_element
        .as_ref()
        .ok_or_else(|| "The original focused input is unavailable.".to_string())?;
    focused
        .0
        .set_messaging_timeout(0.25)
        .map_err(|error| error.to_string())?;
    focused
        .0
        .set_attribute(&AXAttribute::focused(), CFBoolean::true_value())
        .map_err(|error| error.to_string())
}

#[cfg(target_os = "macos")]
fn activate_on_main_thread(target: &TargetSnapshot) -> bool {
    use objc2_app_kit::{NSApplicationActivationOptions, NSRunningApplication};

    let Some(app) = NSRunningApplication::runningApplicationWithProcessIdentifier(target.pid)
    else {
        return false;
    };
    if app.isTerminated() {
        return false;
    }
    if app.isActive() {
        return true;
    }
    app.activateWithOptions(NSApplicationActivationOptions::ActivateAllWindows)
}

#[cfg(target_os = "windows")]
pub async fn activate(_app: &tauri::AppHandle, target: &TargetSnapshot) -> bool {
    let Some(focused) = target.focused else {
        return false;
    };
    let activated = activate_hwnd(focused.hwnd());
    // Foreground activation is the step most likely to fail on a machine we
    // have never run on, so record what we aimed at before reporting.
    let _ = crate::debug_log::append(&format!(
        "[dictation-target] activate pid={} app={} focus={:#x} ok={activated}",
        target.pid,
        target.app_name.as_deref().unwrap_or("unknown"),
        focused.focus
    ));
    activated
}

#[cfg(target_os = "windows")]
fn activate_hwnd(focus: HWND) -> bool {
    use windows::Win32::UI::WindowsAndMessaging::{
        BringWindowToTop, GetAncestor, GetForegroundWindow, IsIconic, IsWindow,
        SetForegroundWindow, ShowWindow, GA_ROOT, SW_RESTORE,
    };

    if focus.is_invalid() || !unsafe { IsWindow(Some(focus)) }.as_bool() {
        return false;
    }
    let root = unsafe { GetAncestor(focus, GA_ROOT) };
    let root = if root.is_invalid() { focus } else { root };

    // A minimised window ignores SetForegroundWindow, so restore it first.
    if unsafe { IsIconic(root) }.as_bool() {
        let _ = unsafe { ShowWindow(root, SW_RESTORE) };
    }
    if unsafe { SetForegroundWindow(root) }.as_bool() {
        return true;
    }
    // The shell rate-limits foreground changes, and a background process is
    // refused outright. Retrying from the top of the Z-order is the usual way
    // through; if it still fails the caller falls back to clipboard-only.
    let _ = unsafe { BringWindowToTop(root) };
    let _ = unsafe { SetForegroundWindow(root) };
    let foreground = unsafe { GetForegroundWindow() };
    foreground == root
}

#[cfg(target_os = "windows")]
pub fn restore_focus(target: &TargetSnapshot) -> Result<(), String> {
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CLSCTX_SERVER, COINIT_MULTITHREADED,
    };
    use windows::Win32::UI::Accessibility::{CUIAutomation, IUIAutomation};
    use windows::Win32::UI::WindowsAndMessaging::IsWindow;

    let focused = target
        .focused
        .ok_or_else(|| "The original focused control is unavailable.".to_string())?;
    let focus = focused.hwnd();
    if focus.is_invalid() || !unsafe { IsWindow(Some(focus)) }.as_bool() {
        return Err("The original focused control no longer exists.".to_string());
    }

    // user32's SetFocus only accepts windows attached to the calling thread's
    // queue, and AttachThreadInput is not exposed by the windows crate. UIA's
    // SetFocus has no such restriction, and ElementFromHandle resolves the
    // same control from the HWND we captured earlier.
    let _ = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    let automation: IUIAutomation =
        unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_SERVER) }
            .map_err(|error| format!("UI automation is unavailable: {error}"))?;
    let element = unsafe { automation.ElementFromHandle(focus) }
        .map_err(|error| format!("The original focused control is unreachable: {error}"))?;
    unsafe { element.SetFocus() }.map_err(|error| format!("Could not restore focus: {error}"))
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub async fn activate(_app: &tauri::AppHandle, _target: &TargetSnapshot) -> bool {
    false
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn restore_focus(_target: &TargetSnapshot) -> Result<(), String> {
    Err("Restoring input focus is only supported on macOS and Windows.".to_string())
}
