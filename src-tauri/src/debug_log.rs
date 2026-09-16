use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn append(line: &str) -> Result<(), String> {
    let path = log_path()?;
    if let Some(parent) = path.parent() {
        create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();

    writeln!(file, "{timestamp} {line}").map_err(|error| error.to_string())
}

#[tauri::command]
pub fn append_debug_log(message: String) -> Result<(), String> {
    append(&message)
}

/// Writes one STT payload to disk so a bad transcript can be replayed and
/// measured offline instead of guessed at.
///
/// Off by default because it stores the user's meeting audio; set
/// `MEETLY_DUMP_STT=1` to turn it on for a debugging session.
///
/// `label` distinguishes the two probe points: `raw` is the segment as encoded
/// from the capture rate, `norm` is what actually reaches the ASR model.
pub fn dump_stt_audio(label: &str, sample_rate: u32, wav: &[u8]) {
    if std::env::var("MEETLY_DUMP_STT").as_deref() != Ok("1") {
        return;
    }

    let Some(home) = dirs::home_dir() else {
        return;
    };
    let dir = home.join(".meetly").join("stt-dump");
    if create_dir_all(&dir).is_err() {
        return;
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    // Data length past the 44-byte header, at 16-bit mono.
    let samples = wav.len().saturating_sub(44) / 2;
    let seconds = samples as f64 / sample_rate as f64;
    let path = dir.join(format!("{stamp}-{label}-{sample_rate}hz.wav"));

    if std::fs::write(&path, wav).is_err() {
        return;
    }
    prune_stt_dumps(&dir, 24);
    let _ = append(&format!(
        "[stt-dump] {label} rate={sample_rate} bytes={} duration={seconds:.2}s path={}",
        wav.len(),
        path.display()
    ));
}

/// Keeps only the newest `keep` dumps so a long meeting cannot fill the disk
/// with audio. Two files are written per segment (`raw` + `norm`).
fn prune_stt_dumps(dir: &std::path::Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(PathBuf, std::time::SystemTime)> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((entry.path(), modified))
        })
        .collect();
    if files.len() <= keep {
        return;
    }
    files.sort_by_key(|(_, modified)| *modified);
    let excess = files.len() - keep;
    for (path, _) in files.into_iter().take(excess) {
        let _ = std::fs::remove_file(path);
    }
}

fn log_path() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or_else(|| "Failed to resolve home directory.".to_string())?;
    Ok(home.join(".meetly").join("debug.log"))
}
