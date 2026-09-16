use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};
use tokio::task::JoinHandle;

#[cfg(any(target_os = "macos", target_os = "windows"))]
mod microphone;
mod speaker;
mod transcript_buffer;
mod vad;
pub(crate) mod wav;

use transcript_buffer::{TranscriptBuffer, TranscriptSegment};
use vad::{Segmenter, SegmenterEvent, VadConfig};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioRunState {
    Idle,
    Listening,
    SetupRequired,
    Error,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioStatus {
    pub state: AudioRunState,
    pub platform: String,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub sample_rate: Option<u32>,
    pub level: f32,
    pub setup_required: bool,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioLevelChanged {
    pub source: String,
    pub level: f32,
    pub peak: f32,
    pub rms: f32,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TranscriptError {
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TranscriptPartial {
    text: String,
    source: String,
    start_ms: u64,
    end_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureChannelStatus {
    pub ready: bool,
    pub sample_rate: Option<u32>,
    pub device_name: Option<String>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MeetingCaptureStatus {
    pub remote: bool,
    pub system: CaptureChannelStatus,
    pub microphone: CaptureChannelStatus,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AudioChannelFailure {
    source: String,
    message: String,
}

#[derive(Debug, Default)]
struct AudioRuntime {
    task: Option<JoinHandle<()>>,
    stop_signal: Option<Arc<AtomicBool>>,
    sample_rate: Option<u32>,
    device_name: Option<String>,
    level: f32,
    last_error: Option<String>,
}

#[derive(Debug, Default)]
pub struct AudioState {
    runtime: Arc<Mutex<AudioRuntime>>,
    microphone_runtime: Arc<Mutex<AudioRuntime>>,
    // Kept separate from `AudioRuntime` because it should survive a
    // stop/start cycle: a user pausing and resuming listening should not
    // lose the last few minutes of context.
    transcript: Arc<Mutex<TranscriptBuffer>>,
    // True while a meeting capture session is live. In-flight transcription
    // tasks check this before emitting their result so a stop doesn't cause
    // "paused but still transcribing" surprises.
    capture_active: Arc<AtomicBool>,
    // Reorders finished segments so `transcript_final` reaches the UI in the
    // order the words were spoken rather than the order STT happened to
    // finish. See `TranscriptOrderGate`.
    transcript_order: Arc<Mutex<TranscriptOrderGate>>,
}

#[derive(Clone, Copy)]
enum CaptureChannel {
    System,
    Microphone,
}

impl CaptureChannel {
    fn source(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Microphone => "microphone",
        }
    }

    fn speaker(self) -> &'static str {
        match self {
            Self::System => "interviewer",
            Self::Microphone => "user",
        }
    }
}

/// Returns transcript segments from the last `window_ms` milliseconds.
/// Called directly (not through Tauri IPC) by other Rust modules, e.g. the
/// future assistant service building an Ask prompt.
pub fn recent_transcript(state: &AudioState, window_ms: u64) -> Vec<TranscriptSegmentDto> {
    let Ok(buffer) = state.transcript.lock() else {
        return Vec::new();
    };
    buffer
        .recent(window_ms)
        .into_iter()
        .map(Into::into)
        .collect()
}

pub fn is_listening(state: &AudioState) -> bool {
    state
        .runtime
        .lock()
        .map(|runtime| runtime.task.is_some())
        .unwrap_or(false)
}

/// Public DTO alias so callers outside this module don't need to reach into
/// the private `transcript_buffer` submodule.
pub type TranscriptSegmentDto = TranscriptSegment;

#[tauri::command]
pub fn get_audio_status(state: tauri::State<AudioState>) -> AudioStatus {
    build_audio_status(&state, None)
}

#[tauri::command]
pub fn get_recent_transcript(
    state: tauri::State<AudioState>,
    window_ms: u64,
) -> Vec<TranscriptSegmentDto> {
    recent_transcript(&state, window_ms)
}

#[tauri::command]
pub async fn start_listening(
    app: AppHandle,
    state: tauri::State<'_, AudioState>,
) -> Result<AudioStatus, String> {
    tracing::info!("start_listening: invoked");

    {
        let runtime = state
            .runtime
            .lock()
            .map_err(|error| format!("Failed to read audio state: {error}"))?;

        if runtime.task.is_some() {
            tracing::info!("start_listening: already listening, no-op");
            return Ok(build_audio_status(&state, Some(AudioRunState::Listening)));
        }
    }

    reset_transcript_order(&state);
    state.capture_active.store(true, Ordering::Release);

    match start_system_capture_task(
        app,
        state.runtime.clone(),
        state.transcript.clone(),
        Instant::now(),
    )
    .await
    {
        Ok(sample_rate) => {
            tracing::info!(sample_rate, "start_listening: capture task started");
            if let Ok(mut runtime) = state.runtime.lock() {
                runtime.sample_rate = Some(sample_rate);
                runtime.last_error = None;
            }

            Ok(build_audio_status(&state, Some(AudioRunState::Listening)))
        }
        Err(error) => {
            let message = error.to_string();
            tracing::error!(%message, "start_listening: failed to start capture task");

            if let Ok(mut runtime) = state.runtime.lock() {
                runtime.task = None;
                runtime.sample_rate = None;
                runtime.level = 0.0;
                runtime.last_error = Some(message.clone());
            }

            Err(message)
        }
    }
}

#[tauri::command]
pub async fn start_meeting_capture(
    app: AppHandle,
    state: tauri::State<'_, AudioState>,
    remote: bool,
    record_user_mic: bool,
) -> Result<MeetingCaptureStatus, String> {
    tracing::info!(remote, record_user_mic, "start_meeting_capture: invoked");
    let started_at = Instant::now();

    reset_transcript_order(&state);
    state.capture_active.store(true, Ordering::Release);

    let system = if remote {
        start_system_channel(
            app.clone(),
            state.runtime.clone(),
            state.transcript.clone(),
            started_at,
        )
        .await
    } else {
        stop_runtime(state.runtime.clone()).await?;
        CaptureChannelStatus {
            ready: false,
            sample_rate: None,
            device_name: None,
            message: None,
        }
    };

    // A remote interview assistant should be able to listen to only the
    // interviewer (system audio) without the candidate's own microphone
    // polluting the transcript. In-person mode always uses the microphone.
    let microphone = if record_user_mic || !remote {
        start_microphone_channel(
            app.clone(),
            state.microphone_runtime.clone(),
            state.transcript.clone(),
            started_at,
        )
        .await
    } else {
        stop_runtime(state.microphone_runtime.clone()).await?;
        CaptureChannelStatus {
            ready: false,
            sample_rate: None,
            device_name: None,
            message: None,
        }
    };

    for (channel, status) in [
        (CaptureChannel::System, &system),
        (CaptureChannel::Microphone, &microphone),
    ] {
        if let Some(message) = status.message.as_ref().filter(|_| !status.ready) {
            let _ = app.emit(
                "audio_channel_failed",
                AudioChannelFailure {
                    source: channel.source().to_string(),
                    message: message.clone(),
                },
            );
        }
    }

    Ok(MeetingCaptureStatus {
        remote,
        system,
        microphone,
    })
}

#[tauri::command]
pub async fn stop_listening(state: tauri::State<'_, AudioState>) -> Result<AudioStatus, String> {
    state.capture_active.store(false, Ordering::Release);
    stop_runtime(state.runtime.clone()).await?;

    Ok(build_audio_status(&state, Some(AudioRunState::Idle)))
}

#[tauri::command]
pub async fn stop_meeting_capture(
    state: tauri::State<'_, AudioState>,
) -> Result<MeetingCaptureStatus, String> {
    state.capture_active.store(false, Ordering::Release);
    stop_runtime(state.runtime.clone()).await?;
    stop_runtime(state.microphone_runtime.clone()).await?;
    Ok(MeetingCaptureStatus {
        remote: false,
        system: idle_channel_status(),
        microphone: idle_channel_status(),
    })
}

async fn stop_runtime(runtime: Arc<Mutex<AudioRuntime>>) -> Result<(), String> {
    let task = {
        let mut runtime = runtime
            .lock()
            .map_err(|error| format!("Failed to update audio state: {error}"))?;
        runtime.level = 0.0;
        runtime.sample_rate = None;
        runtime.device_name = None;
        if let Some(stop_signal) = runtime.stop_signal.take() {
            stop_signal.store(true, Ordering::Release);
        }
        runtime.task.take()
    };

    if let Some(task) = task {
        task.abort();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    Ok(())
}

fn idle_channel_status() -> CaptureChannelStatus {
    CaptureChannelStatus {
        ready: false,
        sample_rate: None,
        device_name: None,
        message: None,
    }
}

fn current_channel_status(runtime: &Arc<Mutex<AudioRuntime>>) -> Option<CaptureChannelStatus> {
    let runtime = runtime.lock().ok()?;
    runtime.task.as_ref()?;
    Some(CaptureChannelStatus {
        ready: true,
        sample_rate: runtime.sample_rate,
        device_name: runtime.device_name.clone(),
        message: None,
    })
}

async fn start_system_channel(
    app: AppHandle,
    runtime: Arc<Mutex<AudioRuntime>>,
    transcript: Arc<Mutex<TranscriptBuffer>>,
    started_at: Instant,
) -> CaptureChannelStatus {
    if let Some(status) = current_channel_status(&runtime) {
        return status;
    }

    match start_system_capture_task(app, runtime, transcript, started_at).await {
        Ok(sample_rate) => CaptureChannelStatus {
            ready: true,
            sample_rate: Some(sample_rate),
            device_name: speaker::probe_devices().output_device,
            message: None,
        },
        Err(error) => CaptureChannelStatus {
            ready: false,
            sample_rate: None,
            device_name: None,
            message: Some(error.to_string()),
        },
    }
}

async fn start_microphone_channel(
    app: AppHandle,
    runtime: Arc<Mutex<AudioRuntime>>,
    transcript: Arc<Mutex<TranscriptBuffer>>,
    started_at: Instant,
) -> CaptureChannelStatus {
    if let Some(status) = current_channel_status(&runtime) {
        return status;
    }

    #[cfg(any(target_os = "macos", target_os = "windows"))]
    {
        match start_microphone_capture_task(app, runtime, transcript, started_at).await {
            Ok((sample_rate, device_name)) => CaptureChannelStatus {
                ready: true,
                sample_rate: Some(sample_rate),
                device_name: Some(device_name),
                message: None,
            },
            Err(error) => CaptureChannelStatus {
                ready: false,
                sample_rate: None,
                device_name: None,
                message: Some(error.to_string()),
            },
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = (app, runtime, transcript, started_at);
        CaptureChannelStatus {
            ready: false,
            sample_rate: None,
            device_name: None,
            message: Some(
                "Native microphone meeting capture is only implemented on macOS and Windows."
                    .to_string(),
            ),
        }
    }
}

async fn start_system_capture_task(
    app: AppHandle,
    runtime: Arc<Mutex<AudioRuntime>>,
    transcript: Arc<Mutex<TranscriptBuffer>>,
    started_at: Instant,
) -> Result<u32> {
    tracing::debug!("start_capture_task: creating SpeakerInput (CoreAudio process tap)");
    let input = speaker::SpeakerInput::new(None).inspect_err(|error| {
        tracing::error!(%error, "start_capture_task: SpeakerInput::new failed");
    })?;

    tracing::debug!("start_capture_task: SpeakerInput created, requesting stream");
    let stream = input.stream().inspect_err(|error| {
        tracing::error!(%error, "start_capture_task: input.stream() failed");
    })?;
    let sample_rate = stream.sample_rate();
    let live_rate = stream.live_rate();
    let dropped_samples = stream.dropped_samples();
    tracing::debug!(sample_rate, "start_capture_task: stream obtained");

    if !(8_000..=96_000).contains(&sample_rate) {
        return Err(anyhow!("Invalid sample rate: {sample_rate}"));
    }

    let runtime_for_task = runtime.clone();
    let app_for_task = app.clone();
    let task = tokio::spawn(async move {
        run_level_capture(
            app_for_task,
            runtime_for_task,
            transcript,
            stream,
            sample_rate,
            CaptureChannel::System,
            started_at,
            live_rate,
            dropped_samples,
        )
        .await;
    });

    let mut guard = runtime
        .lock()
        .map_err(|error| anyhow!("Failed to store audio task: {error}"))?;
    guard.task = Some(task);
    guard.sample_rate = Some(sample_rate);
    guard.level = 0.0;
    guard.last_error = None;

    Ok(sample_rate)
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
async fn start_microphone_capture_task(
    app: AppHandle,
    runtime: Arc<Mutex<AudioRuntime>>,
    transcript: Arc<Mutex<TranscriptBuffer>>,
    started_at: Instant,
) -> Result<(u32, String)> {
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let stop_signal = Arc::new(AtomicBool::new(false));
    let stop_for_task = stop_signal.clone();
    let runtime_for_task = runtime.clone();
    let task = tokio::task::spawn_blocking(move || {
        let mut stream = match microphone::MicrophoneStream::new() {
            Ok(stream) => stream,
            Err(error) => {
                let _ = ready_tx.send(Err(error.to_string()));
                return;
            }
        };
        let sample_rate = stream.sample_rate();
        let device_name = stream.device_name().to_string();
        let _ = ready_tx.send(Ok((sample_rate, device_name)));
        run_microphone_capture_blocking(
            app,
            runtime_for_task,
            transcript,
            &mut stream,
            sample_rate,
            started_at,
            stop_for_task,
        );
    });

    let (sample_rate, device_name) = ready_rx
        .await
        .map_err(|_| anyhow!("Native microphone task stopped during startup"))?
        .map_err(anyhow::Error::msg)?;

    let mut guard = runtime
        .lock()
        .map_err(|error| anyhow!("Failed to store microphone task: {error}"))?;
    guard.task = Some(task);
    guard.stop_signal = Some(stop_signal);
    guard.sample_rate = Some(sample_rate);
    guard.device_name = Some(device_name.clone());
    guard.level = 0.0;
    guard.last_error = None;
    Ok((sample_rate, device_name))
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
fn run_microphone_capture_blocking(
    app: AppHandle,
    runtime: Arc<Mutex<AudioRuntime>>,
    transcript: Arc<Mutex<TranscriptBuffer>>,
    stream: &mut microphone::MicrophoneStream,
    sample_rate: u32,
    started_at: Instant,
    stop_signal: Arc<AtomicBool>,
) {
    let channel = CaptureChannel::Microphone;
    let _ = app.emit("audio_channel_started", channel.source());
    let mut segmenter = Segmenter::new(VadConfig::from_millis(sample_rate));
    let mut samples = vec![0.0f32; 2_048];
    let mut level_chunk = Vec::with_capacity(2_048);

    while !stop_signal.load(Ordering::Acquire) {
        let count = stream.read_samples(&mut samples);
        if count == 0 {
            std::thread::sleep(std::time::Duration::from_millis(5));
            continue;
        }

        for &sample in &samples[..count] {
            level_chunk.push(sample);
            match segmenter.push(sample) {
                SegmenterEvent::None => {}
                SegmenterEvent::SpeechStarted => {
                    let _ = app.emit("speech_segment_started", channel.source());
                }
                SegmenterEvent::Discarded => {}
                SegmenterEvent::SegmentReady(segment) => {
                    let end_ms = started_at.elapsed().as_millis() as u64;
                    let start_ms = end_ms
                        .saturating_sub(segment.samples.len() as u64 * 1000 / sample_rate as u64);
                    spawn_transcription(
                        app.clone(),
                        transcript.clone(),
                        sample_rate,
                        segment.samples,
                        start_ms,
                        end_ms,
                        channel,
                    );
                }
            }
        }

        if level_chunk.len() >= 1_024 {
            let (rms, peak) = calculate_audio_metrics(&level_chunk);
            let level = (rms * 8.0).clamp(0.0, 1.0);
            if let Ok(mut guard) = runtime.lock() {
                guard.level = level;
            }
            let _ = app.emit(
                "audio_level_changed",
                AudioLevelChanged {
                    source: channel.source().to_string(),
                    level,
                    peak,
                    rms,
                    sample_rate,
                },
            );
            level_chunk.clear();
        }
    }

    if let Ok(mut guard) = runtime.lock() {
        guard.task = None;
        guard.stop_signal = None;
        guard.sample_rate = None;
        guard.device_name = None;
        guard.level = 0.0;
    }
    let _ = app.emit("audio_channel_stopped", channel.source());
}

async fn run_level_capture<S>(
    app: AppHandle,
    runtime: Arc<Mutex<AudioRuntime>>,
    transcript: Arc<Mutex<TranscriptBuffer>>,
    mut stream: S,
    sample_rate: u32,
    channel: CaptureChannel,
    started_at: Instant,
    live_rate: Arc<AtomicU32>,
    dropped_samples: Arc<AtomicUsize>,
) where
    S: futures_util::Stream<Item = f32> + Unpin,
{
    let _ = app.emit("audio_channel_started", channel.source());
    if matches!(channel, CaptureChannel::System) {
        let _ = app.emit("audio_capture_started", sample_rate);
    }

    let hop_size = 1024usize;
    let mut chunk = Vec::with_capacity(hop_size);
    let mut segmenter = Segmenter::new(VadConfig::from_millis(sample_rate));
    // Rate the *current* segment must be encoded with. Conferencing apps
    // reconfigure the output device when a call starts, and the tap follows
    // them, so this is re-read continuously rather than captured once.
    let mut active_rate = sample_rate;
    let mut since_rate_probe = 0usize;
    // Wall-clock start of the phrase being accumulated. Comparing it against
    // `samples / rate` is the only rate check that does not trust the value
    // under test: if the capture rate is wrong, the WAV header carries the
    // same wrong number and every downstream check agrees with it.
    let mut speech_started_at: Option<std::time::Instant> = None;

    // Throughput probe: a consumer that cannot keep up with the capture rate
    // lets the ring buffer overrun, which deletes samples from the middle of
    // the stream. Comparing consumed samples against wall time catches that
    // directly, and timing `emit` shows whether UI traffic is the cause.
    let mut probe_samples = 0usize;
    let mut probe_started = std::time::Instant::now();
    let mut emit_nanos = 0u128;
    let mut emit_decimator = 0u32;
    // UI level meters look identical at ~12 Hz and this fires 47x/s at 48 kHz.
    const EMIT_EVERY_N_HOPS: u32 = 4;

    while let Some(sample) = stream.next().await {
        probe_samples += 1;
        chunk.push(sample);
        since_rate_probe += 1;
        if since_rate_probe >= hop_size {
            since_rate_probe = 0;
            let live = live_rate.load(Ordering::Acquire);
            if live > 0 && live != active_rate {
                let _ = crate::debug_log::append(&format!(
                    "[audio] re-segmenting at new capture rate {active_rate} -> {live}"
                ));
                active_rate = live;
                segmenter = Segmenter::new(VadConfig::from_millis(active_rate));
            }
        }

        match segmenter.push(sample) {
            SegmenterEvent::None => {}
            SegmenterEvent::SpeechStarted => {
                speech_started_at = Some(std::time::Instant::now());
                let _ = app.emit("speech_segment_started", ());
            }
            SegmenterEvent::Discarded => {}
            SegmenterEvent::SegmentReady(segment) => {
                let nominal_seconds = segment.samples.len() as f64 / active_rate as f64;
                if let Some(began) = speech_started_at.take() {
                    let wall_seconds = began.elapsed().as_secs_f64();
                    // Slightly above 1.0 is healthy: the segment's wall time
                    // includes the trailing silence gap that ends it, which
                    // the samples do not. Materially BELOW 1.0 means the
                    // header rate is too low for the samples actually
                    // delivered, so the audio is stretched and ASR breaks.
                    let ratio = if nominal_seconds > 0.0 {
                        wall_seconds / nominal_seconds
                    } else {
                        0.0
                    };
                    let _ = crate::debug_log::append(&format!(
                        "[audio] rate check rate={active_rate} nominal={nominal_seconds:.2}s wall={wall_seconds:.2}s ratio={ratio:.2}"
                    ));
                }
                let end_ms = started_at.elapsed().as_millis() as u64;
                let start_ms =
                    end_ms.saturating_sub(segment.samples.len() as u64 * 1000 / active_rate as u64);

                spawn_transcription(
                    app.clone(),
                    transcript.clone(),
                    active_rate,
                    segment.samples,
                    start_ms,
                    end_ms,
                    channel,
                );
            }
        }

        if chunk.len() < hop_size {
            continue;
        }

        let (rms, peak) = calculate_audio_metrics(&chunk);
        let level = (rms * 8.0).clamp(0.0, 1.0);

        if let Ok(mut guard) = runtime.lock() {
            guard.level = level;
        }

        emit_decimator += 1;
        if emit_decimator >= EMIT_EVERY_N_HOPS {
            emit_decimator = 0;
            let emit_started = std::time::Instant::now();
            let _ = app.emit(
                "audio_level_changed",
                AudioLevelChanged {
                    source: channel.source().to_string(),
                    level,
                    peak,
                    rms,
                    sample_rate,
                },
            );
            emit_nanos += emit_started.elapsed().as_nanos();
        }

        chunk.clear();

        // ~2 s of audio per report: long enough to average out scheduling
        // noise, short enough to show up while a meeting is still running.
        let probe_window = (active_rate as usize).saturating_mul(2).max(1);
        if probe_samples >= probe_window {
            let elapsed = probe_started.elapsed().as_secs_f64();
            let dropped = dropped_samples.swap(0, Ordering::Relaxed);
            if elapsed > 0.0 {
                let consumed_rate = probe_samples as f64 / elapsed;
                let emit_ms = emit_nanos as f64 / 1_000_000.0;
                let _ = crate::debug_log::append(&format!(
                    "[audio] throughput consumed={:.0}/s expected={} deficit={:.0}% dropped={} emit_total={:.1}ms emit_share={:.1}%",
                    consumed_rate,
                    active_rate,
                    (1.0 - (consumed_rate / active_rate as f64)) * 100.0,
                    dropped,
                    emit_ms,
                    (emit_ms / 1000.0) / elapsed * 100.0
                ));

                // Conferencing apps switch the output device to telephony
                // rates (48 kHz -> 16 kHz) and the tap follows, but `asbd()`
                // keeps reporting the old nominal rate. Encoding with that
                // stale rate time-stretches the audio and the ASR answers with
                // invention. Trust measured arrival rate instead — but only
                // when nothing was dropped, since an overrun makes the
                // measurement read low and would calibrate in the wrong
                // direction.
                if dropped == 0 {
                    let drift = (consumed_rate - active_rate as f64).abs() / active_rate as f64;
                    let measured = consumed_rate.round() as u32;
                    if drift > 0.15 && (8_000..=96_000).contains(&measured) {
                        let _ = crate::debug_log::append(&format!(
                            "[audio] rate recalibrated {active_rate} -> {measured} (measured; asbd still reports {sample_rate})"
                        ));
                        active_rate = measured;
                        live_rate.store(measured, Ordering::Release);
                        segmenter = Segmenter::new(VadConfig::from_millis(measured));
                    }
                }
            }
            probe_samples = 0;
            emit_nanos = 0;
            probe_started = std::time::Instant::now();
        }
    }

    if let Ok(mut guard) = runtime.lock() {
        guard.task = None;
        guard.sample_rate = None;
        guard.level = 0.0;
    }

    let _ = app.emit("audio_channel_stopped", channel.source());
    if matches!(channel, CaptureChannel::System) {
        let _ = app.emit("audio_capture_stopped", ());
    }
}

/// How long a finished segment may wait for an earlier one to finish before it
/// is released anyway.
///
/// Without this, a single hung STT request would hold back every later segment
/// for as long as the provider's own timeout allows — the transcript would look
/// dead rather than merely late.
const ORDER_HOLD_LIMIT: Duration = Duration::from_secs(3);

/// Releases finished transcript segments in the order they were spoken.
///
/// Every segment is transcribed by its own task (see `spawn_transcription`), so
/// a short question routinely finishes while a longer, earlier one is still
/// running. Emitting in completion order meant the coach answered question B
/// before question A — fine for a log, wrong for an interview, where the answer
/// to a question only makes sense after the question.
///
/// Segments are keyed by `start_ms`, which is monotonic within a capture
/// channel. The gate is per channel because the two channels timestamp their
/// segments from their own start instant and are not comparable to each other.
#[derive(Debug, Default)]
struct TranscriptOrderGate {
    channels: HashMap<&'static str, ChannelOrder>,
}

#[derive(Debug, Default)]
struct ChannelOrder {
    /// `start_ms` of segments that have been spawned but not yet delivered.
    in_flight: Vec<u64>,
    /// Transcribed but not yet released. A `BTreeMap` so they come out sorted.
    held: BTreeMap<u64, HeldSegment>,
}

#[derive(Debug)]
struct HeldSegment {
    segment: TranscriptSegment,
    finished_at: Instant,
}

impl TranscriptOrderGate {
    fn begin(&mut self, channel: &'static str, start_ms: u64) {
        self.channels.entry(channel).or_default().in_flight.push(start_ms);
    }

    /// The segment transcribed successfully. Releases it plus anything after it
    /// that is no longer waiting on an earlier result.
    fn finish(
        &mut self,
        channel: &'static str,
        start_ms: u64,
        segment: TranscriptSegment,
    ) -> Vec<TranscriptSegment> {
        let entry = self.channels.entry(channel).or_default();
        remove_in_flight(&mut entry.in_flight, start_ms);
        entry.held.insert(
            start_ms,
            HeldSegment {
                segment,
                finished_at: Instant::now(),
            },
        );
        release_ready(entry)
    }

    /// The segment produced nothing usable — an error, empty text, or the
    /// capture stopped mid-flight. It still has to leave `in_flight`, or every
    /// later segment would wait for a result that is never arriving.
    fn abandon(&mut self, channel: &'static str, start_ms: u64) -> Vec<TranscriptSegment> {
        let entry = self.channels.entry(channel).or_default();
        remove_in_flight(&mut entry.in_flight, start_ms);
        release_ready(entry)
    }

    /// Releases anything that has waited longer than `ORDER_HOLD_LIMIT`, even
    /// if an earlier segment is still in flight.
    fn release_expired(&mut self, channel: &'static str) -> Vec<TranscriptSegment> {
        let Some(entry) = self.channels.get_mut(channel) else {
            return Vec::new();
        };

        let expired: Vec<u64> = entry
            .held
            .iter()
            .filter(|(_, held)| held.finished_at.elapsed() >= ORDER_HOLD_LIMIT)
            .map(|(&start_ms, _)| start_ms)
            .collect();

        let mut released = Vec::new();
        for start_ms in expired {
            if let Some(held) = entry.held.remove(&start_ms) {
                released.push(held.segment);
            }
        }
        released
    }

    fn has_held(&self, channel: &'static str) -> bool {
        self.channels
            .get(channel)
            .map(|entry| !entry.held.is_empty())
            .unwrap_or(false)
    }

    /// Drops everything buffered. Called when a capture session starts so a
    /// half-finished previous session can't hold the new one back.
    fn reset(&mut self) {
        self.channels.clear();
    }
}

/// Emits every held segment that started before the earliest segment still in
/// flight. With nothing in flight the bound is `u64::MAX`, so everything goes.
fn release_ready(entry: &mut ChannelOrder) -> Vec<TranscriptSegment> {
    let bound = entry.in_flight.iter().copied().min().unwrap_or(u64::MAX);

    let mut released = Vec::new();
    while let Some(start_ms) = entry.held.keys().next().copied() {
        if start_ms > bound {
            break;
        }
        if let Some(held) = entry.held.remove(&start_ms) {
            released.push(held.segment);
        }
    }
    released
}

fn remove_in_flight(in_flight: &mut Vec<u64>, start_ms: u64) {
    if let Some(index) = in_flight.iter().position(|&value| value == start_ms) {
        in_flight.remove(index);
    }
}

fn reset_transcript_order(state: &AudioState) {
    if let Ok(mut gate) = state.transcript_order.lock() {
        gate.reset();
    }
}

/// Transcribes one completed speech segment as an independent task so a
/// slow STT response never blocks the capture loop from processing the
/// next chunk. The task checks `AudioState::capture_active` before emitting:
/// if the session was stopped while the request was in flight, the result is
/// silently dropped instead of surfacing after the user paused.
///
/// The result is handed to `TranscriptOrderGate` rather than emitted directly,
/// so the UI still sees segments in spoken order even though they finish out of
/// order.
fn spawn_transcription(
    app: AppHandle,
    transcript: Arc<Mutex<TranscriptBuffer>>,
    sample_rate: u32,
    samples: Vec<f32>,
    start_ms: u64,
    end_ms: u64,
    channel: CaptureChannel,
) {
    let source = channel.source();
    let (capture_active, order) = {
        let state = app.state::<AudioState>();
        (state.capture_active.clone(), state.transcript_order.clone())
    };

    if let Ok(mut gate) = order.lock() {
        gate.begin(source, start_ms);
    }
    // Let the UI show "recognizing…" immediately instead of waiting for the
    // (slow) batch response. Pairs with `transcript_final` / `transcript_error`.
    let _ = app.emit("transcript_started", source);

    tokio::spawn(async move {
        let outcome = transcribe_segment(
            &app,
            sample_rate,
            &samples,
            start_ms,
            end_ms,
            channel,
            &capture_active,
        )
        .await;

        let released = {
            let Ok(mut gate) = order.lock() else {
                return;
            };
            match outcome {
                Some(segment) => gate.finish(source, start_ms, segment),
                None => gate.abandon(source, start_ms),
            }
        };

        emit_transcript_segments(&app, &transcript, released);
        schedule_hold_expiry(&app, &transcript, &order, source);
    });
}

/// Runs STT for one segment and returns the finished segment, or `None` when
/// there is nothing to show (error, empty transcript, capture already stopped).
///
/// Split out of `spawn_transcription` so every exit path funnels into one
/// `finish`/`abandon` call — a stray early `return` would leave the segment in
/// the gate's in-flight set and stall every later one behind it.
async fn transcribe_segment(
    app: &AppHandle,
    sample_rate: u32,
    samples: &[f32],
    start_ms: u64,
    end_ms: u64,
    channel: CaptureChannel,
    capture_active: &Arc<AtomicBool>,
) -> Option<TranscriptSegment> {
    let wav_bytes = match wav::encode_wav(sample_rate, samples) {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = crate::debug_log::append(&format!(
                "[audio] segment encode failed rate={sample_rate} samples={} error={error}",
                samples.len()
            ));
            let _ = app.emit(
                "transcript_error",
                TranscriptError {
                    message: format!("Failed to encode speech segment: {error}"),
                },
            );
            return None;
        }
    };

    crate::debug_log::dump_stt_audio("raw", sample_rate, &wav_bytes);
    let _ = crate::debug_log::append(&format!(
        "[audio] segment ready rate={sample_rate} samples={} bytes={}",
        samples.len(),
        wav_bytes.len()
    ));

    let provider = match crate::providers::stt::build_from_saved_config(app) {
        Ok(provider) => provider,
        Err(error) => {
            let _ = app.emit(
                "transcript_error",
                TranscriptError {
                    message: error.to_string(),
                },
            );
            return None;
        }
    };

    let source = channel.source().to_string();
    let app_for_partial = app.clone();
    let capture_for_partial = capture_active.clone();
    let on_delta = Box::new(move |text: String| {
        // Don't surface partial text after the session was stopped.
        if !capture_for_partial.load(Ordering::Acquire) {
            return;
        }
        let _ = app_for_partial.emit(
            "transcript_partial",
            TranscriptPartial {
                text,
                source: source.clone(),
                start_ms,
                end_ms,
            },
        );
    });

    let text = match provider
        .transcribe_streaming(
            crate::providers::stt::BatchAsrRequest::new(wav_bytes, "segment.wav", "audio/wav"),
            on_delta,
        )
        .await
    {
        Ok(text) => text,
        Err(error) => {
            let _ = app.emit(
                "transcript_error",
                TranscriptError {
                    message: error.to_string(),
                },
            );
            return None;
        }
    };

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Session ended while this request was in flight: drop it so a paused
    // meeting doesn't keep showing newly transcribed text.
    if !capture_active.load(Ordering::Acquire) {
        return None;
    }

    Some(TranscriptSegment {
        id: uuid_like_id(),
        source: channel.source().to_string(),
        speaker: channel.speaker().to_string(),
        text: trimmed.to_string(),
        start_ms,
        end_ms,
    })
}

fn emit_transcript_segments(
    app: &AppHandle,
    transcript: &Arc<Mutex<TranscriptBuffer>>,
    segments: Vec<TranscriptSegment>,
) {
    for segment in segments {
        if let Ok(mut buffer) = transcript.lock() {
            buffer.push(segment.clone());
        }

        let _ = crate::debug_log::append(&format!(
            "[audio] transcript released start_ms={} id={} text={}",
            segment.start_ms,
            segment.id,
            segment.text.chars().take(40).collect::<String>()
        ));
        let _ = app.emit("transcript_final", segment);
    }
}

/// Fires a one-shot release for anything still held once the hold limit passes.
///
/// Without it, a segment blocked behind a stalled STT request would only be
/// released when some *other* segment happened to finish.
fn schedule_hold_expiry(
    app: &AppHandle,
    transcript: &Arc<Mutex<TranscriptBuffer>>,
    order: &Arc<Mutex<TranscriptOrderGate>>,
    channel: &'static str,
) {
    let has_held = match order.lock() {
        Ok(gate) => gate.has_held(channel),
        Err(_) => false,
    };
    if !has_held {
        return;
    }

    let app = app.clone();
    let transcript = transcript.clone();
    let order = order.clone();
    tokio::spawn(async move {
        tokio::time::sleep(ORDER_HOLD_LIMIT).await;
        let released = match order.lock() {
            Ok(mut gate) => gate.release_expired(channel),
            Err(_) => Vec::new(),
        };
        if !released.is_empty() {
            let _ = crate::debug_log::append(&format!(
                "[audio] transcript order hold expired released={} channel={channel}",
                released.len()
            ));
        }
        emit_transcript_segments(&app, &transcript, released);
    });
}

/// Small dependency-free unique id, sufficient for a per-process transcript
/// segment id (not used for anything security-sensitive or persisted
/// across restarts).
fn uuid_like_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("seg-{nanos:x}")
}

fn build_audio_status(state: &AudioState, override_state: Option<AudioRunState>) -> AudioStatus {
    let devices = speaker::probe_devices();
    let runtime = state.runtime.lock();
    let (is_listening, sample_rate, level, last_error) = match runtime {
        Ok(guard) => (
            guard.task.is_some(),
            guard.sample_rate,
            guard.level,
            guard.last_error.clone(),
        ),
        Err(error) => (
            false,
            None,
            0.0,
            Some(format!("Failed to read audio state: {error}")),
        ),
    };

    let mut setup_required = devices.output_device.is_none();
    let mut message = devices.error.or(last_error);

    if devices.output_device.is_none() && message.is_none() {
        message = Some("No default output audio device found.".to_string());
    }

    let mut state = if setup_required {
        AudioRunState::SetupRequired
    } else if message.is_some() && !is_listening {
        AudioRunState::Error
    } else if is_listening {
        AudioRunState::Listening
    } else {
        AudioRunState::Idle
    };

    if let Some(next_state) = override_state {
        state = next_state;
        if matches!(state, AudioRunState::Listening) {
            setup_required = false;
        }
    }

    AudioStatus {
        state,
        platform: std::env::consts::OS.to_string(),
        input_device: devices.input_device,
        output_device: devices.output_device,
        sample_rate,
        level,
        setup_required,
        message,
    }
}

fn calculate_audio_metrics(chunk: &[f32]) -> (f32, f32) {
    if chunk.is_empty() {
        return (0.0, 0.0);
    }

    let mut sumsq = 0.0f32;
    let mut peak = 0.0f32;

    for &sample in chunk {
        let abs = sample.abs();
        peak = peak.max(abs);
        sumsq += sample * sample;
    }

    ((sumsq / chunk.len() as f32).sqrt(), peak)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(id: &str, start_ms: u64) -> TranscriptSegment {
        TranscriptSegment {
            id: id.to_string(),
            source: "system".to_string(),
            speaker: "interviewer".to_string(),
            text: format!("segment {id}"),
            start_ms,
            end_ms: start_ms + 1_000,
        }
    }

    fn ids(segments: &[TranscriptSegment]) -> Vec<&str> {
        segments.iter().map(|s| s.id.as_str()).collect()
    }

    #[test]
    fn releases_a_second_question_only_after_the_first_one_finishes() {
        // The bug: question A is long, question B is short, so B's STT returns
        // first. Emitting on completion made the coach answer B before A.
        let mut gate = TranscriptOrderGate::default();
        gate.begin("system", 0);
        gate.begin("system", 5_000);

        assert!(ids(&gate.finish("system", 5_000, segment("b", 5_000))).is_empty());
        assert_eq!(ids(&gate.finish("system", 0, segment("a", 0))), vec!["a", "b"]);
    }

    #[test]
    fn releases_immediately_when_nothing_earlier_is_in_flight() {
        let mut gate = TranscriptOrderGate::default();
        gate.begin("system", 1_000);

        assert_eq!(ids(&gate.finish("system", 1_000, segment("a", 1_000))), vec!["a"]);
    }

    #[test]
    fn an_abandoned_segment_does_not_stall_the_ones_behind_it() {
        // Empty text, an STT error, and a stopped capture all abandon. Any of
        // them leaving the in-flight set would hold every later segment back
        // until the hold limit expired.
        let mut gate = TranscriptOrderGate::default();
        gate.begin("system", 0);
        gate.begin("system", 5_000);

        gate.abandon("system", 0);
        assert_eq!(ids(&gate.finish("system", 5_000, segment("b", 5_000))), vec!["b"]);
    }

    #[test]
    fn orders_segments_that_finish_in_reverse() {
        let mut gate = TranscriptOrderGate::default();
        for start_ms in [0_u64, 3_000, 6_000] {
            gate.begin("system", start_ms);
        }

        assert!(ids(&gate.finish("system", 6_000, segment("c", 6_000))).is_empty());
        assert!(ids(&gate.finish("system", 3_000, segment("b", 3_000))).is_empty());
        assert_eq!(ids(&gate.finish("system", 0, segment("a", 0))), vec!["a", "b", "c"]);
    }

    #[test]
    fn channels_are_ordered_independently() {
        // Microphone and system audio timestamp from their own start instant, so
        // their start_ms values say nothing about which was spoken first. The
        // microphone segment must not sit behind the system one.
        let mut gate = TranscriptOrderGate::default();
        gate.begin("system", 10_000);
        gate.begin("microphone", 20);

        assert_eq!(
            ids(&gate.finish("microphone", 20, segment("mic", 20))),
            vec!["mic"]
        );
        assert_eq!(
            ids(&gate.finish("system", 10_000, segment("sys", 10_000))),
            vec!["sys"]
        );
    }

    #[test]
    fn a_stalled_segment_is_released_once_the_hold_limit_passes() {
        let mut gate = TranscriptOrderGate::default();
        gate.begin("system", 0);
        gate.begin("system", 5_000);
        gate.finish("system", 5_000, segment("b", 5_000));

        // Wind the clock forward without sleeping through the real limit.
        if let Some(entry) = gate.channels.get_mut("system") {
            if let Some(held) = entry.held.get_mut(&5_000) {
                held.finished_at = Instant::now() - ORDER_HOLD_LIMIT;
            }
        }

        assert_eq!(ids(&gate.release_expired("system")), vec!["b"]);
        assert_eq!(ids(&gate.finish("system", 0, segment("a", 0))), vec!["a"]);
    }

    #[test]
    fn reset_drops_everything_buffered() {
        let mut gate = TranscriptOrderGate::default();
        gate.begin("system", 0);
        gate.begin("system", 5_000);
        gate.finish("system", 5_000, segment("b", 5_000));

        gate.reset();

        assert!(!gate.has_held("system"));
        assert_eq!(ids(&gate.finish("system", 0, segment("a", 0))), vec!["a"]);
    }
}
