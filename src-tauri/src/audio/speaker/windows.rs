//! System audio capture on Windows, via WASAPI loopback.
//!
//! Two modes share one pipeline:
//!
//! * **Full-device loopback** (default) — opens the default *render* device but
//!   initialises the client with `Direction::Capture`. The `wasapi` crate turns
//!   that combination into `AUDCLNT_STREAMFLAGS_LOOPBACK`, which yields the
//!   entire system output mix.
//! * **Process loopback** — `AudioClient::new_application_loopback_client(pid)`
//!   captures one process tree instead. Select it by passing a device id of the
//!   form `pid:1234`. It is the closest equivalent of the macOS process tap,
//!   but several `AudioClient` queries are documented as non-functional in that
//!   mode (`get_mixformat`, `get_device_period`, `get_buffer_size`), so it must
//!   not reuse the device-period probe that the full-device path relies on.
//!
//! WASAPI only produces packets while something is actually rendering. Silence
//! is therefore *absent* rather than zero-valued, and a stream that stopped
//! during a pause would make the VAD see one long utterance instead of two.
//! The capture loop compensates by emitting silence for wall-clock time that
//! passed without packets, keeping the sample stream continuous for downstream
//! segmentation.

use super::async_ring::RingbufAsyncReader;
use super::{rt_ring, DeviceProbe, BUFFER_SIZE, CHUNK_SIZE};
use anyhow::{anyhow, Result};
use futures_util::task::AtomicWaker;
use futures_util::Stream;
use ringbuf::traits::{Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::pin::Pin;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::task::Poll;
use std::time::{Duration, Instant};
use wasapi::*;

/// How long to block on the WASAPI event before re-checking the stop flag and
/// topping up silence. Short enough that stopping capture does not feel stuck,
/// long enough that an idle machine is not spinning.
const EVENT_WAIT_MS: u32 = 100;

/// Worst-case wait for the capture thread to report the negotiated format.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(10);

/// Fallback format used by process loopback, where `get_mixformat` is not
/// available. Automatic conversion is always on, so the engine resamples.
const FALLBACK_RATE: u32 = 48_000;
const FALLBACK_CHANNELS: u16 = 2;

/// Anything more than a second behind means the consumer stalled; dropping the
/// backlog of silence is better than flooding the ring buffer with it.
const MAX_SILENCE_CATCHUP_SECONDS: f64 = 1.0;

pub fn probe_devices() -> DeviceProbe {
    let outcome = (|| -> Result<DeviceProbe> {
        // COM may already be initialised on this thread; that is not an error,
        // and a real failure surfaces on the first call that needs it.
        if let Err(error) = initialize_mta().ok() {
            let _ = crate::debug_log::append(&format!("[audio] wasapi COM init: {error}"));
        }
        let enumerator =
            DeviceEnumerator::new().map_err(|error| anyhow!("enumerator failed: {error}"))?;

        let input_device = enumerator
            .get_default_device(&Direction::Capture)
            .and_then(|device| device.get_friendlyname())
            .ok();
        let output_device = enumerator
            .get_default_device(&Direction::Render)
            .and_then(|device| device.get_friendlyname())
            .ok();

        Ok(DeviceProbe {
            input_device,
            output_device,
            error: None,
        })
    })();

    match outcome {
        Ok(probe) => probe,
        Err(error) => DeviceProbe {
            error: Some(error.to_string()),
            ..DeviceProbe::default()
        },
    }
}

pub struct SpeakerInput {
    device_id: Option<String>,
    process_id: Option<u32>,
}

pub struct SpeakerStream {
    reader: RingbufAsyncReader<HeapCons<f32>>,
    _stop: Arc<AtomicBool>,
    buffer_rate: u32,
    live_rate: Arc<AtomicU32>,
    dropped_samples: Arc<AtomicUsize>,
}

impl SpeakerStream {
    pub fn sample_rate(&self) -> u32 {
        self.buffer_rate
    }

    pub fn dropped_samples(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.dropped_samples)
    }

    pub fn live_rate(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.live_rate)
    }
}

impl SpeakerInput {
    pub fn new(device_id: Option<String>) -> Result<Self> {
        let process_id = device_id
            .as_deref()
            .and_then(|id| id.strip_prefix("pid:"))
            .and_then(|value| value.trim().parse::<u32>().ok())
            .filter(|pid| *pid > 0);

        Ok(Self {
            device_id,
            process_id,
        })
    }

    pub fn stream(self) -> Result<SpeakerStream> {
        let ring_buffer = HeapRb::<f32>::new(BUFFER_SIZE);
        let (producer, consumer) = ring_buffer.split();

        let waker = Arc::new(AtomicWaker::new());
        let wake_pending = Arc::new(AtomicBool::new(false));
        let dropped_samples = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel();
        // `run_capture` takes ownership of one sender to report the negotiated
        // format; this one stays behind to report a setup failure.
        let error_tx = ready_tx.clone();

        let thread_waker = Arc::clone(&waker);
        let thread_wake_pending = Arc::clone(&wake_pending);
        let thread_dropped = Arc::clone(&dropped_samples);
        let thread_stop = Arc::clone(&stop);

        std::thread::Builder::new()
            .name("meetly-wasapi-capture".to_string())
            .spawn(move || {
                if let Err(error) = run_capture(
                    self.device_id.as_deref(),
                    self.process_id,
                    producer,
                    thread_waker,
                    thread_wake_pending,
                    thread_dropped,
                    thread_stop,
                    ready_tx,
                ) {
                    // Only reached when setup fails; a running loop exits via
                    // the stop flag and has already reported its format.
                    let _ = error_tx.send(Err(error.to_string()));
                }
            })
            .map_err(|error| anyhow!("failed to spawn capture thread: {error}"))?;

        let format = match ready_rx.recv_timeout(STARTUP_TIMEOUT) {
            Ok(Ok(format)) => format,
            Ok(Err(message)) => return Err(anyhow!(message)),
            Err(RecvTimeoutError::Timeout) => {
                stop.store(true, Ordering::Release);
                return Err(anyhow!(
                    "WASAPI capture did not start within {}s.",
                    STARTUP_TIMEOUT.as_secs()
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("WASAPI capture thread exited before starting."))
            }
        };

        let live_rate = Arc::new(AtomicU32::new(format.sample_rate));

        Ok(SpeakerStream {
            reader: RingbufAsyncReader::new(
                consumer,
                waker,
                wake_pending,
                vec![0.0; CHUNK_SIZE],
            )
            .with_dropped_samples(Arc::clone(&dropped_samples)),
            _stop: stop,
            buffer_rate: format.sample_rate,
            live_rate,
            dropped_samples,
        })
    }
}

struct NegotiatedFormat {
    sample_rate: u32,
    channels: u16,
    device_name: String,
}

fn run_capture(
    device_id: Option<&str>,
    process_id: Option<u32>,
    mut producer: HeapProd<f32>,
    waker: Arc<AtomicWaker>,
    wake_pending: Arc<AtomicBool>,
    dropped_samples: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    ready_tx: mpsc::Sender<Result<NegotiatedFormat, String>>,
) -> Result<()> {
    if let Err(error) = initialize_mta().ok() {
        let _ = crate::debug_log::append(&format!("[audio] wasapi COM init: {error}"));
    }

    let format = match process_id {
        Some(pid) => open_process_client(pid)?,
        None => open_device_client(device_id)?,
    };

    let blockalign = format.wave_format.get_blockalign() as usize;
    let channels = format.channels as usize;
    let sample_rate = format.sample_rate;
    let device_name = format.device_name;
    let audio_client = format.client;

    let _ = crate::debug_log::append(&format!(
        "[audio] wasapi start rate={sample_rate} channels={channels} blockalign={blockalign} device={device_name}"
    ));

    let event = audio_client
        .set_get_eventhandle()
        .map_err(|error| anyhow!("event handle failed: {error}"))?;
    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|error| anyhow!("capture client failed: {error}"))?;
    audio_client
        .start_stream()
        .map_err(|error| anyhow!("start stream failed: {error}"))?;

    // Report the negotiated format before entering the loop; `stream()` is
    // blocked waiting for exactly this.
    let _ = ready_tx.send(Ok(NegotiatedFormat {
        sample_rate,
        channels: channels as u16,
        device_name,
    }));

    let mut byte_buffer: Vec<u8> = Vec::new();
    let mut float_buffer: Vec<f32> = Vec::new();
    let mut scratch = vec![0.0f32; rt_ring::DEFAULT_SCRATCH_LEN];
    let mut produced: u64 = 0;
    let started = Instant::now();

    while !stop.load(Ordering::Acquire) {
        // A timeout here is the normal idle case, not an error.
        let _ = event.wait_for_event(EVENT_WAIT_MS);

        drain_packets(
            &capture_client,
            blockalign,
            channels,
            &mut byte_buffer,
            &mut float_buffer,
            &mut scratch,
            &mut producer,
            &mut produced,
            &dropped_samples,
            &waker,
            &wake_pending,
        );

        fill_silence_gap(
            started,
            sample_rate,
            &mut produced,
            &mut scratch,
            &mut producer,
            &waker,
            &wake_pending,
        );
    }

    let _ = audio_client.stop_stream();
    let _ = crate::debug_log::append("[audio] wasapi capture stopped");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn drain_packets(
    capture_client: &AudioCaptureClient,
    blockalign: usize,
    channels: usize,
    byte_buffer: &mut Vec<u8>,
    float_buffer: &mut Vec<f32>,
    scratch: &mut [f32],
    producer: &mut HeapProd<f32>,
    produced: &mut u64,
    dropped_samples: &Arc<AtomicUsize>,
    waker: &Arc<AtomicWaker>,
    wake_pending: &Arc<AtomicBool>,
) {
    loop {
        let frames = match capture_client.get_next_packet_size() {
            Ok(Some(frames)) if frames > 0 => frames as usize,
            _ => break,
        };

        let byte_len = frames * blockalign;
        if byte_buffer.len() < byte_len {
            byte_buffer.resize(byte_len, 0);
        }

        let read = match capture_client.read_from_device(&mut byte_buffer[..byte_len]) {
            Ok((frames_read, _info)) => frames_read as usize,
            Err(error) => {
                let _ = crate::debug_log::append(&format!(
                    "[audio] wasapi read failed: {error}"
                ));
                break;
            }
        };

        if read == 0 {
            break;
        }

        let sample_count = read * channels;
        if float_buffer.len() < sample_count {
            float_buffer.resize(sample_count, 0.0);
        }
        // The client is initialised for Float32 with autoconvert on, so the
        // bytes are little-endian f32 frames. Decoding explicitly keeps this
        // free of alignment assumptions about the driver's buffer.
        for (slot, bytes) in float_buffer[..sample_count]
            .iter_mut()
            .zip(byte_buffer[..sample_count * 4].chunks_exact(4))
        {
            *slot = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }

        let stats = rt_ring::downmix_f32_to_ringbuf(
            &float_buffer[..sample_count],
            channels,
            scratch,
            producer,
        );

        if stats.dropped > 0 {
            dropped_samples.fetch_add(stats.dropped, Ordering::Relaxed);
        }
        *produced += stats.pushed as u64;

        if stats.pushed > 0 && wake_pending.load(Ordering::Acquire) {
            wake_pending.store(false, Ordering::Release);
            waker.wake();
        }
    }
}

/// Emits silence for wall-clock time that produced no packets, so a pause in
/// the meeting still reads as silence instead of vanishing from the timeline.
fn fill_silence_gap(
    started: Instant,
    sample_rate: u32,
    produced: &mut u64,
    scratch: &mut [f32],
    producer: &mut HeapProd<f32>,
    waker: &Arc<AtomicWaker>,
    wake_pending: &Arc<AtomicBool>,
) {
    let expected = (started.elapsed().as_secs_f64() * sample_rate as f64) as u64;
    if expected <= *produced {
        return;
    }

    let deficit = expected - *produced;
    if deficit as f64 > sample_rate as f64 * MAX_SILENCE_CATCHUP_SECONDS {
        // Far behind: the consumer is stalled or the clock jumped. Re-anchor
        // rather than queuing a second's worth of silence at once.
        *produced = expected;
        return;
    }

    let mut pending = deficit as usize;
    let mut pushed_total = 0usize;
    while pending > 0 {
        let batch = pending.min(scratch.len());
        scratch[..batch].fill(0.0);
        let pushed = producer.push_slice(&scratch[..batch]);
        if pushed == 0 {
            break;
        }
        pushed_total += pushed;
        pending -= pushed;
    }

    *produced += pushed_total as u64;

    if pushed_total > 0 && wake_pending.load(Ordering::Acquire) {
        wake_pending.store(false, Ordering::Release);
        waker.wake();
    }
}

struct ClientAndFormat {
    client: AudioClient,
    wave_format: WaveFormat,
    sample_rate: u32,
    channels: u16,
    device_name: String,
}

fn open_device_client(device_id: Option<&str>) -> Result<ClientAndFormat> {
    let enumerator =
        DeviceEnumerator::new().map_err(|error| anyhow!("enumerator failed: {error}"))?;

    let device = match device_id {
        Some(id) if !id.starts_with("pid:") => enumerator
            .get_device(id)
            .map_err(|error| anyhow!("device {id} not found: {error}"))?,
        _ => enumerator
            .get_default_device(&Direction::Render)
            .map_err(|error| anyhow!("no default render device: {error}"))?,
    };

    let device_name = device.get_friendlyname().unwrap_or_default();
    let mut audio_client = device
        .get_iaudioclient()
        .map_err(|error| anyhow!("audio client failed: {error}"))?;

    // Mirror the device's own mix format, but as float: the engine converts, so
    // the rate stays honest and downstream always sees f32.
    let mix = audio_client
        .get_mixformat()
        .map_err(|error| anyhow!("mix format failed: {error}"))?;
    let sample_rate = mix.get_samplespersec();
    let channels = mix.get_nchannels().max(1);
    let wave_format = WaveFormat::new(32, 32, &SampleType::Float, sample_rate as usize, channels as usize, None);

    let (_, min_period) = audio_client
        .get_device_period()
        .map_err(|error| anyhow!("device period failed: {error}"))?;

    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: min_period,
    };
    // Render device + Capture direction is what turns on loopback.
    audio_client
        .initialize_client(&wave_format, &Direction::Capture, &mode)
        .map_err(|error| anyhow!("initialize loopback failed: {error}"))?;

    Ok(ClientAndFormat {
        client: audio_client,
        wave_format,
        sample_rate,
        channels,
        device_name,
    })
}

fn open_process_client(process_id: u32) -> Result<ClientAndFormat> {
    let mut audio_client = AudioClient::new_application_loopback_client(process_id, true)
        .map_err(|error| anyhow!("process loopback for pid {process_id} failed: {error}"))?;

    // Process loopback has no usable mix format or device period, so the only
    // options are a fixed format plus autoconvert.
    let wave_format = WaveFormat::new(
        32,
        32,
        &SampleType::Float,
        FALLBACK_RATE as usize,
        FALLBACK_CHANNELS as usize,
        None,
    );
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: 0,
    };
    audio_client
        .initialize_client(&wave_format, &Direction::Capture, &mode)
        .map_err(|error| anyhow!("initialize process loopback failed: {error}"))?;

    Ok(ClientAndFormat {
        client: audio_client,
        wave_format,
        sample_rate: FALLBACK_RATE,
        channels: FALLBACK_CHANNELS,
        device_name: format!("process {process_id}"),
    })
}

impl Stream for SpeakerStream {
    type Item = f32;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.reader).poll_next_sample(cx).poll
    }
}

impl Drop for SpeakerStream {
    fn drop(&mut self) {
        self._stop.store(true, Ordering::Release);
    }
}
