#[cfg(not(any(target_os = "macos", target_os = "windows")))]
use anyhow::anyhow;
use anyhow::Result;
use futures_util::Stream;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::sync::Arc;

mod async_ring;
#[cfg(target_os = "macos")]
mod macos;
mod rt_ring;
#[cfg(target_os = "windows")]
mod windows;

/// Samples pulled from the ring buffer per wake. At 256 this was a park/
/// unpark cycle through the executor every 5.3 ms at 48 kHz — roughly 190
/// wakeups a second, each one a chance to lose a wake and stall the
/// consumer. 2048 stretches the interval to ~43 ms with no added latency.
const CHUNK_SIZE: usize = 2048;
/// ~5.5 s at 48 kHz, independent of `CHUNK_SIZE` so the two can be tuned
/// separately. Every overrun silently deletes samples from the middle of the
/// stream and time-compresses the audio the ASR receives.
const BUFFER_SIZE: usize = 262_144;

#[derive(Debug, Default)]
pub struct DeviceProbe {
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub error: Option<String>,
}

pub fn probe_devices() -> DeviceProbe {
    #[cfg(target_os = "macos")]
    {
        return macos::probe_devices();
    }

    #[cfg(target_os = "windows")]
    {
        return windows::probe_devices();
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        DeviceProbe {
            error: Some(
                "System audio capture is only implemented on macOS and Windows.".to_string(),
            ),
            ..DeviceProbe::default()
        }
    }
}

pub struct SpeakerInput {
    #[cfg(target_os = "macos")]
    inner: macos::SpeakerInput,
    #[cfg(target_os = "windows")]
    inner: windows::SpeakerInput,
}

impl SpeakerInput {
    pub fn new(device_id: Option<String>) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            return Ok(Self {
                inner: macos::SpeakerInput::new(device_id)?,
            });
        }

        #[cfg(target_os = "windows")]
        {
            return Ok(Self {
                inner: windows::SpeakerInput::new(device_id)?,
            });
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            let _ = device_id;
            Err(anyhow!(
                "System audio capture is only implemented on macOS and Windows."
            ))
        }
    }

    pub fn stream(self) -> Result<SpeakerStream> {
        #[cfg(target_os = "macos")]
        {
            return Ok(SpeakerStream {
                inner: self.inner.stream()?,
            });
        }

        #[cfg(target_os = "windows")]
        {
            return Ok(SpeakerStream {
                inner: self.inner.stream()?,
            });
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            let _ = self;
            Err(anyhow!(
                "System audio capture is only implemented on macOS and Windows."
            ))
        }
    }
}

pub struct SpeakerStream {
    #[cfg(target_os = "macos")]
    inner: macos::SpeakerStream,
    #[cfg(target_os = "windows")]
    inner: windows::SpeakerStream,
}

impl SpeakerStream {
    pub fn sample_rate(&self) -> u32 {
        #[cfg(target_os = "macos")]
        {
            return self.inner.sample_rate();
        }

        #[cfg(target_os = "windows")]
        {
            return self.inner.sample_rate();
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            0
        }
    }

    /// Live view of the capture rate. Reads `0` on platforms without a tap
    /// implementation; callers fall back to the rate observed at startup.
    pub fn live_rate(&self) -> Arc<AtomicU32> {
        #[cfg(target_os = "macos")]
        {
            return self.inner.live_rate();
        }

        #[cfg(target_os = "windows")]
        {
            return self.inner.live_rate();
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            Arc::new(AtomicU32::new(0))
        }
    }

    /// Samples lost to ring-buffer overrun since the last read.
    pub fn dropped_samples(&self) -> Arc<AtomicUsize> {
        #[cfg(target_os = "macos")]
        {
            return self.inner.dropped_samples();
        }

        #[cfg(target_os = "windows")]
        {
            return self.inner.dropped_samples();
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            Arc::new(AtomicUsize::new(0))
        }
    }
}

impl Stream for SpeakerStream {
    type Item = f32;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        #[cfg(target_os = "macos")]
        {
            return Pin::new(&mut self.inner).poll_next(cx);
        }

        #[cfg(target_os = "windows")]
        {
            return Pin::new(&mut self.inner).poll_next(cx);
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            let _ = cx;
            std::task::Poll::Ready(None)
        }
    }
}
