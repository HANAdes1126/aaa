use super::BatchAsrRequest;
use anyhow::{anyhow, bail, Context, Result};
use std::io::Cursor;
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const TARGET_SAMPLE_RATE: u32 = 16_000;

pub fn normalize_to_wav_16k_mono(request: BatchAsrRequest) -> Result<Vec<u8>> {
    let (sample_rate, mono_samples) = decode_mono(request)?;
    let samples = resample_sinc(&mono_samples, sample_rate, TARGET_SAMPLE_RATE)?;
    crate::audio::wav::encode_wav(TARGET_SAMPLE_RATE, &samples)
}

fn decode_mono(request: BatchAsrRequest) -> Result<(u32, Vec<f32>)> {
    if request.audio_bytes.is_empty() {
        bail!("Cannot normalize empty audio.");
    }

    let mut hint = Hint::new();
    if let Some(extension) = request.filename.rsplit('.').next() {
        hint.with_extension(extension);
    }
    if let Some(subtype) = request.mime_type.split('/').nth(1) {
        hint.with_extension(subtype.split(';').next().unwrap_or(subtype));
    }

    let source = MediaSourceStream::new(
        Box::new(Cursor::new(request.audio_bytes)),
        MediaSourceStreamOptions::default(),
    );
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Unsupported or invalid audio container")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("Audio container has no decodable track"))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Unsupported audio codec")?;
    let mut output = Vec::new();
    let mut source_rate = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(error) => return Err(error).context("Failed to read audio packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => return Err(error).context("Failed to decode audio packet"),
        };
        let spec = *decoded.spec();
        if let Some(rate) = source_rate {
            if rate != spec.rate {
                bail!("Audio sample rate changed during the clip");
            }
        } else {
            source_rate = Some(spec.rate);
        }
        append_mono(&mut output, decoded, spec);
    }

    if output.is_empty() {
        bail!("Audio decoder returned no samples");
    }
    Ok((source_rate.unwrap_or(TARGET_SAMPLE_RATE), output))
}

fn append_mono(
    output: &mut Vec<f32>,
    decoded: symphonia::core::audio::AudioBufferRef<'_>,
    spec: SignalSpec,
) {
    let channels = spec.channels.count();
    let mut samples = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
    samples.copy_interleaved_ref(decoded);
    for frame in samples.samples().chunks(channels) {
        output.push(frame.iter().sum::<f32>() / channels as f32);
    }
}

/// Half-width of the resampling kernel. 33 taps gives roughly -60 dB
/// stopband attenuation with a Hann window, which is enough that nothing
/// audible aliases back into the speech band at a 3:1 downsample.
const RESAMPLE_HALF_TAPS: isize = 16;

/// Windowed-sinc (Lanczos-like) resampling with a real anti-aliasing filter.
///
/// The previous linear interpolation is a very weak low-pass. Downsampling
/// 48 kHz capture to the 16 kHz the ASR model expects folds everything above
/// 8 kHz straight back into the speech band as inharmonic distortion, and the
/// decoder — hearing something that only vaguely resembles speech — falls back
/// on its language prior and invents a plausible-sounding sentence instead.
/// A windowed sinc keeps the stopband around -60 dB so only the band the model
/// was trained on survives the conversion.
fn resample_sinc(samples: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>> {
    if source_rate == 0 || target_rate == 0 {
        bail!("Audio sample rate must be non-zero");
    }
    if source_rate == target_rate || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    let ratio = source_rate as f64 / target_rate as f64;
    let output_len = ((samples.len() as f64) / ratio).ceil() as usize;
    if output_len == 0 {
        return Ok(Vec::new());
    }

    // Cut off at the lower of the two Nyquist limits, expressed as a fraction
    // of the *source* rate (where 0.5 is the source Nyquist). When
    // downsampling, ratio > 1 and the binding limit is the target Nyquist at
    // 0.5 / ratio; when upsampling the source Nyquist (0.5) binds instead.
    // The 0.95 guard pulls the transition band far enough inside the limit
    // that the Hann rolloff is already deep by the time it arrives.
    let cutoff = 0.5 * (1.0f64 / ratio).min(1.0) * 0.95;

    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let center = index as f64 * ratio;
        let start = center.floor() as isize;
        let mut weighted = 0.0f64;
        let mut weight_sum = 0.0f64;
        for tap in -RESAMPLE_HALF_TAPS..=RESAMPLE_HALF_TAPS {
            let position = start + tap;
            if position < 0 || position >= samples.len() as isize {
                continue;
            }
            let weight = sinc_kernel(center - position as f64, cutoff);
            weighted += weight * samples[position as usize] as f64;
            weight_sum += weight;
        }
        // Normalising by the weights actually applied keeps unity gain at the
        // clip edges, where part of the kernel hangs off the end of the buffer.
        output.push(if weight_sum.abs() > 1e-9 {
            (weighted / weight_sum) as f32
        } else {
            0.0
        });
    }
    Ok(output)
}

/// Hann-windowed sinc evaluated at `x` samples from the kernel centre.
fn sinc_kernel(x: f64, cutoff: f64) -> f64 {
    let scaled = 2.0 * cutoff * x;
    let sinc = if scaled.abs() < 1e-9 {
        1.0
    } else {
        let pi_x = std::f64::consts::PI * scaled;
        pi_x.sin() / pi_x
    };
    let window = if x.abs() >= RESAMPLE_HALF_TAPS as f64 {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * x / RESAMPLE_HALF_TAPS as f64).cos())
    };
    2.0 * cutoff * sinc * window
}

pub fn silence_probe_wav() -> Vec<u8> {
    crate::audio::wav::encode_wav(TARGET_SAMPLE_RATE, &vec![0.0; 3_200])
        .expect("silence probe WAV must encode")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_wav_to_16k_mono() {
        let input = crate::audio::wav::encode_wav(48_000, &vec![0.25; 4_800]).unwrap();
        let normalized =
            normalize_to_wav_16k_mono(BatchAsrRequest::new(input, "input.wav", "audio/wav"))
                .unwrap();
        let reader = hound::WavReader::new(Cursor::new(normalized)).unwrap();
        assert_eq!(reader.spec().sample_rate, TARGET_SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.duration(), 1_600);
    }

    fn tone(sample_rate: u32, frequency: f32, seconds: f32) -> Vec<f32> {
        let count = (sample_rate as f32 * seconds) as usize;
        (0..count)
            .map(|index| {
                0.5 * (2.0 * std::f32::consts::PI * frequency * index as f32 / sample_rate as f32)
                    .sin()
            })
            .collect()
    }

    fn peak_of(samples: &[f32]) -> f32 {
        // Skip the ramp-up where the kernel is still filling.
        samples
            .iter()
            .skip(samples.len() / 8)
            .fold(0.0f32, |max, &sample| max.max(sample.abs()))
    }

    #[test]
    fn keeps_energy_in_the_speech_band() {
        let out = resample_sinc(&tone(48_000, 1_000.0, 1.0), 48_000, 16_000).unwrap();
        let peak = peak_of(&out);
        assert!(peak > 0.4, "speech-band tone was attenuated: peak {peak}");
    }

    /// A 20 kHz tone is above the 8 kHz Nyquist of the 16 kHz target. With
    /// no anti-aliasing filter it folds down to ~4 kHz — squarely inside the
    /// speech band, where it corrupts every formant estimate. The sinc kernel
    /// has to remove it.
    #[test]
    fn suppresses_energy_above_the_new_nyquist() {
        let out = resample_sinc(&tone(48_000, 20_000.0, 1.0), 48_000, 16_000).unwrap();
        let peak = peak_of(&out);
        assert!(
            peak < 0.02,
            "ultrasonic energy aliased into the speech band: peak {peak}"
        );
    }

    #[test]
    fn resampling_preserves_duration() {
        let out = resample_sinc(&vec![0.25; 48_000], 48_000, 16_000).unwrap();
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn three_minute_wav_fits_mimo_base64_limit() {
        let bytes = crate::audio::wav::encode_wav(
            TARGET_SAMPLE_RATE,
            &vec![0.0; TARGET_SAMPLE_RATE as usize * 180],
        )
        .unwrap();
        let encoded_size = "data:audio/wav;base64,".len() + bytes.len().div_ceil(3) * 4;
        assert!(encoded_size < 10_000_000);
    }

    #[test]
    #[ignore = "requires MEETLY_TEST_AUDIO_PATH"]
    fn normalizes_external_encoded_clip() {
        let path = std::env::var("MEETLY_TEST_AUDIO_PATH")
            .expect("MEETLY_TEST_AUDIO_PATH must point to a local audio fixture");
        let bytes = std::fs::read(&path).unwrap();
        let extension = std::path::Path::new(&path)
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("mp4");
        let normalized = normalize_to_wav_16k_mono(BatchAsrRequest::new(
            bytes,
            &format!("fixture.{extension}"),
            &format!("audio/{extension}"),
        ))
        .unwrap();
        let reader = hound::WavReader::new(Cursor::new(normalized)).unwrap();
        assert_eq!(reader.spec().sample_rate, TARGET_SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        assert!(reader.duration() > 0);
    }
}
