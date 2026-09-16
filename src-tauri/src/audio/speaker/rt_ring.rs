use ringbuf::traits::{Observer, Producer};

pub(crate) const DEFAULT_SCRATCH_LEN: usize = 8192;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PushStats {
    pub(crate) pushed: usize,
    pub(crate) dropped: usize,
}

/// Converts to f32, averages interleaved channels down to mono, and pushes.
///
/// The tap follows the output device, which is stereo far more often than not.
/// Pushing interleaved samples straight into a mono ring buffer doubles the
/// sample rate *and* interleaves two signals, so the ASR hears a track that is
/// both the wrong speed and mixed with itself. Averaging per frame is what
/// the downstream mono pipeline already assumes.
pub(crate) fn convert_and_push_to_ringbuf<T, P>(
    samples: &[T],
    scratch: &mut [f32],
    producer: &mut P,
    channels: usize,
    mut convert: impl FnMut(T) -> f32,
) -> PushStats
where
    T: Copy,
    P: Producer<Item = f32> + Observer,
{
    if scratch.is_empty() || samples.is_empty() {
        return PushStats::default();
    }

    let channels = channels.max(1);
    let frames = samples.len() / channels;
    let mut offset = 0usize;
    let mut pushed_total = 0usize;
    let mut dropped_total = 0usize;

    while offset < frames {
        let count = (frames - offset).min(scratch.len());
        let vacant = producer.vacant_len();

        if vacant == 0 {
            dropped_total += frames - offset;
            break;
        }

        let convert_count = count.min(vacant);

        for i in 0..convert_count {
            let base = (offset + i) * channels;
            let mut sum = 0.0f32;
            for channel in 0..channels {
                if let Some(&sample) = samples.get(base + channel) {
                    sum += convert(sample);
                }
            }
            scratch[i] = sum / channels as f32;
        }

        let pushed = producer.push_slice(&scratch[..convert_count]);
        pushed_total += pushed;
        dropped_total += convert_count - pushed;
        if pushed < convert_count {
            dropped_total += frames - offset - convert_count;
            break;
        }
        offset += convert_count;
    }

    PushStats {
        pushed: pushed_total,
        // Any trailing partial frame cannot form a mono sample.
        dropped: dropped_total + (samples.len() % channels),
    }
}

/// Same downmix for taps that already deliver f32, so no conversion is needed.
pub(crate) fn downmix_f32_to_ringbuf<P>(
    samples: &[f32],
    channels: usize,
    scratch: &mut [f32],
    producer: &mut P,
) -> PushStats
where
    P: Producer<Item = f32>,
{
    if samples.is_empty() {
        return PushStats::default();
    }
    if channels <= 1 || scratch.is_empty() {
        return push_f32_to_ringbuf(samples, producer);
    }

    let frames = samples.len() / channels;
    let mut offset = 0usize;
    let mut pushed_total = 0usize;
    let mut dropped_total = 0usize;

    while offset < frames {
        let count = (frames - offset).min(scratch.len());
        for i in 0..count {
            let base = (offset + i) * channels;
            let mut sum = 0.0f32;
            for channel in 0..channels {
                if let Some(&sample) = samples.get(base + channel) {
                    sum += sample;
                }
            }
            scratch[i] = sum / channels as f32;
        }

        let pushed = producer.push_slice(&scratch[..count]);
        pushed_total += pushed;
        dropped_total += count - pushed;
        if pushed < count {
            dropped_total += frames - offset - count;
            break;
        }
        offset += count;
    }

    PushStats {
        pushed: pushed_total,
        dropped: dropped_total + (samples.len() % channels),
    }
}

pub(crate) fn push_f32_to_ringbuf<P>(data: &[f32], producer: &mut P) -> PushStats
where
    P: Producer<Item = f32>,
{
    if data.is_empty() {
        return PushStats::default();
    }

    let pushed = producer.push_slice(data);
    PushStats {
        pushed,
        dropped: data.len() - pushed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ringbuf::traits::{Consumer, Split};
    use ringbuf::HeapRb;

    fn drained(consumer: &mut impl Consumer<Item = f32>) -> Vec<f32> {
        let mut out = vec![0.0f32; 64];
        let count = consumer.pop_slice(&mut out);
        out.truncate(count);
        out
    }

    #[test]
    fn stereo_frames_are_averaged_into_one_mono_track() {
        let (mut producer, mut consumer) = HeapRb::<f32>::new(16).split();
        let mut scratch = vec![0.0; 8];

        let stats = downmix_f32_to_ringbuf(
            &[1.0, 0.0, 0.6, 0.2],
            2,
            &mut scratch,
            &mut producer,
        );

        assert_eq!((stats.pushed, stats.dropped), (2, 0));
        let out = drained(&mut consumer);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 1e-6, "got {:?}", out);
        assert!((out[1] - 0.4).abs() < 1e-6, "got {:?}", out);
    }

    #[test]
    fn mono_input_is_passed_through_untouched() {
        let (mut producer, mut consumer) = HeapRb::<f32>::new(16).split();
        let mut scratch = vec![0.0; 8];

        let stats = downmix_f32_to_ringbuf(&[0.1, 0.2, 0.3], 1, &mut scratch, &mut producer);

        assert_eq!((stats.pushed, stats.dropped), (3, 0));
        assert_eq!(drained(&mut consumer), vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn interleaved_stereo_is_not_mistaken_for_double_the_audio() {
        // The bug this guards: 4 interleaved stereo samples are 2 frames of
        // audio, not 4. Without downmixing, a stereo tap looks like 2x the
        // sample rate and the segment is half as long as the speaker talked.
        let (mut producer, mut consumer) = HeapRb::<f32>::new(16).split();
        let mut scratch = vec![0.0; 8];

        let stats = convert_and_push_to_ringbuf(
            &[2i16, 2, 4, 4],
            &mut scratch,
            &mut producer,
            2,
            |sample| sample as f32,
        );

        assert_eq!((stats.pushed, stats.dropped), (2, 0));
        assert_eq!(drained(&mut consumer), vec![2.0, 4.0]);
    }

    #[test]
    fn overrun_reports_every_sample_it_could_not_store() {
        let (mut producer, mut consumer) = HeapRb::<f32>::new(2).split();
        let mut scratch = vec![0.0; 8];

        let stats = downmix_f32_to_ringbuf(&[0.1, 0.1, 0.2, 0.2, 0.3, 0.3], 2, &mut scratch, &mut producer);

        assert_eq!(stats.pushed, 2);
        assert_eq!(stats.dropped, 1);
        assert_eq!(drained(&mut consumer).len(), 2);
    }
}
