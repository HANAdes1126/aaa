//! Energy-threshold voice activity detection, segmenting a continuous PCM
//! stream into discrete speech phrases. Ports the algorithm from
//! `pluely-master`'s `run_vad_capture`
//! (src-tauri/src/speaker/commands.rs:135-257 in that project), using the
//! parameter values already specified in docs/TECHNICAL_DESIGN.md section
//! 4.4. No neural VAD model; energy threshold is enough for system-audio
//! input, which is relatively clean compared to a live microphone.
//!
//! # Why the threshold adapts
//!
//! A fixed RMS threshold only works when the noise floor is stable and low.
//! Real meetings are not: fan noise, a colleague's open mic, meeting-app
//! processing noise and room tone all push the floor up. Once the floor sits
//! above the fixed threshold, every hop looks like speech, the segmenter
//! never sees a silence gap, and each segment is only ever ended by the
//! hard duration cap. That produces 30-second slabs of mostly noise, which
//! is exactly what the local ASR chokes on.
//!
//! So the threshold is now `noise_floor * multiplier`, where the floor is
//! tracked as a smoothed minimum over a trailing window (minimum-statistics
//! style: the quietest hop in the last few seconds is, by definition, the
//! noise). Rising floors are tracked faster than falling ones so the
//! estimator recovers from a sudden noise source while still not chasing
//! individual syllables.

use std::collections::VecDeque;

/// Tunable thresholds. Values match docs/TECHNICAL_DESIGN.md section 4.4
/// and pluely-master's defaults.
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Number of samples analyzed per RMS/peak check.
    pub hop_size: usize,
    /// Baseline RMS level above which a hop is considered speech. Used as the
    /// *starting* threshold and as the clamp target; once the noise floor
    /// estimate diverges from it, the adaptive threshold takes over.
    pub sensitivity_rms: f32,
    /// Peak level above which a hop is considered speech (OR'd with RMS).
    pub peak_threshold: f32,
    /// Soft-knee noise gate threshold applied before RMS/peak calculation.
    pub noise_gate_threshold: f32,
    /// Minimum consecutive silent hops after speech before the segment ends.
    pub silence_hops: u32,
    /// Minimum consecutive speech hops for a segment to be kept (shorter
    /// bursts are discarded as noise/clicks).
    pub min_speech_hops: u32,
    /// Rolling pre-speech buffer length in hops, prepended to a segment so
    /// the first word isn't clipped.
    pub pre_speech_hops: u32,
    /// Hard cap on segment length; a segment is force-flushed at this point
    /// even without a silence gap.
    pub max_segment_samples: usize,
    /// Hops kept in the trailing window used to estimate the noise floor.
    pub noise_window_hops: usize,
    /// EMA rate when the estimated floor rises (react quickly to a new noise
    /// source: a fan switching on, someone unmuting).
    pub noise_floor_alpha_up: f32,
    /// EMA rate when the floor falls (slow, so brief pauses don't drag the
    /// threshold down onto the noise).
    pub noise_floor_alpha_down: f32,
    /// How far above the noise floor a hop must sit to count as speech.
    pub noise_floor_multiplier: f32,
    /// Lower bound for the adaptive RMS threshold; the threshold never falls
    /// below this even in a perfectly silent room.
    pub min_rms_threshold: f32,
    /// Upper bound for the adaptive RMS threshold; in an extremely loud room
    /// we still prefer noise over dropping a quiet speaker entirely.
    pub max_rms_threshold: f32,
    /// Lower/upper bounds for the adaptive peak threshold (same reasoning).
    pub min_peak_threshold: f32,
    pub max_peak_threshold: f32,
    /// A completed segment whose overall RMS sits below this is pure noise and
    /// is discarded instead of being sent to ASR.
    pub quiet_segment_rms: f32,
    /// How far back from the hard cap the segmenter looks for a low-energy
    /// point to split on, so a long monologue is cut between words rather
    /// than mid-syllable.
    pub trough_search_samples: usize,
}

impl VadConfig {
    /// Builds a config from the millisecond-based parameters specified in
    /// docs/TECHNICAL_DESIGN.md section 4.4, resolved against the actual
    /// capture sample rate.
    pub fn from_millis(sample_rate: u32) -> Self {
        const HOP_SIZE: usize = 1024;
        let hop_duration_ms = (HOP_SIZE as f32 / sample_rate as f32) * 1000.0;

        const MIN_SPEECH_MS: f32 = 300.0;
        // Widened back to 1000ms: 300ms was too aggressive — natural "thinking
        // pause" mid-sentence (e.g. explaining a concept like 双亲委派) is
        // usually 0.5-1.5s, and cutting on 300ms splits a single explanation
        // into useless 1-2 sentence fragments that don't carry enough context
        // for the coach to analyze. 1s strikes a balance: long enough to ride
        // over mid-sentence pauses, short enough to still feel responsive
        // when the interviewer actually finishes a question.
        const END_SILENCE_MS: f32 = 1000.0;
        // Lowered from 30s. The cap is only a safety net: with an adaptive
        // noise floor the normal exit is the 1s silence gap above, so 30s was
        // doing nothing except giving the ASR an untranscribable 30-second
        // slab whenever VAD did get stuck (observed: a whole 42-minute
        // meeting cut into exactly 30s pieces). 15s still covers a complete
        // technical explanation, and the force-flush now splits at a
        // low-energy point so the cut lands between words.
        const MAX_SEGMENT_MS: f32 = 15_000.0;
        const PRE_ROLL_MS: f32 = 300.0;
        // ~6s window: long enough to contain a between-words gap (so speech
        // doesn't pull the floor up), short enough to adapt within a few
        // seconds of the noise character changing.
        const NOISE_WINDOW_MS: f32 = 6_000.0;
        const TROUGH_SEARCH_MS: f32 = 2_000.0;

        let noise_window_hops = (NOISE_WINDOW_MS / hop_duration_ms).ceil() as usize;

        Self {
            hop_size: HOP_SIZE,
            sensitivity_rms: 0.012,
            peak_threshold: 0.035,
            noise_gate_threshold: 0.003,
            silence_hops: (END_SILENCE_MS / hop_duration_ms).ceil() as u32,
            min_speech_hops: (MIN_SPEECH_MS / hop_duration_ms).ceil() as u32,
            pre_speech_hops: (PRE_ROLL_MS / hop_duration_ms).ceil() as u32,
            max_segment_samples: ((MAX_SEGMENT_MS / 1000.0) * sample_rate as f32) as usize,
            noise_window_hops: noise_window_hops.max(8),
            noise_floor_alpha_up: 0.2,
            noise_floor_alpha_down: 0.05,
            noise_floor_multiplier: 3.0,
            min_rms_threshold: 0.0025,
            max_rms_threshold: 0.05,
            min_peak_threshold: 0.008,
            max_peak_threshold: 0.12,
            quiet_segment_rms: 0.003,
            trough_search_samples: ((TROUGH_SEARCH_MS / 1000.0) * sample_rate as f32) as usize,
        }
    }

    /// Config used by the unit tests: tiny hop so a few samples exercise a
    /// full state transition, adaptive floor enabled with a short window.
    #[cfg(test)]
    fn testing(hop_size: usize) -> Self {
        Self {
            hop_size,
            sensitivity_rms: 0.01,
            peak_threshold: 0.01,
            noise_gate_threshold: 0.0,
            silence_hops: 2,
            min_speech_hops: 3,
            pre_speech_hops: 1,
            max_segment_samples: 1000,
            noise_window_hops: 16,
            noise_floor_alpha_up: 0.2,
            noise_floor_alpha_down: 0.05,
            noise_floor_multiplier: 3.0,
            min_rms_threshold: 0.0025,
            max_rms_threshold: 0.05,
            min_peak_threshold: 0.008,
            max_peak_threshold: 0.12,
            quiet_segment_rms: 0.003,
            trough_search_samples: hop_size * 4,
        }
    }
}

/// A completed speech segment, ready to be WAV-encoded and transcribed.
pub struct SpeechSegment {
    pub samples: Vec<f32>,
}

/// What happened as a result of feeding one sample into the segmenter.
pub enum SegmenterEvent {
    /// No state change worth reporting.
    None,
    /// Speech just started (crossed from silence into the speech state).
    SpeechStarted,
    /// A complete segment is ready (either a natural silence-triggered end,
    /// or a max-duration force-flush).
    SegmentReady(SpeechSegment),
    /// A burst was too short to count as real speech, or too quiet overall;
    /// it was discarded instead of being transcribed.
    Discarded,
}

/// Minimum-statistics noise floor tracker.
///
/// The quietest hop in the trailing window is the noise: speech is
/// intermittent, background noise is not. Smoothing that minimum with an
/// asymmetric EMA keeps the estimate stable without letting it get stuck
/// when the noise character changes.
#[derive(Debug)]
struct NoiseFloor {
    window: VecDeque<(f32, f32)>,
    capacity: usize,
    rms: f32,
    peak: f32,
    alpha_up: f32,
    alpha_down: f32,
}

impl NoiseFloor {
    fn new(config: &VadConfig) -> Self {
        // Start from the legacy fixed thresholds divided by the multiplier so
        // that, before any adaptation has happened, the effective thresholds
        // are exactly the old fixed values. Behaviour on a clean signal is
        // therefore unchanged.
        Self {
            window: VecDeque::with_capacity(config.noise_window_hops),
            capacity: config.noise_window_hops,
            rms: config.sensitivity_rms / config.noise_floor_multiplier,
            peak: config.peak_threshold / config.noise_floor_multiplier,
            alpha_up: config.noise_floor_alpha_up,
            alpha_down: config.noise_floor_alpha_down,
        }
    }

    fn push(&mut self, rms: f32, peak: f32) {
        self.window.push_back((rms, peak));
        while self.window.len() > self.capacity {
            self.window.pop_front();
        }

        let mut min_rms = f32::MAX;
        let mut min_peak = f32::MAX;
        for &(hop_rms, hop_peak) in &self.window {
            if hop_rms < min_rms {
                min_rms = hop_rms;
            }
            if hop_peak < min_peak {
                min_peak = hop_peak;
            }
        }
        if !min_rms.is_finite() {
            min_rms = 0.0;
        }
        if !min_peak.is_finite() {
            min_peak = 0.0;
        }

        // Bootstrap: for the first few hops there is nothing to smooth yet, so
        // take the observed minimum directly. Without this the first second or
        // two of a noisy meeting is segmented against the optimistic default
        // floor and produces one junk segment before the estimator catches up.
        if self.window.len() <= BOOTSTRAP_HOPS {
            self.rms = min_rms;
            self.peak = min_peak;
            return;
        }

        self.rms = approach(self.rms, min_rms, self.alpha_up, self.alpha_down);
        self.peak = approach(self.peak, min_peak, self.alpha_up, self.alpha_down);
    }
}

/// Hops tracked before the floor estimate switches from "take the minimum"
/// to "smooth the minimum".
const BOOTSTRAP_HOPS: usize = 8;

/// Moves `current` toward `target`, using `alpha_up` when the target is above
/// the current value and `alpha_down` when it is below.
fn approach(current: f32, target: f32, alpha_up: f32, alpha_down: f32) -> f32 {
    let alpha = if target > current { alpha_up } else { alpha_down };
    current + (target - current) * alpha
}

/// Stateful energy-threshold segmenter. Feed it samples one at a time via
/// `push`; it reports segment boundaries through `SegmenterEvent`.
pub struct Segmenter {
    config: VadConfig,
    hop_buffer: Vec<f32>,
    pre_speech: VecDeque<f32>,
    speech_buffer: Vec<f32>,
    in_speech: bool,
    silence_hop_count: u32,
    speech_hop_count: u32,
    noise_floor: NoiseFloor,
}

impl Segmenter {
    pub fn new(config: VadConfig) -> Self {
        let noise_floor = NoiseFloor::new(&config);
        Self {
            hop_buffer: Vec::with_capacity(config.hop_size),
            pre_speech: VecDeque::with_capacity(config.pre_speech_hops as usize * config.hop_size),
            speech_buffer: Vec::new(),
            in_speech: false,
            silence_hop_count: 0,
            speech_hop_count: 0,
            noise_floor,
            config,
        }
    }

    /// Current RMS threshold, i.e. the noise floor scaled by the multiplier
    /// and clamped. Exposed mainly for diagnostics and tests.
    pub fn rms_threshold(&self) -> f32 {
        (self.noise_floor.rms * self.config.noise_floor_multiplier)
            .clamp(self.config.min_rms_threshold, self.config.max_rms_threshold)
    }

    /// Current peak threshold (same derivation as `rms_threshold`).
    pub fn peak_threshold(&self) -> f32 {
        (self.noise_floor.peak * self.config.noise_floor_multiplier)
            .clamp(self.config.min_peak_threshold, self.config.max_peak_threshold)
    }

    /// Feeds one PCM sample into the segmenter. Returns an event if this
    /// sample completed a hop-sized analysis window and something notable
    /// happened.
    pub fn push(&mut self, sample: f32) -> SegmenterEvent {
        self.hop_buffer.push(sample);
        if self.hop_buffer.len() < self.config.hop_size {
            return SegmenterEvent::None;
        }

        let hop = std::mem::replace(
            &mut self.hop_buffer,
            Vec::with_capacity(self.config.hop_size),
        );
        self.process_hop(hop)
    }

    fn process_hop(&mut self, hop: Vec<f32>) -> SegmenterEvent {
        let gated = apply_noise_gate(&hop, self.config.noise_gate_threshold);
        let (rms, peak) = calculate_metrics(&gated);
        self.noise_floor.push(rms, peak);

        let is_speech = rms > self.rms_threshold() || peak > self.peak_threshold();

        if is_speech {
            self.handle_speech_hop(gated)
        } else {
            self.handle_silence_hop(gated)
        }
    }

    fn handle_speech_hop(&mut self, hop: Vec<f32>) -> SegmenterEvent {
        let just_started = !self.in_speech;
        if just_started {
            self.in_speech = true;
            self.speech_hop_count = 0;
            self.speech_buffer.extend(self.pre_speech.drain(..));
        }

        self.speech_hop_count += 1;
        self.speech_buffer.extend_from_slice(&hop);
        self.silence_hop_count = 0;

        if self.speech_buffer.len() >= self.config.max_segment_samples {
            return self.emit_force_flushed();
        }

        if just_started {
            SegmenterEvent::SpeechStarted
        } else {
            SegmenterEvent::None
        }
    }

    fn handle_silence_hop(&mut self, hop: Vec<f32>) -> SegmenterEvent {
        if !self.in_speech {
            self.pre_speech.extend(hop);
            let max_len = self.config.pre_speech_hops as usize * self.config.hop_size;
            while self.pre_speech.len() > max_len {
                self.pre_speech.pop_front();
            }
            return SegmenterEvent::None;
        }

        self.silence_hop_count += 1;
        self.speech_buffer.extend_from_slice(&hop);

        if self.silence_hop_count < self.config.silence_hops {
            return SegmenterEvent::None;
        }

        // Silence gap satisfied: end the segment.
        let long_enough = self.speech_hop_count >= self.config.min_speech_hops;
        let segment_samples = std::mem::take(&mut self.speech_buffer);
        self.in_speech = false;
        self.silence_hop_count = 0;
        self.speech_hop_count = 0;

        if long_enough && is_audible(&segment_samples, &self.config) {
            SegmenterEvent::SegmentReady(SpeechSegment {
                samples: segment_samples,
            })
        } else {
            SegmenterEvent::Discarded
        }
    }

    /// Hard cap reached. Rather than cutting exactly at the cap — which lands
    /// mid-word and hands the ASR a fragment starting and ending in the middle
    /// of a syllable — back up to the quietest hop-sized position inside the
    /// trailing search window and split there. The remainder stays in
    /// `speech_buffer` and becomes the start of the next segment, so nothing
    /// is lost and the speaker is still considered to be talking.
    fn emit_force_flushed(&mut self) -> SegmenterEvent {
        let hop = self.config.hop_size.max(1);
        let search_start = self
            .speech_buffer
            .len()
            .saturating_sub(self.config.trough_search_samples.max(hop));

        let mut cut_at = self.speech_buffer.len();
        let mut lowest = f32::MAX;
        let mut offset = search_start;
        while offset + hop <= self.speech_buffer.len() {
            let (rms, _) = calculate_metrics(&self.speech_buffer[offset..offset + hop]);
            if rms < lowest {
                lowest = rms;
                cut_at = offset + hop;
            }
            offset += hop;
        }

        if cut_at >= self.speech_buffer.len() {
            // No usable split point: fall back to cutting at the cap.
            let segment = self.flush_segment();
            return if is_audible(&segment.samples, &self.config) {
                SegmenterEvent::SegmentReady(segment)
            } else {
                SegmenterEvent::Discarded
            };
        }

        let tail = self.speech_buffer[cut_at..].to_vec();
        self.speech_buffer.truncate(cut_at);
        let segment = std::mem::take(&mut self.speech_buffer);
        self.speech_buffer = tail;
        self.silence_hop_count = 0;
        self.speech_hop_count = (self.speech_buffer.len() / hop) as u32;

        if is_audible(&segment, &self.config) {
            SegmenterEvent::SegmentReady(SpeechSegment { samples: segment })
        } else {
            SegmenterEvent::Discarded
        }
    }

    fn flush_segment(&mut self) -> SpeechSegment {
        let segment_samples = std::mem::take(&mut self.speech_buffer);
        self.in_speech = false;
        self.silence_hop_count = 0;
        self.speech_hop_count = 0;
        SpeechSegment {
            samples: segment_samples,
        }
    }
}

/// A segment only goes to ASR if it actually carries signal. Without this the
/// segmenter happily emits a slab of room tone whenever the adaptive floor is
/// still converging (e.g. the first seconds after capture starts), and the
/// local ASR turns steady noise into a hallucinated loop.
fn is_audible(samples: &[f32], config: &VadConfig) -> bool {
    if samples.is_empty() {
        return false;
    }
    let (rms, _) = calculate_metrics(samples);
    rms >= config.quiet_segment_rms
}

/// Soft-knee noise gate: samples below `threshold` are compressed toward
/// zero rather than hard-clipped, matching pluely-master's
/// `apply_noise_gate` (src-tauri/src/speaker/commands.rs:363-378).
fn apply_noise_gate(samples: &[f32], threshold: f32) -> Vec<f32> {
    const KNEE_RATIO: f32 = 3.0;

    if threshold <= 0.0 {
        return samples.to_vec();
    }

    samples
        .iter()
        .map(|&sample| {
            let magnitude = sample.abs();
            if magnitude < threshold {
                sample * (magnitude / threshold).powf(1.0 / KNEE_RATIO)
            } else {
                sample
            }
        })
        .collect()
}

fn calculate_metrics(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }

    let mut sum_sq = 0.0f32;
    let mut peak = 0.0f32;
    for &sample in samples {
        let magnitude = sample.abs();
        peak = peak.max(magnitude);
        sum_sq += sample * sample;
    }

    ((sum_sq / samples.len() as f32).sqrt(), peak)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silent_hop(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn loud_hop(len: usize) -> Vec<f32> {
        vec![0.5; len]
    }

    /// Deterministic pseudo-noise in [-amplitude, amplitude].
    ///
    /// splitmix64 over a sequential counter: a plain LCG's high bits are not
    /// uniform enough over the 8-sample hops used here (measured RMS swung
    /// between 0.005 and 0.135 for the same amplitude, which made every
    /// threshold assertion a coin flip).
    fn noise(seed: u64, len: usize, amplitude: f32) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..len)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let unit = ((z >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0;
                unit * amplitude
            })
            .collect()
    }

    fn feed(segmenter: &mut Segmenter, samples: &[f32]) -> Vec<SegmenterEvent> {
        let mut events = Vec::new();
        for &sample in samples {
            match segmenter.push(sample) {
                SegmenterEvent::None => {}
                event => events.push(event),
            }
        }
        events
    }

    #[test]
    fn discards_short_burst() {
        let mut segmenter = Segmenter::new(VadConfig::testing(4));
        let mut last_event = None;
        for hop in [loud_hop(4), silent_hop(4), silent_hop(4)] {
            for sample in hop {
                let event = segmenter.push(sample);
                if !matches!(event, SegmenterEvent::None) {
                    last_event = Some(event);
                }
            }
        }
        assert!(matches!(last_event, Some(SegmenterEvent::Discarded)));
    }

    #[test]
    fn emits_segment_after_enough_speech_then_silence() {
        let mut config = VadConfig::testing(4);
        config.min_speech_hops = 2;
        let mut segmenter = Segmenter::new(config);
        let mut segment_ready = false;
        for hop in [loud_hop(4), loud_hop(4), silent_hop(4), silent_hop(4)] {
            for sample in hop {
                if let SegmenterEvent::SegmentReady(segment) = segmenter.push(sample) {
                    assert!(!segment.samples.is_empty());
                    segment_ready = true;
                }
            }
        }
        assert!(segment_ready);
    }

    #[test]
    fn force_flushes_at_max_duration() {
        let mut config = VadConfig::testing(2);
        config.silence_hops = 100;
        config.min_speech_hops = 1;
        config.max_segment_samples = 4;
        let mut segmenter = Segmenter::new(config);

        let mut segment_ready = false;
        for _ in 0..10 {
            if let SegmenterEvent::SegmentReady(_) = segmenter.push(0.5) {
                segment_ready = true;
                break;
            }
        }
        assert!(segment_ready);
    }

    /// The regression this whole change exists for: a steady noise floor above
    /// the legacy fixed threshold must eventually be recognised as silence,
    /// instead of producing one endless "speech" segment punctuated only by
    /// the hard cap.
    #[test]
    fn adapts_to_a_steady_noise_floor() {
        let mut segmenter = Segmenter::new(VadConfig::testing(8));
        // Noise at RMS ~0.02, well above the legacy 0.012 fixed threshold.
        let floor = noise(1, 8, 0.035);
        let mut stream = Vec::new();
        for _ in 0..200 {
            stream.extend_from_slice(&floor);
        }

        let events = feed(&mut segmenter, &stream);
        let segments = events
            .iter()
            .filter(|event| matches!(event, SegmenterEvent::SegmentReady(_)))
            .count();

        // The floor rose, so after adaptation the noise is silence: at most
        // the single segment produced while the estimator was still
        // converging, and never a stream of them.
        assert!(
            segments <= 1,
            "steady noise produced {segments} segments; threshold={}",
            segmenter.rms_threshold()
        );
        assert!(
            segmenter.rms_threshold() > 0.012,
            "threshold should have risen above the legacy fixed value, got {}",
            segmenter.rms_threshold()
        );
    }

    #[test]
    fn still_detects_speech_above_an_adapted_noise_floor() {
        let mut segmenter = Segmenter::new(VadConfig::testing(8));
        let floor = noise(2, 8, 0.035);
        let speech = noise(3, 8, 0.30);

        // 100 hops of noise to let the floor adapt ...
        for _ in 0..100 {
            feed(&mut segmenter, &floor);
        }
        // ... then a loud phrase, then noise again to close the segment.
        let mut phrase = Vec::new();
        for _ in 0..20 {
            phrase.extend_from_slice(&speech);
        }
        for _ in 0..50 {
            phrase.extend_from_slice(&floor);
        }

        let events = feed(&mut segmenter, &phrase);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, SegmenterEvent::SegmentReady(_))),
            "speech above an adapted noise floor was not segmented"
        );
    }

    #[test]
    fn clean_silence_keeps_the_lowest_threshold() {
        let mut segmenter = Segmenter::new(VadConfig::testing(8));
        for _ in 0..100 {
            feed(&mut segmenter, &silent_hop(8));
        }
        let config = VadConfig::testing(8);
        assert!(
            (segmenter.rms_threshold() - config.min_rms_threshold).abs() < 1e-6,
            "in true silence the threshold should sit at the floor, got {}",
            segmenter.rms_threshold()
        );
    }

    /// A segment carrying no real signal must never reach ASR: it is the input
    /// that makes the local model hallucinate a loop.
    #[test]
    fn discards_segments_that_are_only_noise() {
        let mut config = VadConfig::testing(8);
        config.min_speech_hops = 1;
        config.silence_hops = 6;
        // Gate raised above the test signal so the assertion is about the
        // wiring, not about the exact production constant.
        config.quiet_segment_rms = 0.05;
        let mut segmenter = Segmenter::new(config);

        // Loud enough per-hop to open a segment (~0.069 RMS), but the segment
        // as a whole — three hops of signal plus six of silence — averages
        // below the gate.
        let mut stream = Vec::new();
        for _ in 0..3 {
            stream.extend_from_slice(&noise(7, 8, 0.12));
        }
        for _ in 0..6 {
            stream.extend_from_slice(&silent_hop(8));
        }
        let events = feed(&mut segmenter, &stream);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, SegmenterEvent::Discarded)),
            "a segment below the quiet gate should be discarded, events: {}",
            events.len()
        );
    }

    /// The counterpart of the test above: with the production constant, a
    /// normal utterance still gets through.
    #[test]
    fn real_speech_survives_the_quiet_gate() {
        let mut config = VadConfig::testing(8);
        config.min_speech_hops = 2;
        config.silence_hops = 2;
        let mut segmenter = Segmenter::new(config);

        let mut stream = Vec::new();
        for _ in 0..10 {
            stream.extend_from_slice(&noise(8, 8, 0.30));
        }
        for _ in 0..4 {
            stream.extend_from_slice(&silent_hop(8));
        }
        let events = feed(&mut segmenter, &stream);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, SegmenterEvent::SegmentReady(_))),
            "normal speech was dropped by the quiet gate"
        );
    }

    /// The regression, reproduced end to end.
    ///
    /// A 42-minute meeting produced 60 segments whose spacing median was
    /// exactly 30.6s — i.e. every segment was the hard cap firing, because a
    /// steady noise floor above the old fixed 0.012 threshold meant the VAD
    /// never once saw silence. This simulates that shape: 2 minutes of
    /// constant background noise above the legacy threshold, with normal
    /// speech on top, and asserts the segmenter now cuts per utterance.
    #[test]
    fn noisy_meeting_is_segmented_per_utterance() {
        const SAMPLE_RATE: u32 = 48_000;
        let config = VadConfig::from_millis(SAMPLE_RATE);
        let mut segmenter = Segmenter::new(config);

        let floor_amplitude = 0.03f32; // ~0.017 RMS: above the legacy 0.012
        let speech_amplitude = 0.30f32;
        let mut seed = 0x5eed_1234u64;
        let mut next = move || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };

        let mut stream: Vec<f32> = Vec::new();
        let mut utterances = 0usize;
        // 3s of room noise before anyone speaks ...
        let lead_in = (3.0 * SAMPLE_RATE as f32) as usize;
        for _ in 0..lead_in {
            stream.push(next() * floor_amplitude);
        }
        // ... then 16 utterances of 2-6s separated by 2-5s gaps.
        for index in 0..16 {
            let voiced = 2.0 + (index % 5) as f32;
            let gap = 2.0 + ((index * 7) % 4) as f32;
            for phase in 0..(voiced * SAMPLE_RATE as f32) as usize {
                // Syllabic envelope with real inter-word dips, so the noise
                // floor estimator has quiet hops to lock onto.
                let t = phase as f32 / SAMPLE_RATE as f32;
                let envelope = if (t * 4.0) % 1.0 < 0.6 { 1.0 } else { 0.02 };
                stream.push(next() * (floor_amplitude + speech_amplitude * envelope));
            }
            for _ in 0..(gap * SAMPLE_RATE as f32) as usize {
                stream.push(next() * floor_amplitude);
            }
            utterances += 1;
        }

        let mut durations = Vec::new();
        for &sample in &stream {
            if let SegmenterEvent::SegmentReady(segment) = segmenter.push(sample) {
                durations.push(segment.samples.len() as f32 / SAMPLE_RATE as f32);
            }
        }

        assert!(
            !durations.is_empty(),
            "no segments at all — the segmenter is deaf"
        );
        let longest = durations.iter().cloned().fold(0.0f32, f32::max);
        let average = durations.iter().sum::<f32>() / durations.len() as f32;

        assert!(
            durations.len() >= utterances && durations.len() <= utterances + 2,
            "expected about one segment per utterance ({utterances}), got {}",
            durations.len()
        );
        assert!(
            longest <= 16.0,
            "a segment ran {longest:.1}s; the hard cap should have split it"
        );
        assert!(
            average <= 8.0,
            "average segment {average:.1}s is still the hard cap, not utterances"
        );
    }

    /// A long monologue must be split at a quiet point, not wherever the cap
    /// happens to fall.
    #[test]
    fn force_flush_cuts_at_the_energy_trough() {
        let mut config = VadConfig::testing(8);
        config.silence_hops = 1_000; // never end on silence: force the cap
        config.max_segment_samples = 8 * 20; // 20 hops
        config.trough_search_samples = 8 * 6; // look back 6 hops
        let mut segmenter = Segmenter::new(config);

        let loud = noise(5, 8, 0.4);
        let quiet = noise(6, 8, 0.05);
        let mut stream = Vec::new();
        // 18 loud hops then a deliberately quiet hop: the trough must land on
        // it rather than at hop 20.
        for _ in 0..18 {
            stream.extend_from_slice(&loud);
        }
        stream.extend_from_slice(&quiet);
        for _ in 0..6 {
            stream.extend_from_slice(&loud);
        }

        let events = feed(&mut segmenter, &stream);
        let mut cuts = Vec::new();
        for event in events {
            if let SegmenterEvent::SegmentReady(segment) = event {
                cuts.push(segment.samples.len());
            }
        }
        assert!(!cuts.is_empty(), "expected at least one forced flush");
        assert_eq!(
            cuts[0],
            8 * 19,
            "expected the split right after the quiet hop (152 samples), got {}",
            cuts[0]
        );
    }
}

