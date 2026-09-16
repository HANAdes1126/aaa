//! Guard against degenerate ASR output.
//!
//! Qwen3-ASR is greedy-decoded (`temperature: 0`) and, on audio it cannot make
//! sense of — a long slab of room tone, clipped system audio, music — it falls
//! into a decode loop: the same few characters are emitted until `max_tokens`
//! is exhausted. Observed live: 41 of 60 segments in one meeting were >69%
//! repeated characters, many exactly 253 chars long (the 256-token cap).
//!
//! Those segments then leak into the transcript and into the coach prompt, so
//! a bad audio chunk turns into "I couldn't hear the question" answers. This
//! module measures how much of a transcript is one periodic run and lets the
//! caller drop the segment instead of publishing it.

/// How much of the text one repeating unit covers, and where it starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepetitionSpan {
    /// Fraction of significant characters covered by the repeating run (0..1).
    pub ratio: f32,
    /// Index (in significant characters) where the run starts.
    pub start: usize,
    /// Period of the run, in significant characters.
    pub period: usize,
    /// Number of significant characters covered.
    pub covered: usize,
}

impl Default for RepetitionSpan {
    fn default() -> Self {
        Self {
            ratio: 0.0,
            start: 0,
            period: 0,
            covered: 0,
        }
    }
}

/// Whitespace and punctuation are stripped before analysis: the model repeats
/// them along with the words, and leaving them in would let ", " style jitter
/// mask an otherwise perfect loop.
fn is_significant(character: char) -> bool {
    if character.is_whitespace() {
        return false;
    }
    !matches!(
        character,
        '，' | '。'
            | '！'
            | '？'
            | '、'
            | '；'
            | '：'
            | ','
            | '.'
            | '!'
            | '?'
            | ';'
            | ':'
            | '"'
            | '\''
            | '“'
            | '”'
            | '《'
            | '》'
            | '（'
            | '）'
            | '('
            | ')'
            | '【'
            | '】'
            | '['
            | ']'
            | '…'
            | '—'
            | '-'
            | '~'
            | '～'
    )
}

/// Finds the longest periodic run anywhere in the text.
///
/// A sequence is periodic with period `p` wherever `s[i] == s[i + p]`; the
/// longest unbroken stretch of such matches, plus the period itself, is the
/// part of the text produced by looping. Scanning every period (rather than
/// only periods that start at index 0) is what catches the common shape
/// "a plausible opening clause, then a loop".
pub fn analyse(text: &str) -> RepetitionSpan {
    let chars: Vec<char> = text.chars().collect();
    let sequence: Vec<char> = chars
        .iter()
        .copied()
        .filter(|&character| is_significant(character))
        .collect();
    let length = sequence.len();

    // Too short to say anything: a single repeated word is not a decode loop.
    if length < 8 {
        return RepetitionSpan::default();
    }

    let max_period = (length / 3).min(64);
    let mut best = RepetitionSpan::default();

    for period in 1..=max_period {
        let mut best_len = 0usize;
        let mut best_start = 0usize;
        let mut run_start: Option<usize> = None;

        for index in 0..length.saturating_sub(period) {
            if sequence[index] == sequence[index + period] {
                run_start.get_or_insert(index);
            } else if let Some(start) = run_start.take() {
                let run_length = index - start + 1;
                if run_length > best_len {
                    best_len = run_length;
                    best_start = start;
                }
            }
        }
        if let Some(start) = run_start {
            let run_length = (length - period) - start;
            if run_length > best_len {
                best_len = run_length;
                best_start = start;
            }
        }

        if best_len == 0 {
            continue;
        }

        let covered = best_len + period;
        // `covered` ties are common ("哈哈…" is equally well described by
        // period 1, 2, 4 …). Keep the first, i.e. the smallest period: that is
        // the primitive repeating unit, which is the useful description.
        if covered > best.covered {
            best = RepetitionSpan {
                ratio: covered as f32 / length as f32,
                start: best_start,
                period,
                covered,
            };
        }
    }

    best
}

/// Fraction of the transcript covered by a single repeating unit.
pub fn repetition_ratio(text: &str) -> f32 {
    analyse(text).ratio
}

/// True when the transcript is a decode loop rather than speech.
///
/// Two thresholds: an obvious loop (most of the text repeats), and a long
/// transcript that is substantially repetitive — a near-`max_tokens` answer
/// that is 40% loop is already unusable, and it is the shape produced when the
/// model runs the token cap out.
pub fn is_degenerate(text: &str) -> bool {
    let span = analyse(text);
    let significant = text
        .chars()
        .filter(|&character| is_significant(character))
        .count();

    if significant < 12 {
        return false;
    }

    span.ratio >= 0.6 || (significant >= 150 && span.ratio >= 0.4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_short_loop() {
        let looped = "哎呦喂！我天哪！".repeat(12);
        assert!(is_degenerate(&looped));
        assert!(repetition_ratio(&looped) > 0.9);
    }

    #[test]
    fn detects_a_whole_sentence_loop() {
        let looped = "请你做一个自我介绍".repeat(8);
        assert!(is_degenerate(&looped));
    }

    #[test]
    fn detects_a_single_character_loop() {
        let looped = "啊".repeat(80);
        assert!(is_degenerate(&looped));
    }

    #[test]
    fn ignores_whitespace_and_punctuation_jitter() {
        // Same words, punctuation alternating: still one loop.
        let looped = "好的，好的.好的，好的.".repeat(6);
        assert!(is_degenerate(&looped));
    }

    #[test]
    fn catches_a_loop_whose_first_unit_has_an_extra_character() {
        // Observed in a real meeting: the opening repeat carries a trailing
        // "了", so the runs are 6/5/5/5/5 characters and a period scan that
        // demanded exact alignment would miss the loop.
        let looped = "嗯嗯嗯，昨天去哪里了？昨天去哪里？昨天去哪里？昨天去哪里？昨天去哪里？";
        assert!(
            is_degenerate(looped),
            "ratio was {:.3}",
            repetition_ratio(looped)
        );
    }

    #[test]
    fn keeps_normal_interview_speech() {
        let normal = "我们先聊一下项目吧，你在 qzone 后端主要负责哪一块？"
            .to_string()
            + "我主要负责相册写入链路，也就是 photo 和 media data writer 这两个服务，"
            + "日常做得比较多的是错误码映射和限流组件的接入。";
        assert!(!is_degenerate(&normal));
    }

    #[test]
    fn keeps_short_answers_with_repeated_words() {
        // A genuinely short utterance with a stutter must survive.
        assert!(!is_degenerate("这个这个，我觉得可以。"));
        assert!(!is_degenerate("对对对"));
    }

    #[test]
    fn treats_very_short_text_as_clean() {
        assert_eq!(analyse("你好").ratio, 0.0);
        assert!(!is_degenerate("你好"));
        assert!(!is_degenerate(""));
    }

    #[test]
    fn reports_where_the_loop_starts() {
        // Eight characters of plausible speech, then a three-character unit
        // looped. The unit must be reported as-is (period 3) and the loop must
        // be located after the opening clause.
        let looped = "请介绍一下你自己".to_string() + &"不知道".repeat(30);
        let span = analyse(&looped);
        assert!(span.ratio > 0.6, "ratio was {}", span.ratio);
        assert_eq!(span.period, 3);
        assert_eq!(span.start, 8);
    }

    #[test]
    fn clean_text_is_never_touched() {
        // The guard must not rewrite normal transcripts, only flag them.
        assert!(!is_degenerate("你好，请问有什么可以帮您？"));
        assert!(is_degenerate(&"啊".repeat(80)));
    }

    #[test]
    fn long_partially_looped_transcript_is_degenerate() {
        // 150+ significant chars, ~40%+ loop: a model that ran the cap out.
        let looped = "面试官你好".to_string() + &"我不知道啊".repeat(30);
        assert!(is_degenerate(&looped));
    }
}
