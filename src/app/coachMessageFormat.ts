import type { AssistantSuggestion } from "./types";

/// Label above the reserve lines. `bullets` are no longer body points — the
/// spoken answer is one paragraph that stands on its own — so they render as
/// an optional, visually quiet "if they push further" block rather than as
/// part of the answer.
const FOLLOW_UP_LABEL = "追问可补";
const DESIGN_POINTS_LABEL = "设计思路";

/// Kind value the backend sets when the question has no answer it can write
/// for the candidate (see `KIND_BEHAVIORAL` in `providers/llm/mod.rs`).
export const BEHAVIORAL_KIND = "behavioral";

/// Kind value for "the interviewer wants to see code" (`KIND_CODING`).
/// `answer` then holds one framing sentence plus a fenced code block, so the
/// spoken-length expectations do not apply and the body must be rendered
/// verbatim — indentation included.
export const CODING_KIND = "coding";

/// Kind value for an open-ended scenario whose main response is a visible
/// 3-5 point outline rather than a fake-complete spoken paragraph.
export const DESIGN_KIND = "design";

/// Shown above a behavioural answer instead of prose, because in that mode
/// `answer` is coaching, not something to read out. Without this the candidate
/// could mistake the advice for the answer and say "这题只能用你自己的真实经历"
/// out loud to the interviewer.
const BEHAVIORAL_HEADLINE = "这题要用你自己的经历";

const BEHAVIORAL_TIPS_LABEL = "怎么讲";

/**
 * Coloured pictographs, dingbats and flags.
 *
 * A ✅ or 🔴 on screen during an interview is an instant tell: nothing in a
 * normal notes app or IDE emits them mid-sentence, and they survive a glance at
 * a shared screen far better than text does. The model is told not to produce
 * them, but a prompt is a request, not a guarantee — so everything is stripped
 * on the way to the card as well.
 *
 * Deliberately NOT stripped: CJK punctuation, the `·` bullet we render
 * ourselves, arrows and math symbols. Those are legitimate in a technical
 * answer ("O(1) → O(n)") and render as plain glyphs, not colour.
 */
const PICTOGRAPH_CLASS =
  "[\\u{1F000}-\\u{1FAFF}\\u{2190}-\\u{21FF}\\u{2300}-\\u{23FF}\\u{2460}-\\u{24FF}\\u{25A0}-\\u{27BF}\\u{2B00}-\\u{2BFF}\\u{FE0F}\\u{20E3}\\u{E0020}-\\u{E007F}]";

/** Non-global on purpose: `.test()` on a /g regex advances `lastIndex` and the
 * next call then starts mid-string, which makes detection silently flaky. */
const HAS_PICTOGRAPH_RE = new RegExp(PICTOGRAPH_CLASS, "u");

/** An icon run plus the spacing that hugged it, so both go in one pass.
 * Spaces BETWEEN icons are part of the run ("搞定 🎉 👍"), otherwise the run
 * splits in two and the trailing one no longer looks like it is at the line
 * end — leaving a stray space behind. */
const PICTOGRAPH_RUN_RE = new RegExp(
  `[ \\t]*(?:${PICTOGRAPH_CLASS})(?:[ \\t]*(?:${PICTOGRAPH_CLASS}))*[ \\t]*`,
  "gu"
);

/** Keep these: they read as text, not as coloured icons. */
const ALLOWED = new Set(["→", "←", "↑", "↓", "·", "≈", "≤", "≥", "≠", "∈", "∑", "√", "∞"]);

/**
 * Remove anything that would render as a coloured icon on the coach card.
 *
 * Runs on every string that reaches the UI.
 *
 * Whitespace is only cleaned up AT THE POSITION an icon was removed from —
 * never globally. An earlier version collapsed runs of spaces and trimmed every
 * line, which silently destroyed code indentation: a Python answer came back
 * with every line flush-left and therefore unrunnable. Leading whitespace is
 * meaningful content here, not formatting noise.
 */
export function stripPictographs(text: string): string {
  if (!text) return "";
  // Fast path, and more importantly a guarantee: text with no icons is returned
  // byte-for-byte, so indentation and alignment can never be collateral damage.
  if (!HAS_PICTOGRAPH_RE.test(text)) return text;

  return (
    text
      // Consume the spaces that sat immediately around the icon, so
      // "方案 ✅ 可行" collapses to "方案 可行" rather than "方案  可行".
      .replace(PICTOGRAPH_RUN_RE, (match, offset: number, whole: string) => {
        if (Array.from(match).some((ch) => ALLOWED.has(ch))) return match;
        const atLineStart = offset === 0 || whole[offset - 1] === "\n";
        const end = offset + match.length;
        const atLineEnd = end === whole.length || whole[end] === "\n";
        // Only keep a separator when the icon actually sat between two things
        // on the same line ("方案 ✅ 可行"). At either edge the space was
        // padding and would render as a stray gap.
        if (atLineStart || atLineEnd) return "";
        return /^[ \t]/.test(match) && /[ \t]$/.test(match) ? " " : "";
      })
      // CJK punctuation carries its own spacing, so a space left behind by a
      // removed icon reads as a typo ("不行， 有风险").
      .replace(/([，。！？；：、）】」』])[ \t]+(?=\S)/g, "$1")
      // A line left holding nothing but whitespace had only an icon on it.
      .replace(/^[ \t]+$/gm, "")
  );
}

/**
 * True when the suggestion is coaching the candidate rather than answering.
 *
 * Anything unrecognized falls through to "answer", matching the backend: a
 * missing or unexpected `kind` must never turn a real answer into advice,
 * because withholding content the candidate needs is the worse failure.
 */
export function isBehavioralSuggestion(suggestion: AssistantSuggestion): boolean {
  return suggestion.kind === BEHAVIORAL_KIND;
}

export function isDesignSuggestion(suggestion: AssistantSuggestion): boolean {
  return suggestion.kind === DESIGN_KIND;
}

/**
 * Flattens the structured suggestion into the coach card's plain-text body.
 *
 * Three shapes, decided by `kind`:
 * - knowledge: `answer` is the whole spoken answer, read aloud as-is;
 *   `bullets` hang below a labelled separator as reserve lines for 追问.
 * - design: `answer` is a one-line framing and `bullets` are the primary
 *   3-5 point outline, labelled as 设计思路 rather than follow-up material.
 * - behavioral: the answer depends on the candidate's own history, so
 *   `answer` is second-person coaching and `bullets` are delivery tips. It
 *   gets a headline so the card never reads as a script to recite.
 */
export function formatSuggestionText(suggestion: AssistantSuggestion): string {
  const bullets = suggestion.bullets
    .map((bullet) => stripPictographs(bullet).trim())
    .filter(Boolean);
  const answer = stripPictographs(suggestion.answer).trim();
  const lines: string[] = [];

  if (isBehavioralSuggestion(suggestion)) {
    lines.push(BEHAVIORAL_HEADLINE);
    lines.push(answer);
    if (bullets.length > 0) {
      lines.push("");
      lines.push(BEHAVIORAL_TIPS_LABEL);
    }
  } else if (isDesignSuggestion(suggestion)) {
    lines.push(answer);
    if (bullets.length > 0) {
      lines.push("");
      lines.push(DESIGN_POINTS_LABEL);
    }
  } else {
    lines.push(answer);
    if (bullets.length > 0) {
      lines.push("");
      lines.push(FOLLOW_UP_LABEL);
    }
  }

  for (const bullet of bullets) {
    lines.push(`· ${bullet}`);
  }
  return lines.join("\n").trim();
}
