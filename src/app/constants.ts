export const CARD_SURFACE =
  "border border-white/[0.09] bg-[rgb(19_21_22_/_0.86)] shadow-[0_14px_36px_rgb(0_0_0_/_0.24)] backdrop-blur-[14px]";
export const ICON_BUTTON =
  "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border-0 bg-white/[0.06] text-[#f5f5f5] transition-[background,color,transform] duration-150 hover:bg-white/[0.11] active:scale-[0.98] [&_svg]:h-4 [&_svg]:w-4";
export const SESSION_BUTTON =
  "inline-flex h-9 shrink-0 items-center justify-center gap-1.5 rounded-lg border-0 bg-white/[0.06] px-2.5 text-[13px] font-medium text-[#f5f5f5] transition-[background,color,transform] duration-150 hover:bg-white/[0.11] active:scale-[0.98] [&_svg]:h-4 [&_svg]:w-4";
export const GHOST_ICON_BUTTON =
  "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border-0 bg-transparent text-white/52 transition-[background,color,transform] duration-150 hover:bg-white/[0.09] hover:text-white/80 active:scale-[0.98] [&_svg]:h-4 [&_svg]:w-4";
export const DRAG_CURSOR = "cursor-grab active:cursor-grabbing";

export const MIC_SEGMENT_MS = 4_000;
export const MIC_MIN_SEGMENT_MS = 1_200;
export const MIC_VAD_INTERVAL_MS = 100;
export const MIC_VAD_SILENCE_MS = 1_100;
export const MIC_VAD_RMS_THRESHOLD = 0.018;
export const FULL_SESSION_SEGMENT_LIMIT = 500;
export const AUTO_ASSIST_MIN_CONFIDENCE = 0.68;
export const AUTO_ASSIST_PREFETCH_CONFIDENCE = 0.88;
export const AUTO_ASSIST_HINT_TTL_MS = 16_000;
export const AUTO_ASSIST_HINT_COOLDOWN_MS = 10_000;
export const AUTO_ASSIST_DEDUPE_WINDOW_MS = 45_000;
export const AUTO_ASSIST_CACHE_TTL_MS = 30_000;
export const AUTO_ASSIST_PREFETCH_ENABLED = true;
// 推测性预取复用阈值：最终问题与预取问题的相似度达到该值才复用缓存答案。
export const SPECULATIVE_REUSE_THRESHOLD = 0.78;
// 极性翻转（否定/反义词替换）时把相似度压制到该值以下，避免复用语义相反的答案。
export const SPECULATIVE_FLIP_CAP = 0.55;
// 单次预取请求的兜底超时，略低于教练的 10s，避免在途 await 与教练重试叠加成超长等待。
export const SPECULATIVE_PREFETCH_TIMEOUT_MS = 8_000;
export const COACH_HEARTBEAT_MS = 10_000;
export const COACH_MAX_MESSAGES = 8;
