mod openai_compatible;

use crate::providers::config::{DiagnosticResult, ProviderId, ProviderKind};
use crate::providers::credentials;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tauri::AppHandle;

pub(crate) use openai_compatible::parse_suggestion;
pub use openai_compatible::OpenAiCompatibleLlm;

/// Endpoints that have already rejected `stream: false`.
///
/// Some OpenAI-compatible gateways (Tencent Copilot among them) answer every
/// non-streaming request with error 11101. Probing anyway costs a full round
/// trip: measured 2026-09-12 across 215 requests, the probe failed 215 times
/// — a 100% loss rate — at 1012ms average, 218s burned in total.
///
/// Remembering the verdict makes that a one-time cost per endpoint per app
/// run. Shared between the LLM adapter and the agent tool loop so the two
/// paths do not each pay their own probe.
///
/// Keyed by `base_url`, never by a hardcoded hostname: the moment this becomes
/// `if url.contains("copilot")` it stops being a compatibility mechanism and
/// becomes a special case that rots when the gateway is renamed or self-hosted.
///
/// Intentionally in-memory only — a provider that gains non-streaming support
/// recovers on the next launch with nobody having to clear a cache file.
static NON_STREAM_REJECTED: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashSet<String>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashSet::new()));

/// True when this endpoint already told us it cannot do non-streaming, so the
/// caller should go straight to SSE.
pub(crate) fn non_stream_known_unsupported(base_url: &str) -> bool {
    NON_STREAM_REJECTED
        .read()
        .map(|set| set.contains(base_url))
        .unwrap_or(false)
}

/// Records an 11101-style rejection so later requests skip the probe.
pub(crate) fn remember_non_stream_rejected(base_url: &str) {
    if let Ok(mut set) = NON_STREAM_REJECTED.write() {
        set.insert(base_url.to_string());
    }
}

/// Timeout budget for one LLM round trip.
///
/// Without these, a gateway that accepts the connection and then stalls hangs
/// the assistant forever: `reqwest` has no default timeout, and the caller
/// (coach, manual ask, vision) waits on the future indefinitely. A stalled
/// coach is worse than a failed one — the UI stays "thinking" and every later
/// question is dropped as already-in-flight, so the whole session looks dead.
///
/// Three layers, because they catch different failures:
/// - `connect`: TCP/TLS never completes (bad network, dead endpoint).
/// - `first_byte`: accepted but the model never starts emitting (queue stall).
///   This is the common "hangs for minutes" case.
/// - `idle`: started streaming then stopped mid-answer (dropped SSE).
///
/// There is deliberately no overall wall-clock cap on the stream read: a long
/// but healthy answer must not be cut off. `first_byte` + `idle` already bound
/// a stalled stream, and the caller (see `agent_tool_loop`) applies the total
/// budget per workflow.
///
/// The idle budget is sized for reasoning models, not for flash ones. Measured
/// 2026-09-12 on a screenshot coding question: claude-opus-5 sent its first
/// byte in 5-6s and then went quiet while thinking, so a 10s idle window killed
/// three runs in a row (`stream stalled saw_first_byte=true`) even though the
/// model was working normally. deepseek-v4-flash on the same workload already
/// needed 18.8s end to end. A reasoning pause is not a stall, so the window has
/// to clear the longest *gap between tokens*, not the total answer time.
///
/// `first_byte` is generous for the same reason, plus image upload: on a 523KB
/// screenshot claude-opus-5 took 48s to emit its first byte. Text-only calls
/// never come close, so the loose bound costs nothing in the common case — it
/// only decides whether a deliberately chosen strong model works at all.
pub const LLM_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
pub const LLM_FIRST_BYTE_TIMEOUT: Duration = Duration::from_secs(90);
pub const LLM_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

/// Wall-clock cap for one non-streaming completion. Streaming answers are
/// bounded by `LLM_FIRST_BYTE_TIMEOUT` + `LLM_STREAM_IDLE_TIMEOUT` instead,
/// so a long-but-healthy answer is never truncated mid-sentence.
pub const LLM_NON_STREAM_TOTAL_TIMEOUT: Duration = Duration::from_secs(90);

/// Builds the HTTP client used for every LLM request.
///
/// Shared so no call site can accidentally get an unbounded client again.
/// Only the connect phase is capped here — capping the whole request would
/// kill legitimately long streaming answers.
pub fn llm_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(LLM_CONNECT_TIMEOUT)
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Structured suggestion returned by the assistant. Matches the schema in
/// docs/TECHNICAL_DESIGN.md section 4.6, minus the `risk` field (dropped;
/// see openspec/changes/add-llm-suggestions/design.md). `camelCase` on the
/// wire in both directions: it's what the LLM is instructed to return (see
/// `prompt_orchestrator::JSON_OUTPUT_CONTRACT`) and what gets emitted to the
/// frontend as the `assistant_done` event payload, matching every other
/// Tauri DTO in this project.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantSuggestion {
    pub answer: String,
    #[serde(default)]
    pub bullets: Vec<String>,
    #[serde(default)]
    pub clarifying_question: Option<String>,
    /// `"knowledge"`, `"design"`, `"coding"`, or `"behavioral"`, as
    /// classified by the model and normalized by `parse_suggestion`. Plain
    /// `String`, not an enum, on
    /// purpose: an unexpected value from the model must never fail
    /// deserialization, because that falls back to treating the whole reply as
    /// prose and silently loses a perfectly good structured answer.
    ///
    /// `"behavioral"` means the answer depends on the candidate's own history,
    /// so `answer` holds coaching instead of a spoken answer — see
    /// `app/prompt_orchestrator.rs`. Consumers that don't care about the
    /// distinction can ignore it: an empty value behaves as `"knowledge"`.
    #[serde(default)]
    pub kind: String,
}

/// The kind value meaning "this question has no answer I can write for you".
pub const KIND_BEHAVIORAL: &str = "behavioral";

/// The kind value meaning "here is the standard answer, read it out".
pub const KIND_KNOWLEDGE: &str = "knowledge";

/// The kind value meaning "show a compact design outline, not a fake-complete
/// paragraph". `answer` is the one-line framing and `bullets` are the 3-5 main
/// design dimensions the candidate can expand aloud.
pub const KIND_DESIGN: &str = "design";

/// The kind value meaning "the interviewer wants code on the screen".
///
/// Split out from `knowledge` because the two need opposite output shapes: a
/// knowledge answer is one spoken paragraph with no code block, while a coding
/// answer is mostly a code block and must not be squeezed into 80-100
/// characters. Measured 2026-09-12, leaving this as an exception buried in the
/// knowledge rules lost the fight against the length limit — deepseek-v4-pro
/// literally answered "下面用 Java 实现" and then stopped. Classification first
/// makes the shape a consequence of the label instead of a rule to remember.
pub const KIND_CODING: &str = "coding";

/// Maps whatever the model emitted onto the kinds we understand.
///
/// Unrecognized or missing input resolves to `"knowledge"`: withholding an
/// answer the candidate needs is a worse failure than answering a question
/// that should have been coached instead.
pub fn normalize_kind(raw: Option<&str>) -> String {
    let lower = raw.unwrap_or("").trim().to_ascii_lowercase();
    if lower.contains("behav")
        || lower.contains("行为")
        || lower.contains("经历")
        || lower.contains("因人而异")
    {
        return KIND_BEHAVIORAL.to_string();
    }
    // Checked after behavioural on purpose: "讲一次你写代码解决问题的经历" is a
    // story, not a coding task, and the behavioural markers are the stronger
    // signal there.
    if lower.contains("cod") || lower.contains("代码") || lower.contains("手撕") {
        return KIND_CODING.to_string();
    }
    if lower.contains("design")
        || lower.contains("设计")
        || lower.contains("架构")
        || lower.contains("方案")
    {
        return KIND_DESIGN.to_string();
    }
    KIND_KNOWLEDGE.to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingControl {
    Unsupported,
    ProviderSpecific,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmCapabilities {
    pub supports_streaming: bool,
    pub thinking_control: ThinkingControl,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// Adapter interface for an LLM provider. One non-streaming call in, one
/// complete structured suggestion out (see
/// openspec/changes/add-llm-suggestions/design.md for why this is
/// non-streaming).
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    fn id(&self) -> ProviderId;
    fn capabilities(&self) -> LlmCapabilities;

    async fn complete_messages(
        &self,
        system_prompt: String,
        messages: Vec<ChatMessage>,
    ) -> Result<AssistantSuggestion>;

    async fn complete_text(
        &self,
        system_prompt: String,
        user_message: String,
        temperature: f32,
        disable_reasoning: bool,
    ) -> Result<String>;

    async fn complete(
        &self,
        system_prompt: String,
        user_message: String,
    ) -> Result<AssistantSuggestion> {
        self.complete_messages(system_prompt, vec![ChatMessage::user(user_message)])
            .await
    }
}

/// Builds an `LlmProvider` from the currently saved config and Keychain API
/// key. Returns an error if no key has been saved yet.
pub fn build_from_saved_config(app: &AppHandle) -> Result<Box<dyn LlmProvider>> {
    let credentials = credentials::resolve(app, ProviderKind::Llm)?;
    match credentials.provider_id {
        ProviderId::OpenAiCompatible => Ok(Box::new(OpenAiCompatibleLlm::new(credentials))),
        provider_id => Err(anyhow!(
            "Provider {} is not registered for LLM.",
            provider_id.as_str()
        )),
    }
}

/// Sends a minimal chat completion request ("respond with OK") to the
/// configured LLM endpoint and reports whether it succeeded. Used by the
/// Settings page "Test connection" button.
pub async fn test_connection(app: &AppHandle) -> DiagnosticResult {
    let provider = match build_from_saved_config(app) {
        Ok(provider) => provider,
        Err(error) => {
            return DiagnosticResult {
                success: false,
                message: error.to_string(),
            }
        }
    };

    let probe_prompt =
        "Respond with a JSON object exactly like {\"answer\": \"OK\", \"bullets\": [], \"clarifyingQuestion\": null} and nothing else.";

    match provider
        .complete(probe_prompt.to_string(), "ping".to_string())
        .await
    {
        Ok(_) => {
            let capabilities = provider.capabilities();
            DiagnosticResult {
                success: true,
                message: format!(
                    "{} 模型接口可访问（流式={}，思维链={:?}）。",
                    provider.id().as_str(),
                    capabilities.supports_streaming,
                    capabilities.thinking_control
                ),
            }
        }
        Err(error) => DiagnosticResult {
            success: false,
            message: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{non_stream_known_unsupported, remember_non_stream_rejected};

    #[test]
    fn non_stream_verdict_is_remembered_per_endpoint() {
        let copilot = "https://copilot.tencent.com/v2/chat/completions";
        let other = "https://api.siliconflow.cn/v1/chat/completions";

        // Unknown endpoints must still be probed: assuming they need streaming
        // would break every provider that only supports non-streaming.
        assert!(!non_stream_known_unsupported(copilot));
        assert!(!non_stream_known_unsupported(other));

        remember_non_stream_rejected(copilot);

        assert!(non_stream_known_unsupported(copilot));
        // One gateway's incompatibility says nothing about another's.
        assert!(!non_stream_known_unsupported(other));
    }
}
