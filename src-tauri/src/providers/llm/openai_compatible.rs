use super::{
    llm_http_client, normalize_kind, AssistantSuggestion, ChatMessage,
    LlmCapabilities, LlmProvider, ThinkingControl, LLM_NON_STREAM_TOTAL_TIMEOUT,
    LLM_STREAM_IDLE_TIMEOUT, KIND_KNOWLEDGE,
};
use crate::providers::config::ProviderId;
use crate::providers::credentials::ResolvedCredentials;
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::StatusCode;
use serde_json::{json, Value};

/// LLM adapter for any endpoint that accepts the OpenAI
/// `chat/completions` request shape. SiliconFlow, OpenAI, DeepSeek, and most
/// domestic OpenAI-compatible providers implement this shape; switching
/// providers only requires changing base_url/model/api_key in Settings.
pub struct OpenAiCompatibleLlm {
    base_url: String,
    model: String,
    api_key: String,
    client: reqwest::Client,
}

impl OpenAiCompatibleLlm {
    pub fn new(credentials: ResolvedCredentials) -> Self {
        Self {
            base_url: credentials.base_url,
            model: credentials.model,
            api_key: credentials.api_key,
            client: llm_http_client(),
        }
    }

    async fn complete_text_request(
        &self,
        system_prompt: String,
        user_message: String,
        temperature: f32,
        disable_reasoning: bool,
    ) -> Result<String> {
        self.complete_raw(
            system_prompt,
            vec![ChatMessage::user(user_message)],
            temperature,
            disable_reasoning,
        )
        .await
    }

    async fn complete_raw(
        &self,
        system_prompt: String,
        messages: Vec<ChatMessage>,
        temperature: f32,
        disable_reasoning: bool,
    ) -> Result<String> {
        let mut request_messages = vec![json!({
            "role": "system",
            "content": system_prompt,
        })];
        request_messages.extend(
            messages
                .into_iter()
                .map(|message| serde_json::to_value(message))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        );
        let mut body = json!({
            "model": self.model,
            "messages": request_messages,
            "temperature": temperature,
            "stream": false,
        });
        if disable_reasoning && self.base_url.contains("siliconflow") {
            body["enable_thinking"] = json!(false);
        }

        // Skip the probe entirely once this endpoint has rejected it: the
        // round trip is pure loss (Copilot has never accepted one) and it
        // lands on the critical path of every coach answer.
        if super::non_stream_known_unsupported(&self.base_url) {
            body["stream"] = json!(true);
            return self.complete_streaming(&body).await;
        }

        // Cap the wait here rather than on the client: a non-streaming reply
        // arrives in one piece, so any longer wait is the gateway stalling.
        let response = tokio::time::timeout(LLM_NON_STREAM_TOTAL_TIMEOUT, self.send_request(&body))
            .await
            .map_err(|_| {
                anyhow!(
                    "LLM request timed out after {}s.",
                    LLM_NON_STREAM_TOTAL_TIMEOUT.as_secs()
                )
            })??;
        let status = response.status();
        if status.is_success() {
            return parse_non_stream_response(response).await;
        }

        let error_body = response.text().await.unwrap_or_default();
        // Tencent Copilot accepts the OpenAI Chat Completions request shape, but
        // explicitly rejects `stream: false` with code 11101. Keep non-streaming
        // as the default and retry only that documented incompatibility once.
        if requires_streaming_retry(status, &error_body) {
            let _ = crate::debug_log::append(
                "[llm] non-stream request rejected with 11101; retrying stream (skipping the probe from now on)",
            );
            // Pay this round trip once per endpoint, not once per request.
            super::remember_non_stream_rejected(&self.base_url);
            body["stream"] = json!(true);
            return self.complete_streaming(&body).await;
        }

        let _ = crate::debug_log::append(&format!(
            "[llm] non-stream request rejected status={status}; no streaming retry",
        ));
        Err(anyhow!("LLM request failed: {status} {error_body}"))
    }

    async fn send_request(&self, body: &Value) -> Result<reqwest::Response> {
        Ok(self
            .client
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
            .await?)
    }

    async fn complete_streaming(&self, body: &Value) -> Result<String> {
        let response = self.send_request(body).await?;
        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            let _ = crate::debug_log::append(&format!(
                "[llm] streaming retry rejected status={status}",
            ));
            return Err(anyhow!("LLM streaming retry failed: {status} {error_body}"));
        }

        let _ = crate::debug_log::append("[llm] streaming retry accepted");
        let mut bytes = response.bytes_stream();
        let mut payload = String::new();
        loop {
            // Each wait is individually capped: an SSE stream that opens and
            // then goes silent looks exactly like a slow model otherwise.
            match tokio::time::timeout(LLM_STREAM_IDLE_TIMEOUT, bytes.next()).await {
                Ok(Some(chunk)) => {
                    let chunk = chunk.context("Failed to read the LLM streaming response")?;
                    payload.push_str(&String::from_utf8_lossy(&chunk));
                }
                Ok(None) => break,
                Err(_) => {
                    return Err(anyhow!(
                        "LLM 响应流中断：等待超过 {} 秒没有新数据。",
                        LLM_STREAM_IDLE_TIMEOUT.as_secs()
                    ));
                }
            }
        }
        parse_sse_content(&payload)
    }
}

async fn parse_non_stream_response(response: reqwest::Response) -> Result<String> {
    let json: Value = response.json().await?;
    let content = json
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("LLM response missing choices[0].message.content"))?;
    Ok(content.trim().to_string())
}

fn requires_streaming_retry(status: StatusCode, body: &str) -> bool {
    if status != StatusCode::BAD_REQUEST {
        return false;
    }

    let code_is_11101 = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            let error = value.get("error").unwrap_or(&value);
            error.get("code").and_then(|code| {
                code.as_i64()
                    .or_else(|| code.as_str().and_then(|value| value.parse::<i64>().ok()))
            })
        })
        == Some(11101);
    code_is_11101 || body.contains("Non-stream chat request is currently not supported")
}

/// Joins OpenAI-compatible server-sent events after the provider has finished
/// streaming. The UI consumes one final answer, so provider-specific streaming
/// transport stays behind the existing LLM interface.
fn parse_sse_content(payload: &str) -> Result<String> {
    let mut content = String::new();
    for line in payload.lines() {
        let Some(data) = line.trim_start().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let event: Value = serde_json::from_str(data)
            .context("LLM streaming response contained an invalid SSE event")?;
        if let Some(delta) = event
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("delta"))
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str)
        {
            content.push_str(delta);
        }
    }

    if content.trim().is_empty() {
        return Err(anyhow!("LLM streaming response contained no choices[0].delta.content"));
    }
    Ok(content.trim().to_string())
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiCompatibleLlm {
    fn id(&self) -> ProviderId {
        ProviderId::OpenAiCompatible
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities {
            supports_streaming: true,
            thinking_control: if self.base_url.contains("siliconflow") {
                ThinkingControl::ProviderSpecific
            } else {
                ThinkingControl::Unsupported
            },
        }
    }

    async fn complete_messages(
        &self,
        system_prompt: String,
        messages: Vec<ChatMessage>,
    ) -> Result<AssistantSuggestion> {
        let content = self
            .complete_raw(system_prompt, messages, 0.3, false)
            .await?;
        Ok(parse_suggestion(&content))
    }

    async fn complete_text(
        &self,
        system_prompt: String,
        user_message: String,
        temperature: f32,
        disable_reasoning: bool,
    ) -> Result<String> {
        self.complete_text_request(system_prompt, user_message, temperature, disable_reasoning)
            .await
    }
}

/// Defensive ceiling on `bullets` across ALL modes — the widest value any
/// output contract promises. Interview design answers use 3-5 primary outline
/// points; knowledge/behavioral reserve lines and voice/general modes use fewer.
///
/// This must be the MAXIMUM across contracts, not the minimum: a ceiling below
/// a contract's promise silently drops that mode's last item mid-answer, which
/// reads as "the answer got cut off". How many bullets a given mode should
/// actually carry is the prompt's job, not this constant's — on the normal
/// path the truncate never fires.
pub(crate) const MAX_BULLETS: usize = 5;

/// Parses the LLM's message content into a suggestion. The request never sets
/// `response_format: json_object` (several OpenAI-compatible gateways reject
/// it), so providers are free to reply in plain prose. Prose is taken whole:
/// the spoken answer IS one paragraph, so there is nothing to rebuild.
pub(crate) fn parse_suggestion(content: &str) -> AssistantSuggestion {
    let normalized = strip_json_code_fence(content);
    match serde_json::from_str::<AssistantSuggestion>(normalized) {
        Ok(mut suggestion) => {
            suggestion.bullets.truncate(MAX_BULLETS);
            // The model may omit `kind` (older contract, other providers, or
            // just drift); normalize here so downstream never sees an empty
            // value and never has to compare against raw model output.
            suggestion.kind = normalize_kind(Some(&suggestion.kind));
            suggestion
        }
        Err(_) => prose_as_suggestion(content),
    }
}

/// Keeps a prose reply as one spoken paragraph.
///
/// Deliberately does NOT split prose into a conclusion plus bullet points.
/// That used to be the fallback, and it manufactured the exact failure the
/// output contract now forbids: a list is a document shape, so it read aloud
/// flat, and the synthesised conclusion merely restated the points below it.
/// Cluely reaches the same verdict from the other direction — its
/// `compressTechnicalConcept` post-pass flattens doc-shape back into prose and
/// never truncates. We do not trim either: length is the prompt's job.
fn prose_as_suggestion(text: &str) -> AssistantSuggestion {
    AssistantSuggestion {
        answer: text.trim().to_string(),
        bullets: Vec::new(),
        clarifying_question: None,
        // A reply we couldn't parse is treated as an answer, never as
        // coaching: coaching withholds content, and withholding on top of a
        // parse failure would show the candidate an empty card.
        kind: KIND_KNOWLEDGE.to_string(),
    }
}

fn strip_json_code_fence(content: &str) -> &str {
    let trimmed = content.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };

    let rest = rest
        .strip_prefix("json")
        .or_else(|| rest.strip_prefix("JSON"))
        .unwrap_or(rest)
        .trim_start();

    rest.strip_suffix("```").unwrap_or(rest).trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_suggestion_normalizes_behavioral_kind() {
        let suggestion = parse_suggestion(
            r#"{"kind":"behavioral","answer":"这题只能用你自己的真实经历。","bullets":[],"clarifyingQuestion":null}"#,
        );
        assert_eq!(suggestion.kind, "behavioral");
    }

    #[test]
    fn parse_suggestion_defaults_missing_kind_to_knowledge() {
        // Older contracts and other providers omit `kind`. Treating that as
        // "answer" is deliberate: withholding content the candidate needs is
        // worse than answering a question that should have been coached.
        let suggestion = parse_suggestion(r#"{"answer":"HashMap 底层是数组加链表加红黑树。","bullets":[]}"#);
        assert_eq!(suggestion.kind, "knowledge");
    }

    #[test]
    fn parse_suggestion_survives_unknown_kind_value() {
        // Must not fall through to the prose path: a weird kind value is a
        // classification hiccup, not a reason to lose the answer.
        let suggestion = parse_suggestion(
            r#"{"kind":" behavioural ","answer":"挑一个你真的做过的项目。","bullets":[]}"#,
        );
        assert_eq!(suggestion.kind, "behavioral");
        assert_eq!(suggestion.answer, "挑一个你真的做过的项目。");
    }

    #[test]
    fn prose_fallback_is_never_treated_as_coaching() {
        // Coaching withholds content, so a parse failure must stay an answer.
        assert_eq!(prose_as_suggestion("随便一段散文。").kind, "knowledge");
    }

    #[test]
    fn parse_suggestion_keeps_coding_kind_and_indentation() {
        // The code block is the answer here, so neither the kind nor the
        // leading whitespace may be normalised away — flush-left code does not
        // run, and the candidate pastes this straight into an editor.
        let suggestion = parse_suggestion(
            "{\"kind\":\"coding\",\"answer\":\"分治，平均 O(n log n)。\\n```java\\npublic void sort(int[] a) {\\n    if (a == null) {\\n        return;\\n    }\\n}\\n```\",\"bullets\":[]}",
        );
        assert_eq!(suggestion.kind, "coding");
        assert!(suggestion.answer.contains("\n    if (a == null) {"));
        assert!(suggestion.answer.contains("\n        return;"));
    }

    #[test]
    fn normalize_kind_maps_coding_synonyms() {
        assert_eq!(normalize_kind(Some("coding")), "coding");
        assert_eq!(normalize_kind(Some(" Coding ")), "coding");
        assert_eq!(normalize_kind(Some("代码题")), "coding");
        assert_eq!(normalize_kind(Some("手撕代码")), "coding");
    }

    #[test]
    fn normalize_kind_maps_design_synonyms() {
        assert_eq!(normalize_kind(Some("design")), "design");
        assert_eq!(normalize_kind(Some("system_design")), "design");
        assert_eq!(normalize_kind(Some("系统设计题")), "design");
        assert_eq!(normalize_kind(Some("架构方案")), "design");
    }

    #[test]
    fn normalize_kind_prefers_behavioral_when_a_story_mentions_code() {
        // "讲一次你写代码解决问题的经历" is a story, not a whiteboard task.
        // Behavioural must win, otherwise the coach starts inventing code for
        // an experience question — the exact fabrication we banned.
        assert_eq!(normalize_kind(Some("behavioral 经历 代码")), "behavioral");
        assert_eq!(normalize_kind(Some("行为题(涉及代码)")), "behavioral");
    }

    fn parse_suggestion_accepts_json_code_fence() {
        let suggestion = parse_suggestion(
            r#"```json
{"answer":"可以这样说","bullets":["追问一","追问二"],"clarifyingQuestion":null}
```"#,
        );

        assert_eq!(suggestion.answer, "可以这样说");
        assert_eq!(suggestion.bullets, vec!["追问一", "追问二"]);
        assert_eq!(suggestion.clarifying_question, None);
    }

    #[test]
    fn parse_suggestion_keeps_all_design_outline_points() {
        // `design` uses bullets as the primary answer outline, so all five
        // promised points must survive the defensive parser ceiling.
        let suggestion = parse_suggestion(
            r#"{"kind":"design","answer":"一句总览","bullets":["一","二","三","四","五"],"clarifyingQuestion":null}"#,
        );
        assert_eq!(suggestion.kind, "design");
        assert_eq!(suggestion.bullets, vec!["一", "二", "三", "四", "五"]);
    }

    #[test]
    fn prose_fallback_keeps_the_paragraph_whole() {
        // Regression: this used to be split into a conclusion plus three
        // bullets, which manufactured the flat list shape the output contract
        // now forbids. A spoken answer IS one paragraph — take it as-is.
        let prose = "HashMap 底层是数组加链表：key 算哈希后对数组长度取模定位到桶，\
所以平均读写是 O(1)；多个 key 撞到同一个桶就叫冲突，用链表串起来，冲突多了会退化成 \
O(n)。Java 8 之后链表超过 8 会转成红黑树，把最坏情况压到 O(log n)。";

        let suggestion = parse_suggestion(prose);
        assert_eq!(suggestion.answer, prose);
        assert!(suggestion.bullets.is_empty());
        // Nothing was torn apart: 分号 and "O(1)." both survive in place.
        assert!(suggestion.answer.contains("；"));
        assert!(suggestion.answer.contains("O(1)；"));
    }

    #[test]
    fn prose_fallback_leaves_short_answers_alone() {
        let suggestion = parse_suggestion("有的。我用过 Redis 做分布式锁。");
        assert_eq!(suggestion.answer, "有的。我用过 Redis 做分布式锁。");
        assert!(suggestion.bullets.is_empty());
    }

    #[test]
    fn prose_fallback_never_invents_structure() {
        // Even a long prose reply stays whole: we never synthesise bullets
        // from it, and we never truncate it (length is the prompt's job).
        let prose = "结论先行。第一点。第二点。第三点。第四点。第五点。第六点。";
        let suggestion = parse_suggestion(prose);
        assert_eq!(suggestion.answer, prose);
        assert!(suggestion.bullets.is_empty());
    }

    #[test]
    fn parse_suggestion_still_caps_beyond_contract() {
        let suggestion = parse_suggestion(
            r#"{"answer":"结论","bullets":["一","二","三","四","五","六"],"clarifyingQuestion":null}"#,
        );
        assert_eq!(suggestion.bullets.len(), MAX_BULLETS);
    }

    #[test]
    fn retries_only_copilot_non_stream_error() {
        assert!(requires_streaming_retry(
            StatusCode::BAD_REQUEST,
            r#"{"code":11101,"msg":"Non-stream chat request is currently not supported"}"#,
        ));
        assert!(requires_streaming_retry(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"code":"11101","message":"stream required"}}"#,
        ));
        assert!(requires_streaming_retry(
            StatusCode::BAD_REQUEST,
            "Non-stream chat request is currently not supported",
        ));
        assert!(!requires_streaming_retry(
            StatusCode::UNAUTHORIZED,
            r#"{"code":11101}"#,
        ));
        assert!(!requires_streaming_retry(
            StatusCode::BAD_REQUEST,
            r#"{"code":10001,"msg":"invalid api key"}"#,
        ));
    }

    #[test]
    fn joins_openai_compatible_sse_deltas() {
        let response = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好，\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"世界\"}}]}\n\n",
            "data: [DONE]\n\n",
        );

        assert_eq!(parse_sse_content(response).unwrap(), "你好，世界");
    }
}
