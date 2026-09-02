use super::{AssistantSuggestion, ChatMessage, LlmCapabilities, LlmProvider, ThinkingControl};
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
            client: reqwest::Client::new(),
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

        let response = self.send_request(&body).await?;
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
                "[llm] non-stream request rejected with 11101; retrying stream",
            );
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
        while let Some(chunk) = bytes.next().await {
            let chunk = chunk.context("Failed to read the LLM streaming response")?;
            payload.push_str(&String::from_utf8_lossy(&chunk));
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

/// Parses the LLM's message content into a suggestion. Some
/// OpenAI-compatible providers ignore `response_format: json_object`; when
/// that happens, this falls back to treating the raw text as the answer
/// rather than surfacing a parse error, at the cost of losing the
/// bullets/clarifying_question structure for that one response.
pub(crate) fn parse_suggestion(content: &str) -> AssistantSuggestion {
    let normalized = strip_json_code_fence(content);
    match serde_json::from_str::<AssistantSuggestion>(normalized) {
        Ok(mut suggestion) => {
            suggestion.bullets.truncate(3);
            suggestion
        }
        Err(_) => AssistantSuggestion {
            answer: content.trim().to_string(),
            bullets: Vec::new(),
            clarifying_question: None,
        },
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
    fn parse_suggestion_accepts_json_code_fence() {
        let suggestion = parse_suggestion(
            r#"```json
{"answer":"可以这样说","bullets":["一","二","三","四"],"clarifyingQuestion":null}
```"#,
        );

        assert_eq!(suggestion.answer, "可以这样说");
        assert_eq!(suggestion.bullets.len(), 3);
        assert_eq!(suggestion.clarifying_question, None);
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
