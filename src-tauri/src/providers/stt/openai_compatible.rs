use super::audio_normalization::normalize_to_wav_16k_mono;
use super::{AsrCapabilities, AsrExecutionMode, BatchAsrRequest, SttProvider};
use crate::providers::config::ProviderId;
use crate::providers::credentials::ResolvedCredentials;
use crate::providers::error::{ProviderFailure, ProviderResult};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::StreamExt;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{json, Value};

const MAX_CHAT_AUDIO_DATA_URL_BYTES: usize = 10_000_000;

pub struct OpenAiCompatibleStt {
    base_url: String,
    model: String,
    api_key: String,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct SttResponse {
    text: String,
}

impl OpenAiCompatibleStt {
    pub fn new(credentials: ResolvedCredentials) -> Self {
        Self {
            base_url: credentials.base_url,
            model: credentials.model,
            api_key: credentials.api_key,
            client: reqwest::Client::new(),
        }
    }

    async fn transcribe_via_streaming_chat(
        &self,
        request: BatchAsrRequest,
    ) -> ProviderResult<String> {
        let wav = normalize_to_wav_16k_mono(request)
            .map_err(|error| ProviderFailure::invalid_request(self.id(), error.to_string()))?;
        let audio_data_url = format!("data:audio/wav;base64,{}", BASE64.encode(wav));
        if audio_data_url.len() > MAX_CHAT_AUDIO_DATA_URL_BYTES {
            return Err(ProviderFailure::invalid_request(
                self.id(),
                "Normalized audio exceeds the chat-audio Base64 size limit.",
            ));
        }

        // Bailian's compatible mode allows exactly one item in the user
        // message content array, and it must be `input_audio`. Sending the
        // "transcribe this" instruction as a second `type: "text"` item makes
        // the ASR task reject the whole request with
        //   InternalError.Algo.InvalidParameter: The dedicated task `asr`
        //   ... does not support this input.
        // So no instruction is sent here. Models that then fall back to their
        // training template answer `language Chinese<asr_text>…</asr_text>`,
        // which strip_asr_text_tags() already unwraps.
        let body = json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": { "data": audio_data_url }
                }]
            }],
            "stream": true,
        });
        let response = self
            .client
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|error| ProviderFailure::transport(self.id(), error))?;
        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            let _ = crate::debug_log::append(&format!(
                "[stt] chat-audio streaming retry rejected status={status}",
            ));
            return Err(ProviderFailure::http(self.id(), status, &error_body));
        }

        let _ = crate::debug_log::append("[stt] chat-audio streaming retry accepted");
        let mut stream = response.bytes_stream();
        let mut payload = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| ProviderFailure::transport(self.id(), error))?;
            payload.push_str(&String::from_utf8_lossy(&chunk));
        }
        match parse_sse_transcript(&payload) {
            Ok(transcript) => Ok(transcript),
            Err(message) => {
                // A bare "no transcript" says nothing about *why*. The head of
                // the response is short and is the only way to tell an empty
                // recognition from a payload shape we failed to read.
                let _ = crate::debug_log::append(&format!(
                    "[stt] chat-audio stream yielded no transcript: {message} | head={}",
                    response_preview(&payload, 600)
                ));
                Err(ProviderFailure::invalid_response(
                    self.id(),
                    format!(
                        "{message} Response head: {}",
                        response_preview(&payload, 300)
                    ),
                ))
            }
        }
    }
}

#[async_trait::async_trait]
impl SttProvider for OpenAiCompatibleStt {
    fn id(&self) -> ProviderId {
        ProviderId::OpenAiCompatible
    }

    fn capabilities(&self) -> AsrCapabilities {
        AsrCapabilities {
            execution_mode: AsrExecutionMode::Batch,
            supports_language_hint: false,
            requires_wav_normalization: false,
            max_audio_duration_ms: None,
        }
    }

    async fn transcribe(&self, request: BatchAsrRequest) -> ProviderResult<String> {
        // Preserve the original request only for the narrowly-scoped Copilot
        // fallback below. Normal Whisper-compatible ASR providers retain their
        // existing multipart request and response contract.
        let fallback_request = BatchAsrRequest {
            audio_bytes: request.audio_bytes.clone(),
            filename: request.filename.clone(),
            mime_type: request.mime_type.clone(),
        };
        let part = reqwest::multipart::Part::bytes(request.audio_bytes)
            .file_name(request.filename)
            .mime_str(&request.mime_type)
            .map_err(|error| ProviderFailure::invalid_request(self.id(), error.to_string()))?;

        let form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .part("file", part);

        let response = self
            .client
            .post(&self.base_url)
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(|error| ProviderFailure::transport(self.id(), error))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            // Two known families of chat-only ASR endpoints reject the Whisper
            // multipart upload: Tencent Copilot (code 11101, streaming-only)
            // and Bailian's compatible-mode, which has no multipart endpoint
            // at all ("Required body invalid, please check the request body
            // format"). Both speak the OpenAI audio-in-chat payload, so retry
            // once with that; auth and quota errors are left untouched.
            if requires_chat_audio_streaming_retry(status, &body) {
                let _ = crate::debug_log::append(&format!(
                    "[stt] multipart rejected status={status}; retrying chat-audio stream",
                ));
                return self.transcribe_via_streaming_chat(fallback_request).await;
            }
            let _ = crate::debug_log::append(&format!(
                "[stt] non-stream request rejected status={status}; no streaming retry",
            ));
            return Err(ProviderFailure::http(self.id(), status, &body));
        }

        let parsed: SttResponse = response
            .json()
            .await
            .map_err(|error| ProviderFailure::invalid_response(self.id(), error.to_string()))?;
        Ok(parsed.text)
    }
}

fn requires_chat_audio_streaming_retry(status: StatusCode, body: &str) -> bool {
    // Endpoint-level mismatches: a host that only exposes /chat/completions
    // answers a multipart upload with 404/405/415 regardless of key validity.
    if matches!(
        status,
        StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED | StatusCode::UNSUPPORTED_MEDIA_TYPE
    ) {
        return true;
    }

    if status != StatusCode::BAD_REQUEST {
        return false;
    }

    let code_is_11101 = serde_json::from_str::<Value>(body).ok().and_then(|value| {
        let error = value.get("error").unwrap_or(&value);
        error.get("code").and_then(|code| {
            code.as_i64()
                .or_else(|| code.as_str().and_then(|value| value.parse::<i64>().ok()))
        })
    }) == Some(11101);
    if code_is_11101 || body.contains("Non-stream chat request is currently not supported") {
        return true;
    }

    // Body-format rejections, e.g. Bailian compatible-mode:
    // "Required body invalid, please check the request body format."
    let lowered = body.to_lowercase();
    lowered.contains("required body invalid") || lowered.contains("request body format")
}

fn parse_sse_transcript(payload: &str) -> Result<String, String> {
    let mut transcript = String::new();
    let mut saw_data_line = false;
    for line in payload.lines() {
        let Some(data) = line.trim_start().strip_prefix("data:") else {
            continue;
        };
        saw_data_line = true;
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            return Err(
                "Chat-audio streaming response contained an invalid SSE event.".to_string(),
            );
        };
        // A rejected request can come back as 200 with the error *inside* the
        // stream -- Bailian does exactly this. Such an event carries no delta,
        // so without this check the failure reads as "no transcript text" and
        // the actual reason is lost.
        if let Some(error) = event.get("error") {
            let code = error
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or("unknown_error");
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("(no message)");
            return Err(format!(
                "ASR provider rejected the request in-stream ({code}): {message}"
            ));
        }
        if let Some(content) = event
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("delta"))
            .and_then(|delta| delta.get("content"))
            .and_then(Value::as_str)
        {
            transcript.push_str(content);
        }
    }

    if !transcript.trim().is_empty() {
        return Ok(strip_asr_text_tags(&transcript));
    }

    // A host that ignores `stream: true` answers with a single JSON object
    // instead of an SSE stream. Read that shape before declaring the response
    // empty: `choices[0].message.content` is where its text lives.
    if !saw_data_line {
        if let Some(text) = parse_json_transcript(payload) {
            if !text.trim().is_empty() {
                return Ok(strip_asr_text_tags(&text));
            }
        }
    }

    Err(super::EMPTY_TRANSCRIPT_MESSAGE.to_string())
}

/// Reads a non-streaming `chat.completion` body. `content` is a string in
/// Bailian's compatible mode but is typed as an array of parts elsewhere, so
/// both shapes are accepted.
fn parse_json_transcript(payload: &str) -> Option<String> {
    let value = serde_json::from_str::<Value>(payload.trim()).ok()?;
    let choice = value.get("choices")?.get(0)?;
    let content = choice
        .get("message")
        .and_then(|message| message.get("content"))
        .or_else(|| choice.get("delta").and_then(|delta| delta.get("content")))
        .or_else(|| choice.get("text"))?;
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<String>(),
        ),
        _ => None,
    }
}

fn response_preview(payload: &str, limit: usize) -> String {
    let collapsed: String = payload
        .chars()
        .map(|character| {
            if character.is_whitespace() {
                ' '
            } else {
                character
            }
        })
        .take(limit)
        .collect();
    let collapsed = collapsed.trim().to_string();
    if collapsed.is_empty() {
        "<empty response body>".to_string()
    } else {
        collapsed
    }
}

/// 部分 chat-asr 模型即便在 prompt 里被要求"只输出转写文本"，仍会按
/// 训练时的模板把 `language Chinese<asr_text>…</asr_text>` 一并返回。
/// 这里在解析层兜底，确保前端拿到的只有裸文本。
fn strip_asr_text_tags(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(start) = trimmed.find("<asr_text>") {
        let after = &trimmed[start + "<asr_text>".len()..];
        if let Some(end) = after.find("</asr_text>") {
            return after[..end].trim().to_string();
        }
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retries_only_copilot_non_stream_error() {
        assert!(requires_chat_audio_streaming_retry(
            StatusCode::BAD_REQUEST,
            r#"{"code":11101,"msg":"Non-stream chat request is currently not supported"}"#,
        ));
        assert!(requires_chat_audio_streaming_retry(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"code":"11101","message":"stream required"}}"#,
        ));
        assert!(!requires_chat_audio_streaming_retry(
            StatusCode::BAD_REQUEST,
            r#"{"code":10001,"msg":"invalid api key"}"#,
        ));
        assert!(!requires_chat_audio_streaming_retry(
            StatusCode::UNAUTHORIZED,
            r#"{"code":11101}"#,
        ));
    }

    #[test]
    fn retries_on_body_format_rejections() {
        // Bailian compatible-mode has no multipart endpoint; its 400 is a
        // body-format complaint, not an auth failure.
        assert!(requires_chat_audio_streaming_retry(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"code":"invalid_request_error","message":"Required body invalid, please check the request body format."}}"#,
        ));
        // Endpoint-level mismatches also warrant the chat retry.
        assert!(requires_chat_audio_streaming_retry(
            StatusCode::NOT_FOUND,
            "not found",
        ));
        assert!(requires_chat_audio_streaming_retry(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported media type",
        ));
        // Auth/quota problems must not be masked by a doomed retry.
        assert!(!requires_chat_audio_streaming_retry(
            StatusCode::TOO_MANY_REQUESTS,
            "rate limited",
        ));
    }

    #[test]
    fn joins_streaming_audio_transcript_deltas() {
        let payload = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"第一句。\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"第二句。\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(
            parse_sse_transcript(payload),
            Ok("第一句。第二句。".to_string())
        );
    }

    #[test]
    fn reports_an_in_stream_error_instead_of_an_empty_transcript() {
        // Bailian answers 200 and puts invalid_parameter_error in the stream.
        let payload = concat!(
            "data: {\"error\":{\"code\":\"invalid_parameter_error\",\"message\":\"<400> ",
            "InternalError.Algo.InvalidParameter: The dedicated task `asr` ... does not ",
            "support this input.\",\"type\":\"invalid_request_error\"}}\n\n",
        );
        let error = parse_sse_transcript(payload).unwrap_err();
        assert!(
            error.starts_with("ASR provider rejected the request in-stream"),
            "{error}"
        );
        assert!(error.contains("InvalidParameter"), "{error}");
    }

    #[test]
    fn reads_plain_json_when_streaming_is_ignored() {
        // A host that ignores `stream: true` answers with one JSON object.
        let payload = r#"{"choices":[{"message":{"content":"你好，面试助手。"}}]}"#;
        assert_eq!(
            parse_sse_transcript(payload),
            Ok("你好，面试助手。".to_string())
        );
    }

    #[test]
    fn reads_json_content_typed_as_parts() {
        let payload =
            r#"{"choices":[{"message":{"content":[{"type":"text","text":"第一段。"}]}}]}"#;
        assert_eq!(parse_sse_transcript(payload), Ok("第一段。".to_string()));
    }

    #[test]
    fn reports_empty_payload_instead_of_guessing() {
        assert_eq!(response_preview("", 300), "<empty response body>");
        assert_eq!(response_preview("  \n data: x ", 300), "data: x");
        // Long payloads are cut, not dumped whole into the UI.
        assert!(response_preview(&"a".repeat(5_000), 300).len() <= 300);
    }

    #[test]
    fn streaming_payload_strips_asr_text_markup() {
        let payload = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"language Chinese<asr_text>\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好，面试助手。\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"</asr_text>\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(
            parse_sse_transcript(payload),
            Ok("你好，面试助手。".to_string())
        );
    }
}
