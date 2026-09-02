use super::audio_normalization::normalize_to_wav_16k_mono;
use super::{AsrCapabilities, AsrExecutionMode, BatchAsrRequest, SttProvider};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use crate::providers::config::ProviderId;
use crate::providers::credentials::ResolvedCredentials;
use crate::providers::error::{ProviderFailure, ProviderResult};
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

    async fn transcribe_via_streaming_chat(&self, request: BatchAsrRequest) -> ProviderResult<String> {
        let wav = normalize_to_wav_16k_mono(request)
            .map_err(|error| ProviderFailure::invalid_request(self.id(), error.to_string()))?;
        let audio_data_url = format!("data:audio/wav;base64,{}", BASE64.encode(wav));
        if audio_data_url.len() > MAX_CHAT_AUDIO_DATA_URL_BYTES {
            return Err(ProviderFailure::invalid_request(
                self.id(),
                "Normalized audio exceeds the chat-audio Base64 size limit.",
            ));
        }

        let body = json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": { "data": audio_data_url }
                }, {
                    "type": "text",
                    "text": "请逐字转写这段音频。只输出转写文本，不要解释。"
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
            let chunk = chunk
                .map_err(|error| ProviderFailure::transport(self.id(), error))?;
            payload.push_str(&String::from_utf8_lossy(&chunk));
        }
        parse_sse_transcript(&payload)
            .map_err(|message| ProviderFailure::invalid_response(self.id(), message))
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
            // Tencent Copilot rejects the Whisper upload with code 11101 because
            // its chat endpoint supports streaming only. Retry once using the
            // OpenAI audio-in-chat payload; other 4xx errors remain unchanged.
            if requires_chat_audio_streaming_retry(status, &body) {
                let _ = crate::debug_log::append(
                    "[stt] non-stream request rejected with 11101; retrying chat-audio stream",
                );
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

fn parse_sse_transcript(payload: &str) -> Result<String, &'static str> {
    let mut transcript = String::new();
    for line in payload.lines() {
        let Some(data) = line.trim_start().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            return Err("Chat-audio streaming response contained an invalid SSE event.");
        };
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

    if transcript.trim().is_empty() {
        return Err("Chat-audio streaming response contained no transcript text.");
    }
    Ok(transcript.trim().to_string())
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
    fn joins_streaming_audio_transcript_deltas() {
        let payload = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"第一句。\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"第二句。\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(parse_sse_transcript(payload), Ok("第一句。第二句。".to_string()));
    }
}
