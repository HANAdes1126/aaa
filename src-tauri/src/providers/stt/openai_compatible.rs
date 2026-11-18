use super::audio_normalization::normalize_to_wav_16k_mono;
use super::{AsrCapabilities, AsrExecutionMode, BatchAsrRequest, SttProvider};
use crate::providers::config::ProviderId;
use crate::providers::credentials::ResolvedCredentials;
use crate::providers::error::{ProviderFailure, ProviderFailureKind, ProviderResult};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::StreamExt;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;

const MAX_CHAT_AUDIO_DATA_URL_BYTES: usize = 10_000_000;

/// Ceiling for one *silent* stretch of an STT round trip.
///
/// This is not a budget for a long task — it is how "the server is still
/// working" is told apart from "the socket is dead". An STT request carries one
/// VAD segment (15 seconds of audio at most) and a healthy reply lands in 1–2
/// seconds, so 15 seconds of complete silence can only mean the connection is
/// gone.
///
/// It has to exist because a server that accepts the request and then neither
/// ends the SSE stream nor closes the socket leaves `stream.next()` pending
/// forever. That task never reaches `finish`/`abandon`, so its `start_ms` stays
/// in `TranscriptOrderGate::in_flight` and every later segment of the meeting is
/// held behind it: the transcript stalls or arrives seconds late, and only
/// restarting the session (which resets the gate) brings it back.
const STT_IDLE_TIMEOUT: Duration = Duration::from_secs(15);

/// How long the Whisper-style multipart upload may stay silent before the
/// endpoint is taken to have no multipart endpoint at all.
///
/// Deliberately short, and unrelated to `STT_IDLE_TIMEOUT`: this upload is a
/// probe, and a wrong guess here only costs a hand-off to chat-audio. On the
/// one endpoint measured against it, the probe used to come back with a 400 in
/// ~230ms, so 4s is ~17x margin for a rejection. Waiting the full idle ceiling
/// (15s, what 0.1.11 did) turned a probe into the slowest part of the segment:
/// the first transcript after launch sat out the whole budget before the real
/// request was even sent.
const STT_PROBE_TIMEOUT: Duration = Duration::from_secs(4);

/// Ceiling for the multipart upload *after* chat-audio has already failed.
///
/// At that point multipart is the only remaining path, so the wait is sized for
/// real work rather than for a probe: a 15s clip on a hosted Whisper endpoint
/// answers in a few seconds, and the response headers only arrive once the
/// transcription is done.
const STT_MULTIPART_RETRY_TIMEOUT: Duration = Duration::from_secs(30);

/// Reports a stalled STT call through both channels that matter: the log (so the
/// next occurrence is diagnosable) and the caller (so the segment is abandoned
/// instead of pinning the order gate).
fn stt_timeout(provider: ProviderId, message: String) -> ProviderFailure {
    let _ = crate::debug_log::append(&format!("[stt] {message}"));
    ProviderFailure::new(
        provider,
        ProviderFailureKind::Timeout,
        true,
        message,
    )
}

/// Endpoints that already told us they cannot take a Whisper-style multipart
/// upload, keyed by `base_url|model`.
///
/// Every segment used to pay for that discovery: measured 2026-11-18 across a
/// long log, the probe was rejected 1518 times against 1549 successes, i.e. ~98%
/// of calls wasted an upload. On a gateway that answers the probe with a body
/// rejection that cost ~230ms; on one that never answers it costs the whole
/// probe ceiling, in front of every single segment.
///
/// Remembering the verdict keeps the working path (chat-audio) as the only path
/// after the first segment ever, so the probe is not re-paid on every launch.
///
/// Persisted, not just in-memory like the LLM's `non_stream_known_unsupported`:
/// that cache is rebuilt from a cheap 400, whereas rebuilding this one costs the
/// user a visible pause in front of their first sentence after every restart.
///
/// A stale verdict cannot strand anyone: if chat-audio ever fails, the upload is
/// retried with its full budget before the segment is given up.
static MULTIPART_UNSUPPORTED: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashSet<String>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashSet::new()));

/// Seeds `MULTIPART_UNSUPPORTED` from disk the first time it is consulted.
static MULTIPART_VERDICTS_LOADED: std::sync::LazyLock<()> = std::sync::LazyLock::new(|| {
    let Some(path) = multipart_verdict_path() else {
        return;
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    if let Ok(mut set) = MULTIPART_UNSUPPORTED.write() {
        for line in text.lines() {
            let line = line.trim();
            if !line.is_empty() {
                set.insert(line.to_string());
            }
        }
    }
});

fn multipart_verdict_path() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|home| {
        home.join(".meetly")
            .join("stt-multipart-unsupported.txt")
    })
}

fn multipart_key(base_url: &str, model: &str) -> String {
    format!("{base_url}|{model}")
}

fn multipart_known_unsupported(key: &str) -> bool {
    std::sync::LazyLock::force(&MULTIPART_VERDICTS_LOADED);
    MULTIPART_UNSUPPORTED
        .read()
        .map(|set| set.contains(key))
        .unwrap_or(false)
}

fn remember_multipart_unsupported(key: &str) {
    let already = MULTIPART_UNSUPPORTED
        .read()
        .map(|set| set.contains(key))
        .unwrap_or(true);
    if let Ok(mut set) = MULTIPART_UNSUPPORTED.write() {
        set.insert(key.to_string());
    }
    if already {
        return;
    }
    // Best effort: a read-only home directory only costs a repeated probe.
    let Some(path) = multipart_verdict_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        use std::io::Write as _;
        let _ = writeln!(file, "{key}");
    }
}

/// What one attempt at the Whisper-style upload produced.
///
/// The distinction that matters is *"try the other path"* versus *"this will
/// never work on this endpoint"*: an auth error or a 5xx is not a hint about
/// the payload format, so retrying it over chat-audio would only bury the real
/// message.
enum MultipartAttempt {
    Transcript(String),
    /// The endpoint has no usable multipart path. `retry_later` is true when it
    /// went silent rather than rejecting, i.e. the upload might still have
    /// worked if given longer -- worth one slow retry if chat-audio fails too.
    FallBack { retry_later: bool },
    /// Do not retry: the failure is about the credentials or the server.
    Fatal(ProviderFailure),
}

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
        let response = tokio::time::timeout(
            STT_IDLE_TIMEOUT,
            self.client
                .post(&self.base_url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| {
            stt_timeout(
                self.id(),
                format!(
                    "chat-audio request got no response in {}s.",
                    STT_IDLE_TIMEOUT.as_secs()
                ),
            )
        })?
        .map_err(|error| ProviderFailure::transport(self.id(), error))?;
        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            // The body carries the only explanation the endpoint gives
            // (`InternalError.Algo.InvalidParameter: ...` and friends). Logging
            // just the status made every 400 here impossible to diagnose.
            let _ = crate::debug_log::append(&format!(
                "[stt] chat-audio streaming retry rejected status={status} body={}",
                response_preview(&error_body, 300)
            ));
            return Err(ProviderFailure::http(self.id(), status, &error_body));
        }

        let _ = crate::debug_log::append("[stt] chat-audio streaming retry accepted");
        let mut stream = response.bytes_stream();
        let mut payload = String::new();
        loop {
            // Each wait is capped separately: a stream that opened and then went
            // silent looks exactly like a slow model to a bare `.await`.
            match tokio::time::timeout(STT_IDLE_TIMEOUT, stream.next()).await {
                Ok(Some(chunk)) => {
                    let chunk =
                        chunk.map_err(|error| ProviderFailure::transport(self.id(), error))?;
                    payload.push_str(&String::from_utf8_lossy(&chunk));
                }
                Ok(None) => break,
                Err(_) => {
                    let _ = crate::debug_log::append(&format!(
                        "[stt] chat-audio stream stalled bytes={} idle_s={}",
                        payload.len(),
                        STT_IDLE_TIMEOUT.as_secs()
                    ));
                    return Err(stt_timeout(
                        self.id(),
                        format!(
                            "chat-audio stream had no data for {}s.",
                            STT_IDLE_TIMEOUT.as_secs()
                        ),
                    ));
                }
            }
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

    /// One attempt at the plain Whisper-style upload, capped at `ceiling`.
    async fn multipart_once(
        &self,
        request: &BatchAsrRequest,
        ceiling: Duration,
    ) -> MultipartAttempt {
        let part = match reqwest::multipart::Part::bytes(request.audio_bytes.clone())
            .file_name(request.filename.clone())
            .mime_str(&request.mime_type)
        {
            Ok(part) => part,
            Err(error) => {
                return MultipartAttempt::Fatal(ProviderFailure::invalid_request(
                    self.id(),
                    error.to_string(),
                ))
            }
        };
        let form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .part("file", part);

        let response = match tokio::time::timeout(
            ceiling,
            self.client
                .post(&self.base_url)
                .bearer_auth(&self.api_key)
                .multipart(form)
                .send(),
        )
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                return MultipartAttempt::Fatal(ProviderFailure::transport(self.id(), error))
            }
            Err(_) => {
                // Silence is handled exactly like a refusal: hand the segment to
                // chat-audio instead of spending the whole segment budget, and
                // let the caller retry the upload only if chat-audio also fails.
                let _ = crate::debug_log::append(&format!(
                    "[stt] multipart got no response in {}s; using chat-audio",
                    ceiling.as_secs()
                ));
                return MultipartAttempt::FallBack { retry_later: true };
            }
        };

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            // Two known families of chat-only ASR endpoints reject the Whisper
            // multipart upload: Tencent Copilot (code 11101, streaming-only)
            // and Bailian's compatible-mode, which has no multipart endpoint
            // at all ("Required body invalid, please check the request body
            // format"). Both speak the OpenAI audio-in-chat payload; auth and
            // quota errors are left untouched.
            if requires_chat_audio_streaming_retry(status, &body) {
                let _ = crate::debug_log::append(&format!(
                    "[stt] multipart rejected status={status}; retrying chat-audio stream",
                ));
                return MultipartAttempt::FallBack { retry_later: false };
            }
            let _ = crate::debug_log::append(&format!(
                "[stt] non-stream request rejected status={status}; no streaming retry",
            ));
            return MultipartAttempt::Fatal(ProviderFailure::http(self.id(), status, &body));
        }

        match tokio::time::timeout(STT_IDLE_TIMEOUT, response.json::<SttResponse>()).await {
            Ok(Ok(parsed)) => MultipartAttempt::Transcript(parsed.text),
            Ok(Err(error)) => MultipartAttempt::Fatal(ProviderFailure::invalid_response(
                self.id(),
                error.to_string(),
            )),
            Err(_) => MultipartAttempt::Fatal(stt_timeout(
                self.id(),
                format!(
                    "multipart body had no data for {}s.",
                    STT_IDLE_TIMEOUT.as_secs()
                ),
            )),
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
        // The request survives for the fallback because the two paths need the
        // same audio twice.
        let fallback_request = BatchAsrRequest {
            audio_bytes: request.audio_bytes.clone(),
            filename: request.filename.clone(),
            mime_type: request.mime_type.clone(),
        };

        let key = multipart_key(&self.base_url, &self.model);
        // True only while chat-audio is still unproven *and* the upload went
        // silent rather than refusing: the one case where the slow upload is
        // worth a second attempt.
        let mut may_retry_upload = false;

        if multipart_known_unsupported(&key) {
            // Decided on this run already. Probing again would just repeat it.
            let _ = crate::debug_log::append(
                "[stt] multipart skipped (endpoint answered it before); using chat-audio",
            );
        } else {
            match self.multipart_once(&fallback_request, STT_PROBE_TIMEOUT).await {
                MultipartAttempt::Transcript(text) => return Ok(text),
                MultipartAttempt::Fatal(error) => return Err(error),
                MultipartAttempt::FallBack { retry_later } => {
                    may_retry_upload = retry_later;
                    remember_multipart_unsupported(&key);
                }
            }
        }

        match self
            .transcribe_via_streaming_chat(BatchAsrRequest {
                audio_bytes: fallback_request.audio_bytes.clone(),
                filename: fallback_request.filename.clone(),
                mime_type: fallback_request.mime_type.clone(),
            })
            .await
        {
            Ok(text) => Ok(text),
            // chat-audio failed too, on an endpoint that only went *silent* on
            // the upload. The upload may simply have been slower than a probe
            // is allowed to wait, so give it the real budget rather than
            // dropping the segment. This mirrors the pre-0.1.11 behaviour of
            // waiting for the upload instead of treating it as an error.
            Err(chat_error) if may_retry_upload => {
                let _ = crate::debug_log::append(
                    "[stt] chat-audio failed; retrying multipart upload with the full budget",
                );
                match self
                    .multipart_once(&fallback_request, STT_MULTIPART_RETRY_TIMEOUT)
                    .await
                {
                    MultipartAttempt::Transcript(text) => Ok(text),
                    MultipartAttempt::Fatal(error) => Err(error),
                    MultipartAttempt::FallBack { .. } => Err(chat_error),
                }
            }
            Err(error) => Err(error),
        }
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
