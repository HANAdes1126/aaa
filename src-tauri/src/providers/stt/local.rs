use super::audio_normalization::normalize_to_wav_16k_mono;
use super::{AsrCapabilities, AsrExecutionMode, BatchAsrRequest, SttProvider};
use crate::providers::config::{ProviderConfig, ProviderId};
use crate::providers::error::{ProviderFailure, ProviderFailureKind, ProviderResult};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde_json::{json, Value};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// llama-server listens on this local port. The default `base_url` mirrors it.
const SERVER_PORT: u16 = 8080;
const READY_TIMEOUT_MS: u64 = 60_000;
const READY_POLL_INTERVAL_MS: u64 = 500;

/// Holds the single long-lived llama-server child process so the model is
/// loaded exactly once and reused across transcriptions (lowest latency).
static SERVER: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

fn server_slot() -> &'static Mutex<Option<Child>> {
    SERVER.get_or_init(|| Mutex::new(None))
}

pub struct LocalQwen3AsrStt {
    base_url: String,
    model: String,
    mmproj_path: String,
    client: reqwest::Client,
}

impl LocalQwen3AsrStt {
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            base_url: config.base_url,
            model: config.model,
            mmproj_path: config.mmproj_path.unwrap_or_default(),
            client: reqwest::Client::new(),
        }
    }

    async fn ensure_server(&self) -> ProviderResult<()> {
        if self.health_ok().await {
            return Ok(());
        }

        self.spawn_server_once()?;

        let deadline = Instant::now() + Duration::from_millis(READY_TIMEOUT_MS);
        loop {
            if self.health_ok().await {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(ProviderFailure::new(
                    self.id(),
                    ProviderFailureKind::ServiceUnavailable,
                    true,
                    "llama-server 未在超时时间内就绪",
                ));
            }
            tokio::time::sleep(Duration::from_millis(READY_POLL_INTERVAL_MS)).await;
        }
    }

    async fn health_ok(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        let request = self.client.get(&url).send();
        match tokio::time::timeout(Duration::from_secs(2), request).await {
            Ok(Ok(response)) => response.status().is_success(),
            _ => false,
        }
    }

    fn spawn_server_once(&self) -> ProviderResult<()> {
        let mut slot = server_slot()
            .lock()
            .map_err(|_| ProviderFailure::new(self.id(), ProviderFailureKind::Internal, true, "进程锁已中毒"))?;

        if let Some(child) = slot.as_mut() {
            if matches!(child.try_wait(), Ok(None)) {
                // Existing process is still alive — reuse it.
                return Ok(());
            }
        }

        if self.model.trim().is_empty() {
            return Err(ProviderFailure::invalid_request(
                self.id(),
                "未配置 GGUF 模型路径（model 字段）",
            ));
        }
        if self.mmproj_path.trim().is_empty() {
            return Err(ProviderFailure::invalid_request(
                self.id(),
                "未配置 mmproj 音频投影路径",
            ));
        }

        let llama_server = resolve_llama_server();
        let child = Command::new(&llama_server)
            .arg("-m")
            .arg(&self.model)
            .arg("--mmproj")
            .arg(&self.mmproj_path)
            .arg("-c")
            .arg("2048")
            .arg("-ngl")
            .arg("99")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(SERVER_PORT.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| {
                ProviderFailure::new(
                    self.id(),
                    ProviderFailureKind::Connect,
                    true,
                    format!("无法启动 llama-server（{llama_server}）: {error}"),
                )
            })?;

        let _ = crate::debug_log::append(&format!(
            "[stt] local llama-server spawned pid={}",
            child.id()
        ));
        *slot = Some(child);
        Ok(())
    }
}

#[async_trait::async_trait]
impl SttProvider for LocalQwen3AsrStt {
    fn id(&self) -> ProviderId {
        ProviderId::LocalQwen3Asr
    }

    fn capabilities(&self) -> AsrCapabilities {
        AsrCapabilities {
            execution_mode: AsrExecutionMode::Batch,
            supports_language_hint: false,
            requires_wav_normalization: true,
            max_audio_duration_ms: None,
        }
    }

    async fn transcribe(&self, request: BatchAsrRequest) -> ProviderResult<String> {
        self.ensure_server().await?;

        let wav = normalize_to_wav_16k_mono(request)
            .map_err(|error| ProviderFailure::invalid_request(self.id(), error.to_string()))?;
        let audio_base64 = BASE64.encode(wav);

        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请转写这段音频，只输出转写文本，不要任何解释。"
                    },
                    {
                        "type": "input_audio",
                        "input_audio": { "data": audio_base64, "format": "wav" }
                    }
                ]
            }],
            "max_tokens": 256,
            "temperature": 0,
            "stream": false
        });

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|error| ProviderFailure::transport(self.id(), error))?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            return Err(ProviderFailure::http(self.id(), status, &error_body));
        }

        let value: Value = response
            .json()
            .await
            .map_err(|error| ProviderFailure::invalid_response(self.id(), error.to_string()))?;
        let raw = parse_content(&value).map_err(|message| ProviderFailure::invalid_response(self.id(), message))?;
        Ok(clean_transcript(&raw))
    }
}

fn parse_content(body: &Value) -> Result<String, &'static str> {
    body.get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or("llama-server 响应缺少 choices[0].message.content")
}

/// Qwen3-ASR emits `language Chinese<asr_text>你好...</asr_text>`. Strip the
/// markup and return the bare transcript.
fn clean_transcript(raw: &str) -> String {
    if let Some(start) = raw.find("<asr_text>") {
        let after = &raw[start + "<asr_text>".len()..];
        if let Some(end) = after.find("</asr_text>") {
            return after[..end].trim().to_string();
        }
    }
    raw.trim().to_string()
}

/// Resolve the llama-server executable: env override, then known install
/// locations, then PATH.
fn resolve_llama_server() -> String {
    if let Ok(path) = std::env::var("MEETLY_LLAMA_SERVER") {
        if !path.trim().is_empty() {
            return path;
        }
    }
    if let Some(home) = dirs::home_dir() {
        for relative in [
            "llama.cpp/llama-b10621/llama-server",
            "llama.cpp/build/bin/llama-server",
        ] {
            let candidate = home.join(relative);
            if candidate.is_file() {
                return candidate.to_string_lossy().to_string();
            }
        }
    }
    "llama-server".to_string()
}

/// Kill the managed llama-server process. Called on app exit.
pub fn shutdown_server() {
    if let Some(slot) = SERVER.get() {
        if let Ok(mut guard) = slot.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_asr_text_markup() {
        assert_eq!(
            clean_transcript("language Chinese<asr_text>你好，面试助手。</asr_text>"),
            "你好，面试助手。"
        );
    }

    #[test]
    fn falls_back_to_raw_when_no_markup() {
        assert_eq!(clean_transcript("你好世界"), "你好世界");
    }

    #[test]
    fn parses_chat_completions_content() {
        let body = json!({"choices": [{"message": {"content": "你好"}}]});
        assert_eq!(parse_content(&body), Ok("你好".to_string()));
    }
}
