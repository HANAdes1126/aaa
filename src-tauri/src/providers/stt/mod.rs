mod audio_normalization;
mod local;
mod mimo;
mod openai_compatible;
pub mod text_guard;

use crate::providers::config::{DiagnosticResult, ProviderId, ProviderKind};
use crate::providers::error::{ProviderFailureKind, ProviderResult};
use crate::providers::{credentials, storage};
use anyhow::{anyhow, Result};
use tauri::AppHandle;

/// Raised when an ASR response is well formed but carries no text. Shared so
/// the connectivity probe can recognise the one failure that silence makes
/// expected, without matching on wording scattered across providers.
pub const EMPTY_TRANSCRIPT_MESSAGE: &str =
    "Chat-audio streaming response contained no transcript text.";

pub use local::{shutdown_server, LocalQwen3AsrStt};
pub use mimo::MimoStt;
pub use openai_compatible::OpenAiCompatibleStt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrExecutionMode {
    Batch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsrCapabilities {
    pub execution_mode: AsrExecutionMode,
    pub supports_language_hint: bool,
    pub requires_wav_normalization: bool,
    pub max_audio_duration_ms: Option<u64>,
}

pub struct BatchAsrRequest {
    pub audio_bytes: Vec<u8>,
    pub filename: String,
    pub mime_type: String,
}

impl BatchAsrRequest {
    pub fn new(audio_bytes: Vec<u8>, filename: &str, mime_type: &str) -> Self {
        Self {
            audio_bytes,
            filename: filename.to_string(),
            mime_type: mime_type.to_string(),
        }
    }
}

#[async_trait::async_trait]
pub trait SttProvider: Send + Sync {
    fn id(&self) -> ProviderId;
    fn capabilities(&self) -> AsrCapabilities;
    async fn transcribe(&self, request: BatchAsrRequest) -> ProviderResult<String>;

    /// Transcribes with progressive partial-text callbacks. Providers whose
    /// backend supports SSE streaming override this to emit the accumulated
    /// transcript as it arrives (lower perceived latency); the default falls
    /// back to a single-shot batch response so existing providers keep working
    /// unchanged.
    async fn transcribe_streaming(
        &self,
        request: BatchAsrRequest,
        mut on_delta: Box<dyn FnMut(String) + Send>,
    ) -> ProviderResult<String> {
        let text = self.transcribe(request).await?;
        on_delta(text.clone());
        Ok(text)
    }
}

pub fn build_from_saved_config(app: &AppHandle) -> Result<Box<dyn SttProvider>> {
    let config = storage::get_config(app, ProviderKind::Stt)?;
    let provider: Box<dyn SttProvider> = if config.provider_id == ProviderId::LocalQwen3Asr {
        // Local model requires no API key, so bypass credentials::resolve.
        Box::new(LocalQwen3AsrStt::new(config))
    } else {
        let credentials = credentials::resolve(app, ProviderKind::Stt)?;
        match credentials.provider_id {
            ProviderId::OpenAiCompatible => Box::new(OpenAiCompatibleStt::new(credentials)),
            ProviderId::XiaomiMimo => Box::new(MimoStt::new(credentials)),
            ProviderId::LocalQwen3Asr => unreachable!("handled above"),
        }
    };
    if !provider.id().supports(ProviderKind::Stt) {
        return Err(anyhow!("Configured provider does not support ASR."));
    }
    Ok(provider)
}

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

    let request = BatchAsrRequest::new(
        audio_normalization::silence_probe_wav(),
        "probe.wav",
        "audio/wav",
    );
    let reachable = format!(
        "{} 接口可访问，并已接受语音转写测试请求。",
        provider.id().as_str()
    );
    match provider.transcribe(request).await {
        Ok(_) => DiagnosticResult {
            success: true,
            message: reachable,
        },
        Err(error) => {
            // The probe is 0.2 s of digital silence, so an engine that answers
            // correctly has nothing to transcribe. Reaching that state proves
            // the endpoint, key and payload shape are all accepted -- which is
            // exactly what this button tests. Any other failure is real.
            if error.kind == ProviderFailureKind::InvalidResponse
                && error.message().starts_with(EMPTY_TRANSCRIPT_MESSAGE)
            {
                return DiagnosticResult {
                    success: true,
                    message: format!("{reachable}（探针为静音，无转写文本属预期）"),
                };
            }
            DiagnosticResult {
                success: false,
                message: error.to_string(),
            }
        }
    }
}
