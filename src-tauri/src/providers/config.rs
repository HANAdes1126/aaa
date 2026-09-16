use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Stt,
    Llm,
}

impl ProviderKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderKind::Stt => "stt",
            ProviderKind::Llm => "llm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderId {
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
    XiaomiMimo,
    #[serde(rename = "local_qwen3_asr")]
    LocalQwen3Asr,
}

impl ProviderId {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderId::OpenAiCompatible => "openai_compatible",
            ProviderId::XiaomiMimo => "xiaomi_mimo",
            ProviderId::LocalQwen3Asr => "local_qwen3_asr",
        }
    }

    pub fn supports(self, kind: ProviderKind) -> bool {
        match self {
            ProviderId::OpenAiCompatible => true,
            ProviderId::XiaomiMimo => kind == ProviderKind::Stt,
            ProviderId::LocalQwen3Asr => kind == ProviderKind::Stt,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    pub provider_id: ProviderId,
    pub base_url: String,
    pub model: String,
    /// Model used for screenshot analysis, when it should differ from `model`.
    ///
    /// Vision and live coaching pull in opposite directions: the coach needs an
    /// answer inside ~3s and runs dozens of times per interview, while a
    /// screenshot is a single deliberate action where the user will happily
    /// wait for a stronger model to read the question correctly. Keeping one
    /// field for both forced a compromise that was wrong for one of them.
    ///
    /// `None` (or blank) means "use `model`", so existing configs keep working
    /// untouched and nobody has to fill this in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_path: Option<String>,
}

impl ProviderConfig {
    /// The model to use for image input. Falls back to the text model so an
    /// unset (or accidentally blanked) value can never break screenshots.
    pub fn effective_vision_model(&self) -> &str {
        self.vision_model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(&self.model)
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderDescriptor {
    pub id: ProviderId,
    pub display_name: &'static str,
    pub description: &'static str,
    pub default_base_url: &'static str,
    pub default_model: &'static str,
}

pub fn provider_descriptors(kind: ProviderKind) -> Vec<ProviderDescriptor> {
    let mut descriptors = vec![ProviderDescriptor {
        id: ProviderId::OpenAiCompatible,
        display_name: "OpenAI 兼容",
        description: match kind {
            ProviderKind::Stt => "兼容 Whisper 的分段上传转写接口",
            ProviderKind::Llm => "兼容 OpenAI 的对话补全接口",
        },
        default_base_url: match kind {
            ProviderKind::Stt => "https://api.siliconflow.cn/v1/audio/transcriptions",
            ProviderKind::Llm => "https://api.siliconflow.cn/v1/chat/completions",
        },
        default_model: match kind {
            ProviderKind::Stt => "FunAudioLLM/SenseVoiceSmall",
            ProviderKind::Llm => "Qwen/Qwen3-32B",
        },
    }];

    if kind == ProviderKind::Stt {
        descriptors.push(ProviderDescriptor {
            id: ProviderId::XiaomiMimo,
            display_name: "小米 MiMo",
            description: "MiMo-V2.5-ASR 音频转写接口",
            default_base_url: "https://api.xiaomimimo.com/v1/chat/completions",
            default_model: "mimo-v2.5-asr",
        });
        descriptors.push(ProviderDescriptor {
            id: ProviderId::LocalQwen3Asr,
            display_name: "本地 Qwen3-ASR",
            description: "本地 llama.cpp 离线转写，低延迟",
            default_base_url: "http://127.0.0.1:8080",
            default_model: "",
        });
    }

    descriptors
}

pub fn default_stt_config() -> ProviderConfig {
    ProviderConfig {
        provider_id: ProviderId::OpenAiCompatible,
        base_url: "https://api.siliconflow.cn/v1/audio/transcriptions".to_string(),
        model: "FunAudioLLM/SenseVoiceSmall".to_string(),
        vision_model: None,
        mmproj_path: None,
    }
}

pub fn default_llm_config() -> ProviderConfig {
    ProviderConfig {
        provider_id: ProviderId::OpenAiCompatible,
        base_url: "https://api.siliconflow.cn/v1/chat/completions".to_string(),
        model: "Qwen/Qwen3-32B".to_string(),
        vision_model: None,
        mmproj_path: None,
    }
}

pub fn default_config_for(kind: ProviderKind) -> ProviderConfig {
    match kind {
        ProviderKind::Stt => default_stt_config(),
        ProviderKind::Llm => default_llm_config(),
    }
}

pub(crate) fn infer_legacy_provider(kind: ProviderKind, base_url: &str, model: &str) -> ProviderId {
    if kind == ProviderKind::Stt
        && (base_url.contains("xiaomimimo.com") || model.eq_ignore_ascii_case("mimo-v2.5-asr"))
    {
        ProviderId::XiaomiMimo
    } else {
        ProviderId::OpenAiCompatible
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticResult {
    pub success: bool,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_mimo_stt_config_is_migrated_at_the_storage_boundary() {
        assert_eq!(
            infer_legacy_provider(
                ProviderKind::Stt,
                "https://api.xiaomimimo.com/v1/chat/completions",
                "mimo-v2.5-asr",
            ),
            ProviderId::XiaomiMimo
        );
        assert_eq!(
            infer_legacy_provider(
                ProviderKind::Llm,
                "https://api.deepseek.com/chat/completions",
                "deepseek-chat",
            ),
            ProviderId::OpenAiCompatible
        );
    }

    #[test]
    fn provider_choices_are_independent_per_kind() {
        assert_eq!(provider_descriptors(ProviderKind::Stt).len(), 3);
        assert_eq!(provider_descriptors(ProviderKind::Llm).len(), 1);
        assert!(!ProviderId::XiaomiMimo.supports(ProviderKind::Llm));
        assert!(!ProviderId::LocalQwen3Asr.supports(ProviderKind::Llm));
        assert!(ProviderId::LocalQwen3Asr.supports(ProviderKind::Stt));
    }

    #[test]
    fn frontend_config_uses_camel_case_fields() {
        let value = serde_json::to_value(default_stt_config()).unwrap();
        assert_eq!(value["providerId"], "openai_compatible");
        assert!(value.get("baseUrl").is_some());
    }

    #[test]
    fn vision_model_falls_back_to_the_text_model() {
        // Screenshots must never break just because the override is unset or
        // got blanked out in the form — silently reusing `model` keeps the
        // feature working for everyone who does not care about this field.
        let mut config = default_llm_config();
        assert_eq!(config.effective_vision_model(), config.model);

        config.vision_model = Some("   ".to_string());
        assert_eq!(config.effective_vision_model(), config.model);

        config.vision_model = Some(" claude-opus-5 ".to_string());
        assert_eq!(config.effective_vision_model(), "claude-opus-5");
    }

    #[test]
    fn vision_model_is_omitted_when_unset() {
        // The field is additive: an old config file has no `visionModel`, and a
        // new one must not grow a null that older builds would reject.
        let value = serde_json::to_value(default_llm_config()).unwrap();
        assert!(value.get("visionModel").is_none());
    }
}
