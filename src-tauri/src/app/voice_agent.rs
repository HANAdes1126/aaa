use crate::domain::assistant::AssistantMode;
use crate::providers::llm::{AssistantSuggestion, ChatMessage};
use tauri::AppHandle;

/// Voice-overlay entry point. Defaults to `Interview` because the most common
/// live-situation voice ask is "I'm in an interview, what do I say?" — that
/// mode enforces the "first-person, no meta-advice" persona. The frontend
/// can override via the `mode` argument when the user switches to General
/// (general Q&A) or another mode.
pub async fn complete(
    app: &AppHandle,
    trace_id: String,
    system_prompt: String,
    conversation: Vec<ChatMessage>,
    mode: AssistantMode,
) -> Result<AssistantSuggestion, String> {
    let workflow = workflow_for_mode(mode);
    super::agent_tool_loop::complete(
        app,
        workflow,
        Some(trace_id),
        system_prompt,
        conversation,
    )
    .await
}

/// Maps a user-facing mode to the agent workflow that should handle it. Most
/// live-situation modes route through `MeetingCoach` (stream + cap + low
/// temperature); only `General` keeps the legacy `FnGeneral` workflow so
/// long-form or open-ended voice asks can still hit web_search.
pub(super) fn workflow_for_mode(mode: AssistantMode) -> super::agent_tool_loop::AgentWorkflow {
    match mode {
        AssistantMode::General => super::agent_tool_loop::AgentWorkflow::FnGeneral,
        _ => super::agent_tool_loop::AgentWorkflow::MeetingCoach,
    }
}
