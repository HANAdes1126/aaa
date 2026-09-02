mod agent_tool_loop;
pub mod assistant_service;
mod coach_agent;
pub mod document_service;
pub mod knowledge;
pub mod prompt_orchestrator;
pub mod report_service;
pub mod screen_capture;
mod voice_agent;

#[cfg(test)]
mod tests {
    use super::agent_tool_loop::AgentWorkflow;
    use crate::domain::assistant::AssistantMode;

    #[test]
    fn coach_and_voice_interview_use_the_same_meeting_coach_workflow() {
        // Both agents must funnel Interview / Meeting / Sales voice asks
        // through MeetingCoach so the voice tuning (stream + cap + low
        // temperature) applies.
        assert_eq!(
            super::coach_agent::workflow(),
            super::voice_agent::workflow_for_mode(AssistantMode::Interview)
        );
    }

    #[test]
    fn voice_general_mode_uses_the_distinct_fn_general_workflow() {
        // General voice asks should still go through the legacy FnGeneral
        // path so web_search is available for long-form / open-ended questions.
        assert_eq!(
            super::voice_agent::workflow_for_mode(AssistantMode::General),
            AgentWorkflow::FnGeneral
        );
        assert_ne!(
            super::voice_agent::workflow_for_mode(AssistantMode::General),
            super::coach_agent::workflow()
        );
    }
}
