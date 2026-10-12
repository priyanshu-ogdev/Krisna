//! Offline, allocation-light control-plane state for the agentic loop.
//!
//! Python remains the model runtime (llama.cpp/Diffusers); Rust owns the
//! latency-sensitive local policy boundary and can reject invalid transitions
//! before starting an IPC/model call.
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;

const MAX_TURNS: u8 = 4;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineSnapshot {
    pub revision: u64,
    pub cycle: u64,
    pub turn_count: u8,
    pub max_turns: u8,
    pub stage: String,
    pub last_source: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CommandEnvelope {
    pub command_id: String,
    pub expected_revision: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CommandAck {
    pub command_id: String,
    pub revision: u64,
    pub snapshot: PipelineSnapshot,
}

#[derive(Default)]
pub struct PipelineState(pub Mutex<PipelineSnapshot>);

impl Default for PipelineSnapshot {
    fn default() -> Self {
        Self {
            revision: 0,
            cycle: 0,
            turn_count: 0,
            max_turns: MAX_TURNS,
            stage: "conversing".into(),
            last_source: None,
        }
    }
}

fn begin_command(state: &PipelineSnapshot, command: &CommandEnvelope) -> Result<(), String> {
    if command.command_id.trim().is_empty() {
        return Err("command_id is required".into());
    }
    if command.expected_revision != state.revision {
        return Err(format!(
            "stale pipeline revision: expected {}, current {}",
            command.expected_revision, state.revision
        ));
    }
    Ok(())
}

fn consume(state: &mut PipelineSnapshot, source: &str) -> Result<(), String> {
    if state.turn_count >= state.max_turns {
        return Err("refinement turn budget exhausted".into());
    }
    state.turn_count += 1;
    state.last_source = Some(source.into());
    state.stage = "sketching".into();
    state.revision += 1;
    Ok(())
}

fn ack(state: &PipelineSnapshot, command: &CommandEnvelope) -> CommandAck {
    CommandAck {
        command_id: command.command_id.clone(),
        revision: state.revision,
        snapshot: state.clone(),
    }
}

#[tauri::command]
pub fn pipeline_snapshot(state: State<'_, PipelineState>) -> Result<PipelineSnapshot, String> {
    state
        .0
        .lock()
        .map(|s| s.clone())
        .map_err(|_| "pipeline state poisoned".into())
}

#[tauri::command]
pub fn record_user_turn(
    state: State<'_, PipelineState>,
    command: CommandEnvelope,
) -> Result<CommandAck, String> {
    let mut snapshot = state.0.lock().map_err(|_| "pipeline state poisoned")?;
    begin_command(&snapshot, &command)?;
    consume(&mut snapshot, "user")?;
    Ok(ack(&snapshot, &command))
}

#[tauri::command]
pub fn record_self_review(
    state: State<'_, PipelineState>,
    command: CommandEnvelope,
) -> Result<CommandAck, String> {
    let mut snapshot = state.0.lock().map_err(|_| "pipeline state poisoned")?;
    begin_command(&snapshot, &command)?;
    consume(&mut snapshot, "self_review")?;
    Ok(ack(&snapshot, &command))
}

#[tauri::command]
pub fn begin_revision_cycle(
    state: State<'_, PipelineState>,
    command: CommandEnvelope,
) -> Result<CommandAck, String> {
    let mut snapshot = state.0.lock().map_err(|_| "pipeline state poisoned")?;
    begin_command(&snapshot, &command)?;
    snapshot.cycle += 1;
    snapshot.revision += 1;
    snapshot.turn_count = 0;
    snapshot.last_source = None;
    snapshot.stage = "conversing".into();
    Ok(ack(&snapshot, &command))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enforces_four_turn_budget() {
        let mut state = PipelineSnapshot::default();
        for _ in 0..MAX_TURNS {
            consume(&mut state, "user").unwrap();
        }
        assert!(consume(&mut state, "self_review").is_err());
    }
    #[test]
    fn revision_cycle_resets_budget() {
        let mut state = PipelineSnapshot::default();
        consume(&mut state, "user").unwrap();
        state.cycle += 1;
        state.turn_count = 0;
        consume(&mut state, "self_review").unwrap();
        assert_eq!(state.cycle, 1);
        assert_eq!(state.turn_count, 1);
    }

    #[test]
    fn stale_commands_are_rejected() {
        let state = PipelineSnapshot::default();
        let command = CommandEnvelope {
            command_id: "c1".into(),
            expected_revision: 1,
        };
        assert!(begin_command(&state, &command).is_err());
    }
}
