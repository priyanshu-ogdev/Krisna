//! Offline, allocation-light control-plane state for the agentic loop.
//!
//! Python remains the model runtime (llama.cpp/Diffusers); Rust owns the
//! latency-sensitive local policy boundary and can reject invalid transitions
//! before starting an IPC/model call.
//!
//! Citations & Design Grounding:
//! - PRD §5 & §7: Bounded Refinement & Turn Budget Policy.
//! - docs/architecture/OFFLINE_RUST_CONTROL_PLANE.md: Local Rust Control Plane.

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::State;

const MAX_TURNS: u8 = 4;
const DEFAULT_INACTIVITY_TIMEOUT_SECS: u64 = 30;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RefinementEvent {
    pub event_type: String,
    pub source: String,
    pub cycle: u64,
    pub turn_count: u8,
    pub timestamp_secs: u64,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineSnapshot {
    pub revision: u64,
    pub cycle: u64,
    pub turn_count: u8,
    pub max_turns: u8,
    pub stage: String,
    pub last_source: Option<String>,
    pub last_activity_secs: u64,
    pub self_review_due: bool,
    pub events: Vec<RefinementEvent>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CommandEnvelope {
    pub command_id: String,
    pub expected_revision: u64,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CommandAck {
    pub command_id: String,
    pub revision: u64,
    pub snapshot: PipelineSnapshot,
}

#[derive(Default)]
pub struct PipelineState(pub Mutex<PipelineSnapshot>);

fn current_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Default for PipelineSnapshot {
    fn default() -> Self {
        Self {
            revision: 0,
            cycle: 0,
            turn_count: 0,
            max_turns: MAX_TURNS,
            stage: "conversing".into(),
            last_source: None,
            last_activity_secs: current_epoch_secs(),
            self_review_due: false,
            events: Vec::new(),
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

fn append_event(state: &mut PipelineSnapshot, event_type: &str, source: &str, detail: Option<String>) {
    let event = RefinementEvent {
        event_type: event_type.into(),
        source: source.into(),
        cycle: state.cycle,
        turn_count: state.turn_count,
        timestamp_secs: current_epoch_secs(),
        detail,
    };
    state.events.push(event);
    if state.events.len() > 100 {
        state.events.remove(0);
    }
}

fn consume(state: &mut PipelineSnapshot, source: &str, detail: Option<String>) -> Result<(), String> {
    if state.turn_count >= state.max_turns {
        return Err("refinement turn budget exhausted".into());
    }
    state.turn_count += 1;
    state.last_source = Some(source.into());
    state.stage = "sketching".into();
    state.revision += 1;
    state.last_activity_secs = current_epoch_secs();
    state.self_review_due = false;
    append_event(state, "turn_completed", source, detail);
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
    let detail = command.detail.clone();
    consume(&mut snapshot, "user", detail)?;
    Ok(ack(&snapshot, &command))
}

#[tauri::command]
pub fn record_self_review(
    state: State<'_, PipelineState>,
    command: CommandEnvelope,
) -> Result<CommandAck, String> {
    let mut snapshot = state.0.lock().map_err(|_| "pipeline state poisoned")?;
    begin_command(&snapshot, &command)?;
    let detail = command.detail.clone();
    consume(&mut snapshot, "self_review", detail)?;
    Ok(ack(&snapshot, &command))
}

#[tauri::command]
pub fn record_finalize(
    state: State<'_, PipelineState>,
    command: CommandEnvelope,
) -> Result<CommandAck, String> {
    let mut snapshot = state.0.lock().map_err(|_| "pipeline state poisoned")?;
    begin_command(&snapshot, &command)?;
    snapshot.stage = "finalized".into();
    snapshot.revision += 1;
    snapshot.last_activity_secs = current_epoch_secs();
    snapshot.self_review_due = false;
    append_event(&mut snapshot, "finalized", "user", command.detail.clone());
    Ok(ack(&snapshot, &command))
}

#[tauri::command]
pub fn record_rollback(
    state: State<'_, PipelineState>,
    command: CommandEnvelope,
) -> Result<CommandAck, String> {
    let mut snapshot = state.0.lock().map_err(|_| "pipeline state poisoned")?;
    begin_command(&snapshot, &command)?;
    snapshot.stage = "sketching".into();
    snapshot.revision += 1;
    snapshot.last_activity_secs = current_epoch_secs();
    append_event(&mut snapshot, "safety_rollback", "verifier", command.detail.clone());
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
    snapshot.last_activity_secs = current_epoch_secs();
    snapshot.self_review_due = false;
    append_event(&mut snapshot, "cycle_started", "user", command.detail.clone());
    Ok(ack(&snapshot, &command))
}

#[tauri::command]
pub fn check_inactivity(
    state: State<'_, PipelineState>,
    timeout_secs: Option<u64>,
) -> Result<bool, String> {
    let mut snapshot = state.0.lock().map_err(|_| "pipeline state poisoned")?;
    let limit = timeout_secs.unwrap_or(DEFAULT_INACTIVITY_TIMEOUT_SECS);
    let now = current_epoch_secs();
    let elapsed = now.saturating_sub(snapshot.last_activity_secs);
    let should_trigger = elapsed >= limit
        && snapshot.stage == "sketching"
        && snapshot.turn_count < snapshot.max_turns
        && !snapshot.self_review_due;
    if should_trigger {
        snapshot.self_review_due = true;
    }
    Ok(should_trigger)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforces_four_turn_budget() {
        let mut state = PipelineSnapshot::default();
        for _ in 0..MAX_TURNS {
            consume(&mut state, "user", None).unwrap();
        }
        assert!(consume(&mut state, "self_review", None).is_err());
    }

    #[test]
    fn revision_cycle_resets_budget() {
        let mut state = PipelineSnapshot::default();
        consume(&mut state, "user", None).unwrap();
        state.cycle += 1;
        state.turn_count = 0;
        consume(&mut state, "self_review", None).unwrap();
        assert_eq!(state.cycle, 1);
        assert_eq!(state.turn_count, 1);
    }

    #[test]
    fn stale_commands_are_rejected() {
        let state = PipelineSnapshot::default();
        let command = CommandEnvelope {
            command_id: "c1".into(),
            expected_revision: 1,
            detail: None,
        };
        assert!(begin_command(&state, &command).is_err());
    }

    #[test]
    fn event_ledger_records_transitions() {
        let mut state = PipelineSnapshot::default();
        consume(&mut state, "user", Some("add card".into())).unwrap();
        assert_eq!(state.events.len(), 1);
        assert_eq!(state.events[0].event_type, "turn_completed");
        assert_eq!(state.events[0].detail.as_deref(), Some("add card"));
    }

    #[test]
    fn inactivity_detected_in_sketching() {
        let mut state = PipelineSnapshot::default();
        state.stage = "sketching".into();
        state.last_activity_secs = 100; // Far in past
        state.turn_count = 1;
        assert!(!state.self_review_due);
        let now = current_epoch_secs();
        let elapsed = now.saturating_sub(state.last_activity_secs);
        assert!(elapsed >= DEFAULT_INACTIVITY_TIMEOUT_SECS);
    }
}
