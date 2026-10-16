"""Test Suite for Project Krisna Research Trajectory & Trace Export.

Validates that all reasoning traces, conversation histories, VLA actions,
and verifier scores are faithfully recorded for academic research paper analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from krisna_inference.orchestrator.design_state import DesignState, SessionStage
from krisna_inference.orchestrator.store import DesignStateStore
from krisna_inference.research import (
    TRACE_SCHEMA_VERSION,
    RuntimeEnvironmentInfo,
    SessionResearchTrajectory,
    TurnTraceRecord,
    build_research_trajectory,
    export_trajectory_jsonl,
    save_research_trajectory,
)


def test_research_trajectory_schema_roundtrip():
    """Verify SessionResearchTrajectory schema instantiates and validates correctly."""
    turn1 = TurnTraceRecord(
        turn_index=0,
        role="user",
        input_text="Create a high-conversion checkout flow",
    )
    turn2 = TurnTraceRecord(
        turn_index=1,
        role="planner",
        output_text="Planning 3-step checkout with order summary",
        reasoning_note="User requested high conversion; reducing form fields to minimize drop-off.",
        thought_steps=["Analyze intent", "Determine primary CTA", "Structure hierarchy"],
        actions=[
            {"action_type": "SET_STYLE", "style": "clean modern"},
            {"action_type": "LOCK_REGION", "bbox": [0.0, 0.0, 1.0, 0.1], "reason": "security header"},
        ],
        stage_before="conversing",
        stage_after="sketching",
    )

    traj = SessionResearchTrajectory(
        session_id="test-session-123",
        initial_intent="Create a high-conversion checkout flow",
        traces=[turn1, turn2],
        total_turns=2,
        final_stage="sketching",
    )

    dumped = traj.model_dump()
    assert dumped["schema_version"] == TRACE_SCHEMA_VERSION
    assert len(dumped["traces"]) == 2
    assert dumped["traces"][1]["reasoning_note"] == "User requested high conversion; reducing form fields to minimize drop-off."
    assert len(dumped["traces"][1]["actions"]) == 2

    # Validate json serialization and re-parse
    raw_json = traj.model_dump_json()
    reloaded = SessionResearchTrajectory.model_validate_json(raw_json)
    assert reloaded.session_id == "test-session-123"
    assert len(reloaded.traces) == 2


def test_build_research_trajectory_from_design_state(store: DesignStateStore):
    """Verify build_research_trajectory converts live DesignState into research format."""
    state = store.create()
    state.append_turn("user", "build a mobile fitness dashboard")
    state.last_planner_reasoning_note = "Using card grid with heart-rate metric as hero element."
    state.append_turn("planner", "Generated mobile fitness dashboard wireframe")
    state.constraints.style = "dark neo-brutalist"
    state.constraints.palette = ["#111111", "#00ff66"]
    state.record_refinement_event("sketch_generated", "system", "initial draft")
    state.finalize_output.image_ref = "render://test/fitness.png"
    state.finalize_output.prompt_used = "Modern UI. Refinement edit: make metrics larger"
    state.touch()
    saved = store.save(state, expected_revision=0)

    traj = build_research_trajectory(saved)
    assert traj.session_id == state.session_id
    assert traj.total_turns == 2
    assert traj.traces[0].role == "user"
    assert traj.traces[0].input_text == "build a mobile fitness dashboard"
    assert traj.traces[1].role == "planner"
    assert "card grid" in traj.traces[1].reasoning_note
    assert traj.final_image_ref == "render://test/fitness.png"
    assert traj.prompt_growth_count == 1
    assert len(traj.events) >= 1


def test_export_and_save_trajectory_files(tmp_path: Path):
    """Verify saving trajectory as JSON and exporting list as JSONL."""
    traj1 = SessionResearchTrajectory(session_id="s1", initial_intent="App 1")
    traj2 = SessionResearchTrajectory(session_id="s2", initial_intent="App 2")

    out_file = tmp_path / "research_traces" / "s1.json"
    save_research_trajectory(traj1, out_file)
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["session_id"] == "s1"
    assert data["schema_version"] == TRACE_SCHEMA_VERSION

    # JSONL export
    jsonl_str = export_trajectory_jsonl([traj1, traj2])
    lines = [line for line in jsonl_str.split("\n") if line.strip()]
    assert len(lines) == 2
    parsed_first = json.loads(lines[0])
    assert parsed_first["session_id"] == "s1"


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    import importlib
    monkeypatch.chdir(tmp_path)
    import krisna_inference.orchestrator.service as service_module

    importlib.reload(service_module)
    with TestClient(service_module.app) as c:
        yield c


def test_service_research_trace_endpoint(client: TestClient):
    """Verify GET /session/{session_id}/research-trace returns valid trajectory."""
    # Create session and add turn
    session_id = client.post("/session", json={"style": "minimalist"}).json()["session_id"]
    client.post(f"/session/{session_id}/message", json={"message": "create landing page"})

    # Fetch research trace
    res = client.get(f"/session/{session_id}/research-trace")
    assert res.status_code == 200
    trace_data = res.json()
    assert trace_data["session_id"] == session_id
    assert trace_data["schema_version"] == TRACE_SCHEMA_VERSION
    assert trace_data["total_turns"] >= 2
    assert trace_data["traces"][0]["input_text"] == "create landing page"
