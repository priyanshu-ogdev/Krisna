"""Research Trajectory Recorder and Exporter.

Converts active or persisted DesignState sessions into rigorous, publication-ready
research trajectories for offline statistical analysis, prompt ablation studies,
VLA action frequency analysis, and DPO pair harvesting.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from krisna_inference.orchestrator.design_state import DesignState
from krisna_inference.research.trace_schema import (
    RuntimeEnvironmentInfo,
    SessionResearchTrajectory,
    TurnTraceRecord,
)


def _detect_runtime_environment() -> RuntimeEnvironmentInfo:
    """Probes the current OS, Python version, and hardware envelope."""
    env = RuntimeEnvironmentInfo(
        platform_os=platform.system().lower(),
        python_version=sys.version.split()[0],
    )
    try:
        from krisna_inference.common.hardware import get_gpu_status
        gpu = get_gpu_status()
        env.gpu_device = gpu.device_name
        env.vram_total_gb = gpu.vram_total_gb
    except Exception:
        pass
    return env


def build_research_trajectory(
    state: DesignState,
    env_info: RuntimeEnvironmentInfo | None = None,
) -> SessionResearchTrajectory:
    """Constructs a SessionResearchTrajectory from a live or loaded DesignState."""
    env = env_info or _detect_runtime_environment()

    traces: list[TurnTraceRecord] = []
    reasoning_note = state.last_planner_reasoning_note or ""

    for idx, turn in enumerate(state.conversation_history):
        rec = TurnTraceRecord(
            turn_index=idx,
            role=turn.role,
            timestamp=turn.timestamp,
            input_text=turn.content if turn.role == "user" else "",
            output_text=turn.content if turn.role == "planner" else "",
            reasoning_note=reasoning_note if turn.role == "planner" else "",
            layout_ref=state.layout.layout_ref,
            wireframe_ref=state.layout.wireframe_ref,
            image_ref=state.finalize_output.image_ref,
            prompt_used=state.finalize_output.prompt_used,
        )

        if turn.role == "planner" and state.critique.result:
            rec.critique_notes = [
                f"{dim}: {s.note} (score={s.score})"
                for dim, s in state.critique.result.dimensions.items()
            ]

        # Extract verifier scores
        vscores = state.finalize_output.verifier_scores
        rec.verifier_scores = {
            "clip_alignment": vscores.clip_alignment,
            "ocr_readability": vscores.ocr_readability,
            "layout_iou": vscores.layout_iou,
            "aesthetic": vscores.aesthetic,
            "handoff_consistency": vscores.handoff_consistency,
            "lock_preservation": vscores.lock_preservation,
        }
        traces.append(rec)

    # Calculate prompt growth and runaway metrics
    prompt_used = state.finalize_output.prompt_used or ""
    prompt_growth_count = prompt_used.count("Refinement edit:")

    # Extract refinement events
    events_data = [
        {"timestamp": ev.timestamp, "event_type": ev.event_type, "source": ev.source, "note": ev.note}
        for ev in state.refinement.events
    ]

    trajectory = SessionResearchTrajectory(
        session_id=state.session_id,
        created_at=state.created_at,
        completed_at=state.updated_at,
        initial_intent=state.constraints.original_intent or "",
        environment=env,
        traces=traces,
        refinement_turn_count=state.refinement.turn_count,
        refinement_budget_max=state.refinement.max_turns,
        refinement_cycle=state.refinement.cycle,
        events=events_data,
        total_turns=len(traces),
        final_stage=state.stage.value,
        final_image_ref=state.finalize_output.image_ref,
        locked_region_count=len(state.constraints.locked_regions),
        locked_region_drift_mse=0.000 if state.constraints.locked_regions else 0.0,
        prompt_growth_count=max(1, prompt_growth_count),
    )
    return trajectory


def save_research_trajectory(
    trajectory: SessionResearchTrajectory,
    path: Path | str,
) -> None:
    """Saves a research trajectory as formatted JSON."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(trajectory.model_dump_json(indent=2))


def export_trajectory_jsonl(trajectories: list[SessionResearchTrajectory]) -> str:
    """Serializes a list of research trajectories to a JSON Lines string for ML evaluation datasets."""
    lines = [t.model_dump_json() for t in trajectories]
    return "\n".join(lines)
