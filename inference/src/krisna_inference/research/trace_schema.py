"""Canonical Research Trajectory & Trace Schema for Krisna.

Standardized format for capturing reasoning traces, conversation steps,
VLA actions, spatial states, and verifier evaluations to enable rigorous
offline empirical analysis for publication in top-tier venues (ACM, IEEE,
NeurIPS, ICLR, CVPR, ACL).

Research groundings:
- WebArena (Zhou et al., ICLR 2024): Trajectory recording protocol.
- VisualWebArena (Koh et al., ACL 2024): Multimodal step-by-step traces.
- ShowUI (Lin et al., CVPR 2025): Action-token sequence tracking.
- InstructLayout (Wen et al., IEEE TPAMI 2025): Layout mutation logging.
- DPO for Diffusion (Wallace et al., CVPR 2024): Trajectory preference pairs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TRACE_SCHEMA_VERSION = "krisna.research.trace.v1"


class RuntimeEnvironmentInfo(BaseModel):
    """Execution environment details for scientific reproducibility."""

    model_config = ConfigDict(extra="allow")

    platform_os: str = "windows"
    python_version: str = ""
    gpu_device: str | None = None
    vram_total_gb: float | None = None
    envelope_gb: float = 24.0
    low_vram_mode: bool = False
    strict_mode: bool = False


class TurnTraceRecord(BaseModel):
    """Granular trace record for a single conversation/reasoning step."""

    model_config = ConfigDict(extra="allow")

    turn_index: int
    role: Literal["user", "planner", "system", "critic"]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    input_text: str = ""
    output_text: str = ""

    # Reasoning & Chain-of-Thought traces
    reasoning_note: str = ""
    thought_steps: list[str] = Field(default_factory=list)
    critique_notes: list[str] = Field(default_factory=list)

    # VLA design actions executed in this turn
    actions: list[dict[str, Any]] = Field(default_factory=list)

    # State transitions & constraint deltas
    stage_before: str = ""
    stage_after: str = ""
    constraint_updates: dict[str, Any] = Field(default_factory=dict)

    # Artifact pointers
    layout_ref: str | None = None
    wireframe_ref: str | None = None
    image_ref: str | None = None
    prompt_used: str | None = None

    # Verifier & performance diagnostics
    verifier_scores: dict[str, float | None] = Field(default_factory=dict)
    latency_seconds: float | None = None
    peak_vram_gb: float | None = None
    solver_metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResearchTrajectory(BaseModel):
    """Complete research trajectory containing session metadata, turns, and cumulative metrics."""

    model_config = ConfigDict(extra="allow")

    schema_version: Literal[TRACE_SCHEMA_VERSION] = TRACE_SCHEMA_VERSION
    session_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    initial_intent: str = ""

    # Environment & hardware profile
    environment: RuntimeEnvironmentInfo = Field(default_factory=RuntimeEnvironmentInfo)

    # Chronological turns & traces
    traces: list[TurnTraceRecord] = Field(default_factory=list)

    # Refinement & budget events
    refinement_turn_count: int = 0
    refinement_budget_max: int = 4
    refinement_cycle: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)

    # Cumulative empirical metrics for research paper evaluation
    total_turns: int = 0
    final_stage: str = ""
    final_image_ref: str | None = None
    cumulative_latency_seconds: float = 0.0
    locked_region_count: int = 0
    locked_region_drift_mse: float = 0.000
    prompt_growth_count: int = 1

    # Associated preference pairs for DPO training
    preference_pair_ids: list[str] = Field(default_factory=list)
