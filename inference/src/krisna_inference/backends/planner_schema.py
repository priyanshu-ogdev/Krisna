"""Versioned Planner output contract.

The Planner is frozen; this module owns the product contract around it.  Model
or runtime changes must not change the state-update semantics.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "planner.delta.v1"


class ConstraintDelta(BaseModel):
    model_config = ConfigDict(extra="allow")


class DesignStateDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["conversing", "sketching", "finalizing", "finalized", "critiquing"]
    constraint_updates: dict[str, Any] = Field(default_factory=dict)
    tool_call: str | None = None
    reasoning_note: str = Field(default="", max_length=500)


class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    reply_text: str = ""
    design_state_delta: DesignStateDelta
    retrieved_critique_ids: list[str] = Field(default_factory=list)
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)


def schema_json() -> dict[str, Any]:
    """Return the JSON schema used by llama.cpp/vLLM constrained decoding."""
    return PlannerResponse.model_json_schema()
