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


ActionType = Literal[
    "SET_STYLE",
    "RESIZE",
    "ADD_COMPONENT",
    "LOCK_REGION",
    "SET_PALETTE",
    "REMOVE_COMPONENT",
    "MOVE_COMPONENT",
    "SET_TEXT",
    "SET_LAYOUT_DIRECTION",
    "REORDER_HIERARCHY",
    "SET_ASSET_MODE",
]


class DesignAction(BaseModel):
    """Visual-Language-Action (VLA) design action grammar element.

    Research basis:
    - ShowUI (Lin et al., CVPR 2025): Action tokens for direct visual-GUI manipulation.
    - SeeClick (Cheng et al., ACL 2024): Visual-GUI coordinate grounding and localized targeting.
    - InstructLayout (Wen et al., IEEE TPAMI 2025): Instruction-driven conditional UI layout refinement.
    - Design2Code (Si et al., ACL 2025): Multimodal design-to-code structural alignment and element mutation.
    """

    model_config = ConfigDict(extra="allow")

    action_type: ActionType
    target_id: str | None = None
    style: str | None = None
    palette: list[str] | None = None
    bbox: list[float] | None = None
    reason: str | None = None
    component_type: str | None = None
    text_content: str | None = None
    layout_direction: str | None = None
    hierarchy_position: str | None = None
    source_asset_id: str | None = None
    source_crop_bbox: list[float] | None = None
    props: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)


class SetStyleAction(DesignAction):
    action_type: Literal["SET_STYLE"] = "SET_STYLE"


class SetPaletteAction(DesignAction):
    action_type: Literal["SET_PALETTE"] = "SET_PALETTE"


class LockRegionAction(DesignAction):
    action_type: Literal["LOCK_REGION"] = "LOCK_REGION"


class ResizeAction(DesignAction):
    action_type: Literal["RESIZE"] = "RESIZE"


class AddComponentAction(DesignAction):
    action_type: Literal["ADD_COMPONENT"] = "ADD_COMPONENT"


class RemoveComponentAction(DesignAction):
    action_type: Literal["REMOVE_COMPONENT"] = "REMOVE_COMPONENT"


class MoveComponentAction(DesignAction):
    action_type: Literal["MOVE_COMPONENT"] = "MOVE_COMPONENT"


class SetTextAction(DesignAction):
    action_type: Literal["SET_TEXT"] = "SET_TEXT"


class SetLayoutDirectionAction(DesignAction):
    action_type: Literal["SET_LAYOUT_DIRECTION"] = "SET_LAYOUT_DIRECTION"


class ReorderHierarchyAction(DesignAction):
    action_type: Literal["REORDER_HIERARCHY"] = "REORDER_HIERARCHY"


class SetAssetModeAction(DesignAction):
    action_type: Literal["SET_ASSET_MODE"] = "SET_ASSET_MODE"
    asset_mode: Literal["opaque", "transparent"] = "opaque"


class DesignStateDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["conversing", "sketching", "finalizing", "finalized", "critiquing", "refining"]
    constraint_updates: dict[str, Any] = Field(default_factory=dict)
    actions: list[DesignAction] = Field(default_factory=list)
    tool_call: str | None = None
    reasoning_note: str = Field(default="", max_length=500)
    asset_mode: Literal["opaque", "transparent"] | None = None


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
