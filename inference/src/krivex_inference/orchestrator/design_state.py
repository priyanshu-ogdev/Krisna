"""Design State object — PRD §5.1.

This is the object every model tier reads from and writes back into. The
schema below is a direct, field-for-field implementation of §5.1's JSON
shape, with pydantic validation added so a malformed write from any tier
fails loudly at the boundary instead of corrupting the session silently.

`revision` is the optimistic-concurrency guard: every write must supply the
revision it read, and DesignStateStore.save() rejects a stale write (see
exceptions.StaleDesignStateError) rather than silently overwriting a
concurrent update — this matters once the swap orchestrator, the verifier
stack, and the critique adapter can all plausibly want to update the same
session's state around a finalize/critique boundary.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionStage(str, Enum):
    """`stage` field — where this session currently sits in the product flow.
    This is presentation/product-level state, distinct from ResidencyState
    (state_machine.py), which is about which model weights are physically
    loaded on the GPU right now. The two are related but not 1:1 — e.g. a
    session can sit in FINALIZING stage across several residency
    transitions (unload planner+sketch, load polish, generate, unload
    polish, reload planner+sketch) before it becomes FINALIZED.
    """

    CONVERSING = "conversing"
    SKETCHING = "sketching"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    REFINING = "refining"
    CRITIQUING = "critiquing"


class ConversationTurn(BaseModel):
    role: Literal["user", "planner", "system"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReferenceAsset(BaseModel):
    """Multi-asset input descriptor for multi-subject / multi-reference composition (MS-Diffusion & AnyDoor)."""

    id: str  # e.g., "asset_self", "asset_girl", "asset_bg"
    blob_ref: str
    role: Literal["subject", "background", "style", "canvas"] = "subject"
    description: str = ""
    source_crop_bbox: tuple[float, float, float, float] | None = None  # Normalized [x, y, w, h] in source image
    facial_landmarks: dict[str, Any] | None = None


class LockedRegion(BaseModel):
    bbox: tuple[float, float, float, float]  # target canvas normalized x, y, w, h
    reason: str
    id: str | None = None
    source_asset_id: str | None = None  # Points to specific ReferenceAsset (e.g. "asset_self" vs "asset_girl")
    source_crop_bbox: tuple[float, float, float, float] | None = None  # Crop box in source asset [x, y, w, h]


class Constraints(BaseModel):
    style: str | None = None
    palette: list[str] = Field(default_factory=list)  # hex strings
    layout_hints: str | None = None
    locked_regions: list[LockedRegion] = Field(default_factory=list)
    original_intent: str | None = None  # D2: preserves original user intent across long conversations
    reference_image_ref: str | None = None  # Backward-compatible single image reference
    reference_assets: list[ReferenceAsset] = Field(default_factory=list)  # SOTA multi-asset manifest
    asset_mode: Literal["opaque", "transparent"] = "opaque"

    def get_reference_assets(self) -> list[ReferenceAsset]:
        """Resolves active reference assets, ensuring backward-compatibility with single reference_image_ref."""
        if self.reference_assets:
            return self.reference_assets
        if self.reference_image_ref:
            return [ReferenceAsset(id="default", blob_ref=self.reference_image_ref, role="subject")]
        return []


class LayoutDraft(BaseModel):
    layout_ref: str | None = None
    prior_layout_ref: str | None = None
    wireframe_ref: str | None = None
    segmentation_ref: str | None = None
    mask_ref: str | None = None
    revision: int = 0


class VerifierScores(BaseModel):
    clip_alignment: float | None = None
    ocr_readability: float | None = None
    layout_iou: float | None = None
    aesthetic: float | None = None
    handoff_consistency: float | None = None
    lock_preservation: float | None = None
    seam_continuity: float | None = None


RendererUsed = Literal["qwen_image_2_1", "z_image_turbo", "qwen_image_edit_2511", None]


class FinalizeOutput(BaseModel):
    renderer_used: RendererUsed = None
    image_ref: str | None = None
    verifier_scores: VerifierScores = Field(default_factory=VerifierScores)
    evaluation_receipt: dict[str, Any] | None = None
    prompt_used: str | None = None


class PreviewOutput(BaseModel):
    renderer_used: RendererUsed = None
    image_ref: str | None = None
    step_count: int = 6
    latency_seconds: float | None = None
    prompt_used: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CritiqueDimensionScore(BaseModel):
    score: float
    note: str = Field(default="", max_length=280)  # "<= 2 sentences" per §5.2


class CritiqueResult(BaseModel):
    """Mirrors the Critique Adapter contract (§5.2) exactly — this is
    the `critique.result` field's shape."""

    critique_source: str
    overall_score: float
    dimensions: dict[str, CritiqueDimensionScore] = Field(default_factory=dict)
    suggested_edits: list[dict[str, Any]] = Field(default_factory=list)
    raw_model_output_ref: str | None = None


class Critique(BaseModel):
    requested: bool = False
    source: Literal["planner_review", "gemma4_31b", None] = None
    result: CritiqueResult | None = None
    timestamp: str | None = None


class RefinementEvent(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    source: Literal["user", "system", "self_review"]
    note: str = ""


class RefinementBudget(BaseModel):
    turn_count: int = 0
    max_turns: int = 4
    cycle: int = 0
    self_review_scheduled: bool = False
    last_source: Literal["user", "system", "self_review"] | None = None
    events: list[RefinementEvent] = Field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return self.turn_count >= self.max_turns

    def record_turn(self, source: Literal["user", "self_review"]) -> None:
        self.turn_count += 1
        self.last_source = source
        self.self_review_scheduled = False
        self.events.append(RefinementEvent(event_type="turn_consumed", source=source))

    def reset_cycle(self) -> None:
        self.cycle += 1
        self.turn_count = 0
        self.self_review_scheduled = False
        self.last_source = None
        self.events.append(RefinementEvent(event_type="cycle_reset", source="user"))


class DesignState(BaseModel):
    """Field-for-field implementation of PRD §5.1."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage: SessionStage = SessionStage.CONVERSING
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    layout: LayoutDraft = Field(default_factory=LayoutDraft)
    finalize_output: FinalizeOutput = Field(default_factory=FinalizeOutput)
    critique: Critique = Field(default_factory=Critique)
    preference_pair_refs: list[str] = Field(default_factory=list)
    refinement: RefinementBudget = Field(default_factory=RefinementBudget)
    preview_output: PreviewOutput | None = None

    # Not in the §5.1 JSON literally, but required for optimistic
    # concurrency and audit — kept out of the model's own `dict()` export
    # used for the wire contract when strict_schema=True (see to_wire()).
    revision: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    last_planner_reasoning_note: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """The exact §5.1 shape (no revision/timestamps/internal fields),
        for anything that needs to match the PRD's JSON contract
        byte-for-byte."""
        d = self.model_dump(mode="json")
        d.pop("revision", None)
        d.pop("created_at", None)
        d.pop("updated_at", None)
        d.pop("last_planner_reasoning_note", None)
        d.pop("refinement", None)
        d.pop("preview_output", None)
        return d

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.revision += 1

    def append_turn(self, role: Literal["user", "planner", "system"], content: str) -> None:
        self.conversation_history.append(ConversationTurn(role=role, content=content))
        self.touch()

    def record_refinement_event(self, event_type: str, source: Literal["user", "system", "self_review"], note: str = "") -> None:
        self.refinement.events.append(RefinementEvent(event_type=event_type, source=source, note=note))

    def is_finalize_eligible(self) -> bool:
        """Finalize requires an active sketch to hand off from."""
        return self.stage in (SessionStage.CONVERSING, SessionStage.SKETCHING) and bool(
            self.layout.layout_ref
        )

    def is_refine_eligible(self) -> bool:
        """Refinement requires an already-finalized render image to edit from."""
        return (
            self.stage in (SessionStage.FINALIZED, SessionStage.CRITIQUING, SessionStage.REFINING)
            and bool(self.finalize_output.image_ref)
        )

    def is_critique_eligible(self) -> bool:
        """Critique operates on a finished render (§5.3: 'evaluates
        finalize_output.image_ref + constraints')."""
        return self.stage == SessionStage.FINALIZED and bool(self.finalize_output.image_ref)
