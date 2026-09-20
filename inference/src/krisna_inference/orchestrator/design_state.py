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
    CRITIQUING = "critiquing"


class ConversationTurn(BaseModel):
    role: Literal["user", "planner"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LockedRegion(BaseModel):
    bbox: tuple[float, float, float, float]  # x, y, w, h
    reason: str


class Constraints(BaseModel):
    style: str | None = None
    palette: list[str] = Field(default_factory=list)  # hex strings
    layout_hints: str | None = None
    locked_regions: list[LockedRegion] = Field(default_factory=list)
    original_intent: str | None = None  # D2: preserves original user intent across long conversations


class SketchTokens(BaseModel):
    vq_tokens: str | None = None          # reference to latest committed token grid
    confidence_map: str | None = None     # reference, Token-Critic output
    revision: int = 0                     # increments per sketch round


class VerifierScores(BaseModel):
    clip_alignment: float | None = None
    ocr_readability: float | None = None
    layout_iou: float | None = None
    aesthetic: float | None = None
    handoff_consistency: float | None = None


RendererUsed = Literal["z_image_turbo", "qwen_image_edit_2511", None]


class FinalizeOutput(BaseModel):
    renderer_used: RendererUsed = None
    image_ref: str | None = None
    verifier_scores: VerifierScores = Field(default_factory=VerifierScores)


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
    source: Literal["gemma4_31b", None] = None
    result: CritiqueResult | None = None
    timestamp: str | None = None


class DesignState(BaseModel):
    """Field-for-field implementation of PRD §5.1."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage: SessionStage = SessionStage.CONVERSING
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    sketch_tokens: SketchTokens = Field(default_factory=SketchTokens)
    finalize_output: FinalizeOutput = Field(default_factory=FinalizeOutput)
    critique: Critique = Field(default_factory=Critique)
    preference_pair_refs: list[str] = Field(default_factory=list)

    # Not in the §5.1 JSON literally, but required for optimistic
    # concurrency and audit — kept out of the model's own `dict()` export
    # used for the wire contract when strict_schema=True (see to_wire()).
    revision: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_wire(self) -> dict[str, Any]:
        """The exact §5.1 shape (no revision/timestamps), for anything
        that needs to match the PRD's JSON contract byte-for-byte."""
        d = self.model_dump(mode="json")
        d.pop("revision", None)
        d.pop("created_at", None)
        d.pop("updated_at", None)
        return d

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.revision += 1

    def append_turn(self, role: Literal["user", "planner"], content: str) -> None:
        self.conversation_history.append(ConversationTurn(role=role, content=content))
        self.touch()

    def is_finalize_eligible(self) -> bool:
        """Finalize requires an active sketch to hand off from."""
        return self.stage in (SessionStage.CONVERSING, SessionStage.SKETCHING) and bool(
            self.sketch_tokens.vq_tokens
        )

    def is_critique_eligible(self) -> bool:
        """Critique operates on a finished render (§5.3: 'evaluates
        finalize_output.image_ref + constraints')."""
        return self.stage == SessionStage.FINALIZED and bool(self.finalize_output.image_ref)
