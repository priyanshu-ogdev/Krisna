"""Pydantic models for structured output enforcement via vLLM guided decoding.

Every model output type has a corresponding Pydantic model here. These are
passed to vLLM's `structured_outputs.json` to guarantee schema-valid JSON.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Caption Output ──────────────────────────────────────────────────────

class CaptionOutput(BaseModel):
    """Structured output from the recaptioning stage."""

    caption: str = Field(
        ..., min_length=20, max_length=2000,
        description="Detailed description of the UI screenshot"
    )
    ui_elements_mentioned: list[str] = Field(
        ..., min_length=1,
        description="UI element types referenced in the caption"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence that the caption accurately describes the image"
    )


# ── Structure Output ────────────────────────────────────────────────────

class UIElement(BaseModel):
    """A single UI element in the component tree."""

    type: str = Field(..., description="Component type (button, text_input, etc.)")
    bbox: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Bounding box [x_min, y_min, x_max, y_max] normalized to 0-1"
    )
    label: str | None = Field(None, description="Brief description of the element")
    children: list[UIElement] = Field(
        default_factory=list, description="Nested child elements"
    )


# Rebuild to resolve forward reference
UIElement.model_rebuild()


class StructureOutput(BaseModel):
    """Structured output from the structural extraction stage."""

    elements: list[UIElement] = Field(..., description="UI component tree")
    layout_type: Literal[
        "grid", "list", "form", "navigation", "dashboard",
        "detail", "modal", "split", "tabbed", "freeform"
    ] = Field(..., description="Overall layout pattern")
    hierarchy_depth: int = Field(..., ge=1, description="Max nesting depth")
    background_style: Literal[
        "solid_light", "solid_dark", "gradient", "image", "blurred"
    ] | None = Field(None, description="Background visual style")


# ── Safety Output ───────────────────────────────────────────────────────

class SafetyOutput(BaseModel):
    """Structured output from the safety classification stage."""

    tier: Literal["safe", "borderline", "unsafe"] = Field(
        ..., description="Safety classification tier"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(
        ..., min_length=10,
        description="Explanation of the classification"
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Specific concern categories detected"
    )


# ── License Output ──────────────────────────────────────────────────────

class LicenseOutput(BaseModel):
    """Structured output from the license verification agent."""

    license_type: str = Field(..., description="License name or category")
    redistribution_allowed: bool = Field(False)
    commercial_use_allowed: bool = Field(False)
    attribution_required: bool = Field(False)
    research_only: bool = Field(False)
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_citation: str = Field(
        ..., min_length=10,
        description="Quoted text from the source document"
    )
    key_restrictions: list[str] = Field(default_factory=list)
    summary: str = Field("", description="Plain-English summary")


# ── Audit Output ────────────────────────────────────────────────────────

class AuditCaptionOutput(BaseModel):
    """Audit result for caption accuracy verification."""

    caption_matches_image: bool
    accuracy_issues: list[str] = Field(default_factory=list)
    completeness_issues: list[str] = Field(default_factory=list)
    hallucination_issues: list[str] = Field(default_factory=list)
    overall_pass: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=10)


class AuditStructureOutput(BaseModel):
    """Audit result for structural JSON accuracy verification."""

    structure_matches_image: bool
    missing_elements: list[str] = Field(default_factory=list)
    phantom_elements: list[str] = Field(default_factory=list)
    bbox_accuracy: Literal["good", "approximate", "poor"]
    layout_type_correct: bool
    overall_pass: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=10)


class AuditOutput(BaseModel):
    """Combined audit output for Stage 10."""

    caption_matches_image: bool
    structure_matches_image: bool
    quality_issues: list[str] = Field(default_factory=list)
    safety_issues: list[str] = Field(default_factory=list)
    accuracy_issues: list[str] = Field(default_factory=list)
    hallucination_issues: list[str] = Field(default_factory=list)
    overall_pass: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=10)


# ── OCR Output ──────────────────────────────────────────────────────────

class TextRegion(BaseModel):
    """A single text region extracted by OCR."""

    text: str
    bbox: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Bounding box normalized to 0-1"
    )
    role: Literal[
        "heading", "body", "button_label", "input_placeholder",
        "menu_item", "tab_label", "status_text", "caption",
        "tooltip", "error_message", "notification", "other"
    ]
    font_size_class: Literal["small", "medium", "large", "xlarge"] | None = None


class OCROutput(BaseModel):
    """Structured output from the OCR extraction stage."""

    text_regions: list[TextRegion] = Field(default_factory=list)
    primary_language: str = Field("en", min_length=2, max_length=5)
    total_text_regions: int = Field(0, ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)


# ── Critique Output (Critic Tier / v10 PRD §5.2 Critique Adapter) ───────

class CritiqueDimension(BaseModel):
    """A single scored dimension within a UI critique."""

    score: float = Field(..., ge=0.0, le=1.0)
    note: str = Field(..., max_length=400, description="<= 2 sentences")


class CritiqueOutput(BaseModel):
    """Structured output from the Critic Tier (Gemma 4 31B).

    Field shape deliberately mirrors the PRD's Critique Adapter contract
    (§5.2) so a row written here needs no reshaping to match what the
    product's own preference-pair store expects — `critique_source` is the
    only field that distinguishes this from a UICrit-derived or
    verifier-stack-derived critique row.
    """

    overall_score: float = Field(..., ge=0.0, le=1.0)
    visual_hierarchy_score: float = Field(..., ge=0.0, le=1.0)
    visual_hierarchy_note: str = Field(..., max_length=400)
    readability_score: float = Field(..., ge=0.0, le=1.0)
    readability_note: str = Field(..., max_length=400)
    layout_consistency_score: float = Field(..., ge=0.0, le=1.0)
    layout_consistency_note: str = Field(..., max_length=400)
    brand_alignment_score: float = Field(..., ge=0.0, le=1.0)
    brand_alignment_note: str = Field(..., max_length=400)
    suggested_edits: list[str] = Field(
        default_factory=list,
        description="Short, concrete edit instructions, not full sentences",
    )


# ── Planner Conversation Synthesis (Design State Delta) ─────────────────
# Mirrors the PRD's Design State object (§5.1) closely enough that a
# generated delta here is directly usable as a real training example for
# the Planner's tool-call format, without a translation layer between
# what data-forge produces and what the product's Design State Manager
# actually consumes.

class DesignStateDelta(BaseModel):
    """One simulated planner turn's structured JSON output.

    This is intentionally a *subset* of the full Design State object
    (constraints + stage only) — a single conversational turn updates a
    slice of state, not the whole object at once, same as the real
    planner would emit incrementally rather than re-writing the entire
    design_state on every turn.
    """

    stage: Literal["conversing", "sketching", "finalizing", "finalized", "critiquing"]
    constraint_updates: dict[str, str] = Field(
        default_factory=dict,
        description="Partial updates to style/palette/layout_hints, as plain key:value strings",
    )
    tool_call: str | None = Field(
        default=None,
        description="Name of a tool invoked this turn, if any (e.g. 'update_sketch', 'finalize')",
    )
    reasoning_note: str = Field(..., max_length=300, description="<= 2 sentences, why this delta")


class SynthesizedConversationTurn(BaseModel):
    role: Literal["user", "planner"]
    content: str = Field(..., max_length=800)


class SynthesizedConversation(BaseModel):
    """One full generated training example: turns + the final state delta."""

    turns: list[SynthesizedConversationTurn] = Field(..., min_length=2, max_length=8)
    resulting_delta: DesignStateDelta
    seed_critique_record_id: str = Field(
        ..., description="Which real UICrit-sourced critique this was seeded from — "
                          "keeps every synthetic example traceable to real content, "
                          "not fully hallucinated from nothing"
    )


# ── Quality / Aesthetic Output ──────────────────────────────────────────

class QualityOutput(BaseModel):
    """Structured output from the aesthetic/quality scoring stage."""

    aesthetic_score: float = Field(..., ge=0.0, le=1.0)
    resolution_adequate: bool
    is_complete_ui: bool
    design_era: Literal["legacy", "flat", "modern"]
    issues: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
