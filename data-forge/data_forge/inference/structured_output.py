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


# ── Quality / Aesthetic Output ──────────────────────────────────────────

class QualityOutput(BaseModel):
    """Structured output from the aesthetic/quality scoring stage."""

    aesthetic_score: float = Field(..., ge=0.0, le=1.0)
    resolution_adequate: bool
    is_complete_ui: bool
    design_era: Literal["legacy", "flat", "modern"]
    issues: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
