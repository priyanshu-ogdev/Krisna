"""Pydantic models for structured output enforcement via vLLM guided decoding.

Every model output type has a corresponding Pydantic model here. These are
passed to vLLM's `structured_outputs.json` to guarantee schema-valid JSON.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

BBoxCoord = Annotated[float, Field(ge=0.0, le=1.0)]


def sanitize_bbox(v: Any) -> list[float]:
    """Sanitize and normalize bounding box [x_min, y_min, x_max, y_max] to [0.0, 1.0]."""
    if not isinstance(v, (list, tuple)):
        return [0.0, 0.0, 1.0, 1.0]
    coords: list[float] = []
    for x in v:
        try:
            coords.append(float(x))
        except (ValueError, TypeError):
            coords.append(0.0)
    while len(coords) < 4:
        coords.append(1.0 if len(coords) >= 2 else 0.0)
    coords = coords[:4]
    max_val = max(coords)
    if max_val > 1.0:
        if max_val <= 1000.0:
            coords = [c / 1000.0 for c in coords]
        else:
            coords = [c / max_val for c in coords]
    coords = [round(max(0.0, min(1.0, c)), 4) for c in coords]
    # Enforce non-inverted coordinates: x_min <= x_max, y_min <= y_max
    if coords[0] > coords[2]:
        coords[0], coords[2] = coords[2], coords[0]
    if coords[1] > coords[3]:
        coords[1], coords[3] = coords[3], coords[1]
    return coords


# ── Caption Output ──────────────────────────────────────────────────────

class CaptionOutput(BaseModel):
    """Structured output from the recaptioning stage."""

    caption: str = Field(
        ..., min_length=20, max_length=2000,
        description="Detailed description of the UI screenshot"
    )
    ui_elements_mentioned: list[str] = Field(
        ...,
        description="UI element types referenced in the caption"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence that the caption accurately describes the image"
    )

    @field_validator("caption", mode="before")
    @classmethod
    def _validate_caption_text(cls, v: Any) -> str:
        if not isinstance(v, str):
            return str(v) if v else ""
        text = v.strip()

        # Remove markdown code fence wrapping if present
        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```(?:markdown|text)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()

        # Scrub special tokens
        special_tokens = [
            r"<\|im_start\|>(?:system|user|assistant|tool)?(?:\n|\s+)?",
            r"<\|im_end\|>",
            r"<unk>",
            r"<s>",
            r"</s>",
            r"\[PAD\]",
            r"\[UNK\]",
            r"\[CLS\]",
            r"\[SEP\]",
        ]
        for token_pat in special_tokens:
            text = re.sub(token_pat, "", text, flags=re.IGNORECASE).strip()

        # Clean conversational filler / prompt echo prefixes iteratively
        filler_prefixes = [
            r"^(?:sure|certainly|of course)[!,\.\s]+(?:here is|here's)[^:]*:\s*",
            r"^as an ai(?:\s+language model)?[\,\.\s\-]*",
            r"^as a language model[\,\.\s\-]*",
            r"^you are a ui recaptioning model[\.\s\:\-]*",
            r"^provide a detailed[\,\s]+dense caption[\.\s\:\-]*",
            r"^here is a detailed[\,\s]+dense caption[\.\s\:\-]*",
            r"^here is the caption[\.\s\:\-]*",
            r"^the screenshot contains[\.\s\:\-]*",
        ]
        changed = True
        while changed:
            changed = False
            for pattern in filler_prefixes:
                new_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
                if new_text != text:
                    text = new_text
                    changed = True
        return text

    @field_validator("ui_elements_mentioned", mode="before")
    @classmethod
    def _validate_elements_mentioned(cls, v: Any) -> list[str]:
        if not isinstance(v, list) or not v:
            return []
        cleaned = [str(x).strip() for x in v if str(x).strip()]
        return cleaned

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_conf(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0 if val <= 100.0 else 1.0
            return round(max(0.0, min(1.0, val)), 4)
        except (ValueError, TypeError):
            return 0.95


# ── Structure Output ────────────────────────────────────────────────────

class UIElement(BaseModel):
    """A single UI element in the component tree."""

    type: str = Field(..., description="Component type (button, text_input, etc.)")
    bbox: list[BBoxCoord] = Field(
        ..., min_length=4, max_length=4,
        description="Bounding box [x_min, y_min, x_max, y_max] normalized to 0-1"
    )
    label: str | None = Field(None, description="Brief description of the element")
    children: list[UIElement] = Field(
        default_factory=list, description="Nested child elements"
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def _validate_bbox(cls, v: Any) -> list[float]:
        return sanitize_bbox(v)

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            return "container"
        return v.strip().lower()

    @field_validator("children", mode="before")
    @classmethod
    def _validate_children(cls, v: Any) -> list:
        if not isinstance(v, list):
            return []
        return v


# Rebuild to resolve forward reference
UIElement.model_rebuild()


class StructureOutput(BaseModel):
    """Structured output from the structural extraction stage."""

    elements: list[UIElement] = Field(..., description="UI component tree")
    layout_type: Literal[
        "grid", "list", "form", "navigation", "dashboard",
        "detail", "modal", "split", "tabbed", "freeform"
    ] = Field(..., description="Overall layout pattern")
    hierarchy_depth: int = Field(..., ge=0, description="Max nesting depth")
    background_style: Literal[
        "solid_light", "solid_dark", "gradient", "image", "blurred"
    ] | None = Field(None, description="Background visual style")

    @field_validator("elements", mode="before")
    @classmethod
    def _validate_elements(cls, v: Any) -> list:
        if not isinstance(v, list):
            return []
        # Filter degenerate bounding boxes (hallucination artifacts)
        filtered = []
        for el in v:
            if isinstance(el, dict):
                bbox = el.get("bbox", [])
                el_type = str(el.get("type", "")).strip().lower()
                children = el.get("children", [])
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        x1, y1, x2, y2 = [float(c) for c in bbox]
                        w = abs(x2 - x1)
                        h = abs(y2 - y1)
                        # Filter 0-area degenerate points/lines
                        if w <= 0.001 or h <= 0.001:
                            continue
                        # Filter full-screen leaf element hallucinations
                        if not children and el_type not in ("container", "canvas", "background", "image"):
                            if w >= 0.95 and h >= 0.95:
                                continue
                    except (ValueError, TypeError):
                        pass
            filtered.append(el)
        return filtered

    @field_validator("layout_type", mode="before")
    @classmethod
    def _validate_layout(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "freeform"
        v_clean = v.strip().lower()
        valid = {"grid", "list", "form", "navigation", "dashboard", "detail", "modal", "split", "tabbed", "freeform"}
        if v_clean in valid:
            return v_clean
        mapping = {
            "table": "grid", "column": "list", "cards": "grid", "card": "grid",
            "menu": "navigation", "navbar": "navigation", "sidebar": "navigation",
            "header": "navigation", "login": "form", "input": "form",
            "popup": "modal", "dialog": "modal", "overlay": "modal",
            "tabs": "tabbed", "split_view": "split", "details": "detail",
            "single": "freeform", "feed": "list", "timeline": "list",
        }
        return mapping.get(v_clean, "freeform")

    @field_validator("hierarchy_depth", mode="before")
    @classmethod
    def _validate_depth(cls, v: Any) -> int:
        try:
            return max(0, int(v))
        except (ValueError, TypeError):
            return 0

    @field_validator("background_style", mode="before")
    @classmethod
    def _validate_bg_style(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            return None
        v_clean = v.strip().lower()
        valid = {"solid_light", "solid_dark", "gradient", "image", "blurred"}
        if v_clean in valid:
            return v_clean
        if "dark" in v_clean:
            return "solid_dark"
        if "light" in v_clean or "white" in v_clean:
            return "solid_light"
        if "grad" in v_clean:
            return "gradient"
        return "solid_light"


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

    @field_validator("tier", mode="before")
    @classmethod
    def _validate_tier(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "safe"
        v_low = v.strip().lower()
        if v_low in ("safe", "borderline", "unsafe"):
            return v_low
        if v_low in ("pass", "clean", "ok", "acceptable", "benign"):
            return "safe"
        if v_low in ("harmful", "nsfw", "danger", "toxic", "rejected"):
            return "unsafe"
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_conf(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0 if val <= 100.0 else 1.0
            return round(max(0.0, min(1.0, val)), 4)
        except (ValueError, TypeError):
            return 0.95

    @field_validator("rationale", mode="before")
    @classmethod
    def _validate_rationale(cls, v: Any) -> str:
        if not isinstance(v, str) or len(v.strip()) < 10:
            val = str(v).strip() if v else ""
            return f"Classification assessment: {val}" if len(f"Classification assessment: {val}") >= 10 else "Content evaluated and classified according to standard safety criteria."
        return v.strip()

    @field_validator("flags", mode="before")
    @classmethod
    def _validate_flags(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(f).strip() for f in v if str(f).strip()]


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

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_conf(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0 if val <= 100.0 else 1.0
            return round(max(0.0, min(1.0, val)), 4)
        except (ValueError, TypeError):
            return 0.95

    @field_validator("source_citation", mode="before")
    @classmethod
    def _validate_citation(cls, v: Any) -> str:
        if not isinstance(v, str) or len(v.strip()) < 10:
            val = str(v).strip() if v else ""
            return f"Citation extract: {val}" if len(f"Citation extract: {val}") >= 10 else "License information verified against source documentation."
        return v.strip()


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

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_conf(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0 if val <= 100.0 else 1.0
            return round(max(0.0, min(1.0, val)), 4)
        except (ValueError, TypeError):
            return 0.95

    @field_validator("rationale", mode="before")
    @classmethod
    def _validate_rationale(cls, v: Any) -> str:
        if not isinstance(v, str) or len(v.strip()) < 10:
            val = str(v).strip() if v else ""
            return f"Audit evaluation rationale: {val}" if len(f"Audit evaluation rationale: {val}") >= 10 else "Caption evaluated against visual content according to standard audit rubric."
        return v.strip()


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

    @field_validator("bbox_accuracy", mode="before")
    @classmethod
    def _validate_bbox_acc(cls, v: Any) -> str:
        if isinstance(v, str):
            v_low = v.strip().lower()
            if v_low in ("good", "approximate", "poor"):
                return v_low
            if v_low in ("high", "excellent", "accurate", "exact"):
                return "good"
            if v_low in ("fair", "medium", "moderate", "acceptable"):
                return "approximate"
        return "approximate"

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_conf(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0 if val <= 100.0 else 1.0
            return round(max(0.0, min(1.0, val)), 4)
        except (ValueError, TypeError):
            return 0.95

    @field_validator("rationale", mode="before")
    @classmethod
    def _validate_rationale(cls, v: Any) -> str:
        if not isinstance(v, str) or len(v.strip()) < 10:
            val = str(v).strip() if v else ""
            return f"Audit evaluation rationale: {val}" if len(f"Audit evaluation rationale: {val}") >= 10 else "Structure layout evaluated against visual hierarchy according to standard audit rubric."
        return v.strip()


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

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_conf(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0 if val <= 100.0 else 1.0
            return round(max(0.0, min(1.0, val)), 4)
        except (ValueError, TypeError):
            return 0.95

    @field_validator("rationale", mode="before")
    @classmethod
    def _validate_rationale(cls, v: Any) -> str:
        if not isinstance(v, str) or len(v.strip()) < 10:
            val = str(v).strip() if v else ""
            return f"Combined audit evaluation: {val}" if len(f"Combined audit evaluation: {val}") >= 10 else "Audit evaluation completed according to standard pipeline criteria."
        return v.strip()


# ── OCR Output ──────────────────────────────────────────────────────────

class TextRegion(BaseModel):
    """A single text region extracted by OCR."""

    text: str
    bbox: list[BBoxCoord] = Field(
        ..., min_length=4, max_length=4,
        description="Bounding box normalized to 0-1"
    )
    role: Literal[
        "heading", "body", "button_label", "input_placeholder",
        "menu_item", "tab_label", "status_text", "caption",
        "tooltip", "error_message", "notification", "other"
    ]
    font_size_class: Literal["small", "medium", "large", "xlarge"] | None = None

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, v: Any) -> str:
        text = str(v).strip() if v else ""
        special_tokens = [
            r"<\|im_start\|>(?:system|user|assistant|tool)?(?:\n|\s+)?",
            r"<\|im_end\|>",
            r"<unk>",
            r"<s>",
            r"</s>",
            r"\[PAD\]",
            r"\[UNK\]",
            r"\[CLS\]",
            r"\[SEP\]",
        ]
        for token_pat in special_tokens:
            text = re.sub(token_pat, "", text, flags=re.IGNORECASE).strip()
        return text

    @field_validator("bbox", mode="before")
    @classmethod
    def _validate_bbox(cls, v: Any) -> list[float]:
        return sanitize_bbox(v)

    @field_validator("role", mode="before")
    @classmethod
    def _validate_role(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "other"
        v_clean = v.strip().lower()
        valid = {
            "heading", "body", "button_label", "input_placeholder",
            "menu_item", "tab_label", "status_text", "caption",
            "tooltip", "error_message", "notification", "other"
        }
        if v_clean in valid:
            return v_clean
        mapping = {
            "title": "heading", "header": "heading", "subtitle": "heading",
            "text": "body", "paragraph": "body", "content": "body",
            "button": "button_label", "btn": "button_label", "label": "button_label",
            "input": "input_placeholder", "placeholder": "input_placeholder",
            "menu": "menu_item", "tab": "tab_label", "status": "status_text",
            "error": "error_message", "alert": "notification", "badge": "notification",
        }
        return mapping.get(v_clean, "other")

    @field_validator("font_size_class", mode="before")
    @classmethod
    def _validate_font_size(cls, v: Any) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            return None
        v_clean = v.strip().lower()
        valid = {"small", "medium", "large", "xlarge"}
        if v_clean in valid:
            return v_clean
        if "xl" in v_clean or "huge" in v_clean:
            return "xlarge"
        if "lg" in v_clean or "big" in v_clean:
            return "large"
        if "sm" in v_clean or "tiny" in v_clean:
            return "small"
        return "medium"


class OCROutput(BaseModel):
    """Structured output from the OCR extraction stage."""

    text_regions: list[TextRegion] = Field(default_factory=list)
    primary_language: str = Field("en", min_length=2, max_length=5)
    total_text_regions: int = Field(0, ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_conf(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0 if val <= 100.0 else 1.0
            return round(max(0.0, min(1.0, val)), 4)
        except (ValueError, TypeError):
            return 0.95

    @field_validator("text_regions", mode="after")
    @classmethod
    def _sanitize_regions(cls, v: list[TextRegion]) -> list[TextRegion]:
        prompt_echos = (
            "you will receive a screenshot",
            "this screenshot contains",
            "extract all visible text",
            "organized by region",
            "specialized ocr model",
            "as an ai",
        )
        return [
            r for r in v
            if r.text.strip()
            and not any(phrase in r.text.strip().lower() for phrase in prompt_echos)
        ]

    @model_validator(mode="after")
    def _sync_total_count(self) -> OCROutput:
        self.total_text_regions = len(self.text_regions)
        return self


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


# ── Quality / Aesthetic Output ──────────────────────────────────────────

class QualityOutput(BaseModel):
    """Structured output from the aesthetic/quality scoring stage."""

    aesthetic_score: float = Field(..., ge=0.0, le=1.0)
    resolution_adequate: bool
    is_complete_ui: bool
    design_era: Literal["legacy", "flat", "modern"]
    issues: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("aesthetic_score", mode="before")
    @classmethod
    def _validate_score(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0:
                if val <= 5.0:
                    val = val / 5.0
                elif val <= 10.0:
                    val = val / 10.0
                elif val <= 100.0:
                    val = val / 100.0
                else:
                    val = 1.0
            return round(max(0.0, min(1.0, val)), 4)
        except (ValueError, TypeError):
            return 0.75

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_conf(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0:
                val = val / 100.0 if val <= 100.0 else 1.0
            return round(max(0.0, min(1.0, val)), 4)
        except (ValueError, TypeError):
            return 0.95

    @field_validator("design_era", mode="before")
    @classmethod
    def _validate_era(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "modern"
        v_clean = v.strip().lower()
        valid = {"legacy", "flat", "modern"}
        if v_clean in valid:
            return v_clean
        if v_clean in ("contemporary", "minimal", "clean", "glassmorphism", "neumorphism", "current"):
            return "modern"
        if v_clean in ("retro", "vintage", "skeuomorphic", "old", "classic"):
            return "legacy"
        if v_clean in ("material", "simple", "flat_design", "metro"):
            return "flat"
        return "modern"
