"""Canonical Krivex layout intermediate representation.

This is deliberately independent of any generative model.  LayoutFlow,
Planner JSON, or a future model must all pass through this representation
before rendering or state handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from enum import Enum


class SceneArchetype(str, Enum):
    UI_LAYOUT = "ui_layout"
    GRAPHIC_POSTER = "graphic_poster"
    E_COMMERCE = "e_commerce"
    PORTRAIT_CHARACTER = "portrait_character"
    LANDSCAPE_SCENE = "landscape_scene"
    FREEFORM_ART = "freeform_art"


LAYOUT_SCHEMA_VERSION = "krivex.layout.v1"
NODE_TYPES = {
    # Structural & Layout Containers
    "container", "card", "region", "canvas_zone", "modal", "background",
    # Typography & Textual Spans
    "text", "headline", "subhead", "body_text", "caption", "quote", "label",
    # Visual Media & Subjects
    "image", "focal_subject", "illustration", "character", "product", "photo_frame",
    # Functional & Interactive Elements
    "button", "input", "list", "table", "navbar", "footer", "sidebar",
    # Graphic Design, Accents & Brand Assets
    "badge", "icon", "logo", "divider", "shape", "banner", "decorative",
    # Freeform & Custom
    "sketch", "contour", "overlay", "other",
}


class BBox(BaseModel):
    """Normalized x1,y1,x2,y2 coordinates."""

    model_config = ConfigDict(extra="forbid")

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @field_validator("x2")
    @classmethod
    def x_order(cls, value: float, info):
        x1 = info.data.get("x1")
        if x1 is not None and value <= x1:
            raise ValueError("x2 must be greater than x1")
        return value

    @field_validator("y2")
    @classmethod
    def y_order(cls, value: float, info):
        y1 = info.data.get("y1")
        if y1 is not None and value <= y1:
            raise ValueError("y2 must be greater than y1")
        return value

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> "BBox":
        return cls(x1=x, y1=y, x2=x + width, y2=y + height)

    def to_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (self.x2 - self.x1) * (self.y2 - self.y1) + (other.x2 - other.x1) * (other.y2 - other.y1) - inter
        return inter / union if union else 0.0


class LayoutNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    type: str
    bbox: BBox
    parent: str | None = None
    z_index: int = 0
    style_tokens: dict[str, Any] = Field(default_factory=dict)
    text: str | None = Field(default=None, max_length=4096)
    content_ref: str | None = None
    locked: bool = False
    is_focal_subject: bool = False
    facial_landmarks: dict[str, Any] | None = None
    identity_lock: bool = False
    source_asset_id: str | None = None
    source_crop_bbox: tuple[float, float, float, float] | None = None
    props: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def known_type(cls, value: str) -> str:
        return value if value in NODE_TYPES else "other"


class LayoutRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    relation: str
    target: str


class LayoutIR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = LAYOUT_SCHEMA_VERSION
    scene_archetype: str = "ui_layout"
    canvas_width: int = Field(default=1024, ge=1, le=8192)
    canvas_height: int = Field(default=1024, ge=1, le=8192)
    nodes: list[LayoutNode] = Field(default_factory=list, max_length=64)
    relations: list[LayoutRelation] = Field(default_factory=list, max_length=256)
    diagnostics: list[str] = Field(default_factory=list, max_length=64)

    def node_map(self) -> dict[str, LayoutNode]:
        return {node.id: node for node in self.nodes}


@dataclass(frozen=True)
class LayoutViolation:
    code: str
    node_id: str | None
    detail: str


@dataclass(frozen=True)
class ProjectionReport:
    layout: LayoutIR
    violations_before: tuple[LayoutViolation, ...]
    displacement_mean: float
    displacement_max: float
    changed_nodes: tuple[str, ...]


def validate_layout(layout: LayoutIR) -> list[LayoutViolation]:
    """Return violations without mutating the candidate."""
    violations: list[LayoutViolation] = []
    nodes = layout.node_map()
    if len(nodes) != len(layout.nodes):
        violations.append(LayoutViolation("duplicate_id", None, "node IDs must be unique"))
    for node in layout.nodes:
        if node.parent == node.id:
            violations.append(LayoutViolation("self_parent", node.id, "node cannot parent itself"))
        if node.parent and node.parent not in nodes:
            violations.append(LayoutViolation("missing_parent", node.id, node.parent))
    for relation in layout.relations:
        if relation.source not in nodes or relation.target not in nodes:
            violations.append(LayoutViolation("missing_relation_node", None, f"{relation.source}->{relation.target}"))
    # Detect parent cycles by walking each chain.
    for node in layout.nodes:
        seen: set[str] = set()
        current = node
        while current.parent:
            if current.parent in seen:
                violations.append(LayoutViolation("parent_cycle", node.id, node.parent or ""))
                break
            seen.add(current.parent)
            current = nodes.get(current.parent, current)
            if current.id == node.id:
                break
    return violations


def project_layout(layout: LayoutIR, *, max_displacement: float = 0.08) -> ProjectionReport:
    """Clamp boxes and preserve locked nodes, reporting geometric movement.

    Projection is intentionally conservative. It does not reorder nodes or
    invent missing parents; callers can resample when displacement exceeds the
    configured threshold.
    """
    before = validate_layout(layout)
    changed: list[str] = []
    distances: list[float] = []
    projected: list[LayoutNode] = []
    for node in layout.nodes:
        b = node.bbox
        values = (max(0.0, min(1.0, b.x1)), max(0.0, min(1.0, b.y1)), max(0.0, min(1.0, b.x2)), max(0.0, min(1.0, b.y2)))
        x1, y1, x2, y2 = values
        if x2 <= x1:
            x2 = min(1.0, x1 + 1e-3)
        if y2 <= y1:
            y2 = min(1.0, y1 + 1e-3)
        projected_box = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
        distance = sum(abs(a - b) for a, b in zip(values, (b.x1, b.y1, b.x2, b.y2))) / 4.0
        distances.append(distance)
        if distance > 0:
            changed.append(node.id)
        projected.append(node.model_copy(update={"bbox": projected_box}))
    result = layout.model_copy(update={"nodes": projected})
    mean = sum(distances) / len(distances) if distances else 0.0
    return ProjectionReport(result, tuple(before), mean, max(distances, default=0.0), tuple(changed))
