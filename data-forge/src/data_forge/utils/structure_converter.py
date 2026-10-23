"""Structure-to-LayoutIR Converter and OCR Fusion Bridge.

Converts DataForge `StructureOutput` (recursive UI element tree) and optional
`OCROutput` (verbatim text regions) into a canonical `LayoutIR` suitable for
deterministic wireframe rendering.

Key Architectural Guarantees:
1. OCR Verbatim Fusion: Replaces generic VLM descriptive labels (e.g. "Submit button")
   with exact pixel-level OCR text (e.g. "Deploy Cluster") using spatial containment.
2. Depth-Aware Z-Indexing: Top-level containers render at z=0, nested cards at z=10,
   inner buttons/inputs at z=20, preventing surface fill occlusion.
3. Coordinate Rectification: Clamps boxes to [0.0, 1.0], swaps inverted bounds, and
   enforces minimum width/height >= 0.005.
4. Taxonomy Normalization: Maps fine-grained VLM types (e.g. "text_input", "tab_bar")
   into canonical wireframe archetypes ("input", "container", etc.).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Canonical Schema Definitions (Portable Fallback) ─────────────────────────
try:
    from krivex_inference.layout_ir import BBox, LayoutIR, LayoutNode, SceneArchetype
except ImportError:
    from enum import Enum

    class SceneArchetype(str, Enum):
        UI_LAYOUT = "ui_layout"
        GRAPHIC_POSTER = "graphic_poster"
        E_COMMERCE = "e_commerce"
        PORTRAIT_CHARACTER = "portrait_character"
        LANDSCAPE_SCENE = "landscape_scene"
        FREEFORM_ART = "freeform_art"

    class BBox(BaseModel):
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

    class LayoutIR(BaseModel):
        model_config = ConfigDict(extra="forbid")
        version: str = "krivex.layout.v1"
        scene_archetype: str = "ui_layout"
        canvas_width: int = Field(default=1024, ge=1, le=8192)
        canvas_height: int = Field(default=1024, ge=1, le=8192)
        nodes: list[LayoutNode] = Field(default_factory=list, max_length=64)


# ── Taxonomy Mapping ─────────────────────────────────────────────────────────

TAXONOMY_MAP: dict[str, str] = {
    # Functional Inputs
    "text_input": "input",
    "search_input": "input",
    "text_field": "input",
    "textarea": "input",
    "input": "input",
    "search": "input",
    "dropdown": "input",
    "select": "input",
    "slider": "input",
    # Buttons & Interactive Actions
    "button": "button",
    "btn": "button",
    "submit_button": "button",
    "action_button": "button",
    "toggle": "button",
    "checkbox": "button",
    "radio_button": "button",
    "switch": "button",
    "icon_button": "button",
    "fab": "button",
    # Promotional Badges & Tags
    "badge": "badge",
    "banner": "banner",
    "tag": "badge",
    "chip": "badge",
    "pill": "badge",
    "status": "badge",
    # Structural Layout Containers
    "navigation_bar": "container",
    "navbar": "container",
    "tab_bar": "container",
    "toolbar": "container",
    "status_bar": "container",
    "container": "container",
    "modal": "container",
    "dialog": "container",
    "drawer": "container",
    "sheet": "container",
    "sidebar": "container",
    "header": "container",
    "footer": "container",
    "region": "region",
    # Cards & Content Lists
    "card": "card",
    "list_item": "card",
    "table_row": "card",
    "tile": "card",
    "cell": "card",
    "list": "list",
    "table": "table",
    "divider": "divider",
    "separator": "divider",
    # Typography & Text Labels
    "text_label": "text",
    "text": "text",
    "label": "text",
    "heading": "headline",
    "headline": "headline",
    "title": "headline",
    "subhead": "subhead",
    "subtitle": "subhead",
    "body": "body_text",
    "body_text": "body_text",
    "paragraph": "body_text",
    "caption": "caption",
    "quote": "quote",
    # Media & Visual Assets
    "image": "image",
    "avatar": "product",
    "user_avatar": "product",
    "profile_pic": "product",
    "icon": "icon",
    "logo": "logo",
    "graphic": "shape",
    "shape": "shape",
    "illustration": "illustration",
    "photo": "image",
    "product": "product",
    "hero_image": "focal_subject",
}


def _clean_vlm_label(label: str | None, node_type: str) -> str | None:
    """Sanitize descriptive VLM labels when OCR verbatim text is unavailable."""
    if not label or not label.strip():
        return None
    text = label.strip()

    # Remove generic filler prompt echos
    patterns = [
        r"^(?:a\s+|an\s+|the\s+)?(?:blue|green|red|primary|secondary|action)?\s*button\s*(?:for\s+.*)?$",
        r"^(?:a\s+|an\s+|the\s+)?search\s*input\s*(?:field)?$",
        r"^(?:a\s+|an\s+|the\s+)?user\s*avatar$",
        r"^(?:a\s+|an\s+|the\s+)?(?:navigation|status|tab)\s*bar$",
    ]
    for pat in patterns:
        if re.match(pat, text, flags=re.IGNORECASE):
            # If label was purely descriptive e.g. "Submit button", extract root verb
            clean = re.sub(r"\s*(?:button|input|field|avatar|icon|bar)\b", "", text, flags=re.IGNORECASE).strip()
            return clean if clean else None

    # Truncate overly long descriptive phrases to avoid unreadable wireframe text
    max_len = 36 if node_type == "button" else (60 if node_type == "input" else 120)
    return text[:max_len].strip()


def _sanitize_bbox(raw_bbox: Any) -> tuple[float, float, float, float] | None:
    """Validate, sort, and clamp bounding box to [0.0, 1.0]."""
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        coords = [float(c) for c in raw_bbox]
    except (ValueError, TypeError):
        return None

    # Handle coordinates normalized to 0-1000 scale
    max_val = max(coords)
    if max_val > 1.0:
        divisor = 1000.0 if max_val <= 1000.0 else max_val
        coords = [c / divisor for c in coords]

    x1, y1, x2, y2 = coords
    # Swap inverted coordinates
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Discard zero-area degenerate points/lines before padding
    if abs(x2 - x1) <= 0.001 or abs(y2 - y1) <= 0.001:
        return None

    # Clamp to unit interval with strict positive span
    x1 = round(max(0.0, min(0.99, x1)), 4)
    y1 = round(max(0.0, min(0.99, y1)), 4)
    x2 = round(max(x1 + 0.005, min(1.0, x2)), 4)
    y2 = round(max(y1 + 0.005, min(1.0, y2)), 4)

    return (x1, y1, x2, y2)


def structure_to_layout_ir(
    structure: dict[str, Any] | Any,
    ocr_output: dict[str, Any] | Any | None = None,
    canvas_w: int = 1024,
    canvas_h: int = 1024,
    style: str = "dark",
) -> LayoutIR:
    """Convert DataForge StructureOutput and OCROutput into canonical LayoutIR."""
    struct_dict = structure.model_dump() if hasattr(structure, "model_dump") else (structure or {})
    ocr_dict = ocr_output.model_dump() if hasattr(ocr_output, "model_dump") else (ocr_output or {})

    # Extract raw elements tree
    elements = struct_dict.get("elements", []) if isinstance(struct_dict, dict) else []
    background_style = struct_dict.get("background_style", "solid_dark" if style == "dark" else "solid_light")

    # Step 1: Extract and validate OCR text regions
    ocr_regions: list[dict[str, Any]] = []
    for r in (ocr_dict.get("text_regions", []) if isinstance(ocr_dict, dict) else []):
        r_dict = r if isinstance(r, dict) else (r.model_dump() if hasattr(r, "model_dump") else {})
        t = str(r_dict.get("text", "")).strip()
        bb = _sanitize_bbox(r_dict.get("bbox"))
        if t and bb:
            ocr_regions.append({
                "text": t,
                "bbox": bb,
                "cx": (bb[0] + bb[2]) / 2.0,
                "cy": (bb[1] + bb[3]) / 2.0,
                "area": (bb[2] - bb[0]) * (bb[3] - bb[1]),
            })

    # Step 2: Recursively flatten structure tree with depth-aware z-indexing
    flat_nodes: list[dict[str, Any]] = []
    node_counter = 0

    def _traverse(raw_el: dict[str, Any], parent_id: str | None, depth: int):
        nonlocal node_counter
        if not isinstance(raw_el, dict):
            return

        raw_type = str(raw_el.get("type", "container")).strip().lower()
        norm_type = TAXONOMY_MAP.get(raw_type, "container")
        bb = _sanitize_bbox(raw_el.get("bbox"))
        if not bb:
            return

        node_id = f"node_{node_counter}"
        node_counter += 1

        children = raw_el.get("children", [])
        is_leaf = not bool(children)

        # Depth-aware z-index prevents parent container fills from occluding children
        z_index = depth * 10
        if norm_type in ("button", "badge"):
            z_index += 5  # elevate buttons above flat card backgrounds

        flat_nodes.append({
            "id": node_id,
            "type": norm_type,
            "raw_type": raw_type,
            "bbox": bb,
            "parent": parent_id,
            "z_index": z_index,
            "raw_label": raw_el.get("label"),
            "depth": depth,
            "is_leaf": is_leaf,
            "area": (bb[2] - bb[0]) * (bb[3] - bb[1]),
            "matched_texts": [],
        })

        for child in children:
            _traverse(child, node_id, depth + 1)

    for el in elements:
        _traverse(el, "root", depth=0)

    # Step 3: Verbatim OCR Spatial Fusion
    # Match each OCR text region to the smallest enclosing leaf element
    for region in ocr_regions:
        cx, cy = region["cx"], region["cy"]
        best_candidate: dict[str, Any] | None = None
        smallest_area = float("inf")

        for node in flat_nodes:
            # Containers do not receive verbatim text directly (their children do)
            if not node["is_leaf"] and node["type"] in ("container", "region", "modal"):
                continue

            x1, y1, x2, y2 = node["bbox"]
            # Check center point containment with subtle tolerance
            tol = 0.015
            if (x1 - tol) <= cx <= (x2 + tol) and (y1 - tol) <= cy <= (y2 + tol):
                # Prefer leaf or smaller component to avoid greedy card capture
                if node["area"] < smallest_area:
                    smallest_area = node["area"]
                    best_candidate = node

        if best_candidate is not None:
            best_candidate["matched_texts"].append((region["cy"], region["cx"], region["text"]))

    # Step 4: Construct LayoutNode objects
    layout_nodes: list[LayoutNode] = []

    # Add root background node with palette tokens
    root_style_tokens = {
        "style": style,
        "background_style": background_style,
        "palette": ["#0F172A", "#38BDF8", "#818CF8"] if style == "dark" else ["#F8FAFC", "#4F46E5", "#0F172A"],
    }
    layout_nodes.append(
        LayoutNode(
            id="root",
            type="container",
            bbox=BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
            parent=None,
            z_index=-1,
            style_tokens=root_style_tokens,
            text=None,
        )
    )

    for node in flat_nodes:
        x1, y1, x2, y2 = node["bbox"]

        # Final verbatim text selection
        if node["matched_texts"]:
            # Sort multiple matched lines in reading order (top-to-bottom, left-to-right)
            node["matched_texts"].sort(key=lambda item: (item[0], item[1]))
            final_text = " ".join(item[2] for item in node["matched_texts"]).strip()
        else:
            # Don't draw text on pure containers or dividers
            if node["type"] in ("container", "divider", "region"):
                final_text = None
            else:
                final_text = _clean_vlm_label(node["raw_label"], node["type"])

        # Truncate to Pydantic limit
        if final_text and len(final_text) > 4000:
            final_text = final_text[:4000]

        layout_nodes.append(
            LayoutNode(
                id=node["id"],
                type=node["type"],
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                parent=node["parent"],
                z_index=node["z_index"],
                text=final_text if final_text else None,
                style_tokens={"depth": node["depth"]},
            )
        )

    return LayoutIR(
        version="krivex.layout.v1",
        scene_archetype="ui_layout",
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        nodes=layout_nodes[:64],
    )
