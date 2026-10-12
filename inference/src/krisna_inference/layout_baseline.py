"""Planner-to-layout baseline.

This is intentionally not a learned model. It converts a validated Planner
delta into a deterministic LayoutIR, providing the required baseline before a
LayoutFlow-style model is trained.
"""

from __future__ import annotations

from krisna_inference.layout_ir import BBox, LayoutIR, LayoutNode, project_layout


def layout_from_constraints(constraints: dict, *, text: str = "") -> LayoutIR:
    """Build a conservative starter layout from Planner constraints."""
    style = constraints.get("style")
    palette = constraints.get("palette")
    nodes = [
        LayoutNode(id="root", type="container", bbox=BBox(x1=0.04, y1=0.04, x2=0.96, y2=0.96), style_tokens={"style": style, "palette": palette}),
        LayoutNode(id="title", type="text", bbox=BBox(x1=0.10, y1=0.10, x2=0.90, y2=0.20), parent="root", text=text[:240] or "Title", locked=False),
        LayoutNode(id="content", type="container", bbox=BBox(x1=0.10, y1=0.26, x2=0.90, y2=0.82), parent="root"),
        LayoutNode(id="action", type="button", bbox=BBox(x1=0.10, y1=0.86, x2=0.34, y2=0.94), parent="root", text="Continue"),
    ]
    return project_layout(LayoutIR(nodes=nodes)).layout
