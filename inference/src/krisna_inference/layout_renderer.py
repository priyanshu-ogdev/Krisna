"""Deterministic low-fidelity renderer for LayoutIR handoff and evaluation."""

from __future__ import annotations

from pathlib import Path

from krisna_inference.layout_ir import LayoutIR, project_layout


def render_wireframe(layout: LayoutIR, output_path: str | Path, *, style: str = "gray") -> Path:
    """Render a stable wireframe; no generative model or network is involved."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required for wireframe rendering") from exc

    report = project_layout(layout)
    image = Image.new("RGB", (layout.canvas_width, layout.canvas_height), "white")
    draw = ImageDraw.Draw(image)
    colors = {"gray": "#D1D5DB", "type": "#93C5FD", "dark": "#374151"}
    stroke = colors.get(style, colors["gray"])
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for node in sorted(report.layout.nodes, key=lambda item: (item.z_index, item.id)):
        x1 = int(node.bbox.x1 * layout.canvas_width)
        y1 = int(node.bbox.y1 * layout.canvas_height)
        x2 = int(node.bbox.x2 * layout.canvas_width)
        y2 = int(node.bbox.y2 * layout.canvas_height)
        draw.rectangle((x1, y1, x2, y2), outline=stroke, width=3 if node.locked else 2)
        label = node.text or node.content_ref or node.type
        draw.text((x1 + 4, y1 + 4), label[:80], fill="#111827", font=font)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
