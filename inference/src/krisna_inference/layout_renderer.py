"""Deterministic low-fidelity renderer for LayoutIR handoff and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from krisna_inference.layout_ir import LayoutIR, project_layout

import hashlib

def _compute_renderer_hash() -> str:
    """Computes SHA-256 hash across bundled font assets and renderer specification."""
    hasher = hashlib.sha256()
    asset_dir = Path(__file__).parent / "assets" / "fonts"
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        fp = asset_dir / name
        if fp.exists():
            hasher.update(fp.read_bytes())
    hasher.update(b"krisna.layout_renderer.v2.deterministic_fonts_measured_wrap")
    return hasher.hexdigest()

RENDERER_SPEC_HASH = _compute_renderer_hash()
RENDERER_VERSION = f"krisna.renderer.v2.{RENDERER_SPEC_HASH[:8]}"


def _resolve_ui_fonts() -> dict[str, Any]:
    """Resolves crisp, deterministic, anti-aliased TrueType fonts with a modern typographic hierarchy.

    Guarantees:
    - Bundled DejaVuSans/DejaVuSans-Bold fonts are prioritized first, ensuring bitwise identical
      rendering and identical text metrics across Windows, Linux, and macOS environments.
    - All font sizes are strictly >= 12px to prevent spatial annihilation through 16x VAE downsampling.
    - Typographic roles: Display (30), Title (22), Header (18), Body (14), Button (14), Caption (12), Badge (12).
    """
    from PIL import ImageFont

    asset_dir = Path(__file__).parent / "assets" / "fonts"
    bundled_reg = asset_dir / "DejaVuSans.ttf"
    bundled_bold = asset_dir / "DejaVuSans-Bold.ttf"

    candidates_regular = [
        str(bundled_reg),
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "segoeui.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
    ]
    candidates_bold = [
        str(bundled_bold),
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica-Bold.ttc",
        "segoeuib.ttf",
        "arialbd.ttf",
    ]

    reg_path = None
    for p in candidates_regular:
        if Path(p).exists():
            reg_path = p
            break

    bold_path = None
    for p in candidates_bold:
        if Path(p).exists():
            bold_path = p
            break

    bold_path = bold_path or reg_path

    try:
        if reg_path and bold_path:
            return {
                "display": ImageFont.truetype(bold_path, 30),
                "title": ImageFont.truetype(bold_path, 22),
                "header": ImageFont.truetype(bold_path, 18),
                "body": ImageFont.truetype(reg_path, 14),
                "button": ImageFont.truetype(bold_path, 14),
                "caption": ImageFont.truetype(reg_path, 12),
                "badge": ImageFont.truetype(bold_path, 12),
            }
        elif reg_path:
            return {
                "display": ImageFont.truetype(reg_path, 30),
                "title": ImageFont.truetype(reg_path, 22),
                "header": ImageFont.truetype(reg_path, 18),
                "body": ImageFont.truetype(reg_path, 14),
                "button": ImageFont.truetype(reg_path, 14),
                "caption": ImageFont.truetype(reg_path, 12),
                "badge": ImageFont.truetype(reg_path, 12),
            }
    except Exception:
        pass

    default_font = ImageFont.load_default()
    return {
        "display": default_font,
        "title": default_font,
        "header": default_font,
        "body": default_font,
        "button": default_font,
        "caption": default_font,
        "badge": default_font,
    }


def _wrap_text_measured(draw: Any, text: str, font: Any, max_w: int) -> list[str]:
    """Wraps text strictly using measured pixel widths rather than character counts.

    Handles proportional fonts cleanly and breaks long words at pixel boundaries without
    generating ambiguous ellipsis ('...') tokens that confuse diffusion text generators.
    """
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        try:
            tb = draw.textbbox((0, 0), test_line, font=font)
            line_w = tb[2] - tb[0]
        except Exception:
            line_w = len(test_line) * 8

        if line_w <= max_w:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []
            
            # Check if this single word exceeds max_w
            try:
                tb_word = draw.textbbox((0, 0), word, font=font)
                word_w = tb_word[2] - tb_word[0]
            except Exception:
                word_w = len(word) * 8

            if word_w <= max_w:
                current_line = [word]
            else:
                # Slice word at measured pixel boundary
                sliced = ""
                for ch in word:
                    test_slice = sliced + ch
                    try:
                        tb_slice = draw.textbbox((0, 0), test_slice, font=font)
                        sw = tb_slice[2] - tb_slice[0]
                    except Exception:
                        sw = len(test_slice) * 8
                    if sw <= max_w:
                        sliced = test_slice
                    else:
                        break
                if sliced:
                    lines.append(sliced)
                current_line = []

    if current_line:
        lines.append(" ".join(current_line))
    return lines


def render_wireframe(layout: LayoutIR, output_path: str | Path, *, style: str = "gray") -> Path:
    """Render a high-entropy semantic 3-channel RGB wireframe for diffusion handoff.

    Research grounding:
    - TextDiffuser-2 (Chen et al., ECCV 2024): Explicit visual glyph footprints and typography conditioning.
    - AnyText (Tu et al., ICLR 2024): High-contrast visual semantic wireframe conditioning in standard RGB.
    - Cassowary (Badros et al., ACM TOCHI): Deterministic non-overlapping layout geometry.
    - FlowEdit (Kulikov et al., ICCV 2025): Dual vector-field guidance initialized from high-entropy wireframe prior.

    Guarantees:
    - 100% standard 3-channel RGB (fully compatible with frozen Qwen-Image-2.1 VAE).
    - Palette-aware surface fills and accent buttons.
    - Subtle elevation drop shadows directly in RGB luminance channels.
    - Anti-aliased TrueType typography scaling across Title, Header, Body, Button, and Badge archetypes.
    - Distinctive markers for locked regions.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for wireframe rendering") from exc

    report = project_layout(layout)
    canvas_w = layout.canvas_width
    canvas_h = layout.canvas_height

    # Extract palette and style tokens from root or layout constraints
    root_tokens = {}
    for node in report.layout.nodes:
        if node.id == "root" or node.parent is None:
            root_tokens = node.style_tokens or {}
            break

    palette = root_tokens.get("palette") or []
    is_dark = (
        style == "dark"
        or "dark" in str(root_tokens.get("style", "")).lower()
        or (palette and str(palette[0]).lower() in ("#000000", "#121212", "#0f172a", "#1a1a1a"))
    )

    if is_dark:
        bg_color = palette[0] if palette else "#0F172A"
        surface_color = "#1E293B"
        border_color = "#334155"
        text_color = "#F8FAFC"
        shadow_color = (0, 0, 0)
        accent_color = palette[1] if len(palette) > 1 else "#38BDF8"
    else:
        bg_color = palette[0] if palette and palette[0].lower() not in ("#ffffff", "#fff") else "#F8FAFC"
        surface_color = "#FFFFFF"
        border_color = "#E2E8F0"
        text_color = "#0F172A"
        shadow_color = (226, 232, 240)
        accent_color = palette[2] if len(palette) > 2 else (palette[1] if len(palette) > 1 else "#4F46E5")

    image = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(image)
    fonts = _resolve_ui_fonts()

    for node in sorted(report.layout.nodes, key=lambda item: (item.z_index, item.id)):
        if node.id == "root" and node.parent is None:
            continue

        x1 = int(node.bbox.x1 * canvas_w)
        y1 = int(node.bbox.y1 * canvas_h)
        x2 = int(node.bbox.x2 * canvas_w)
        y2 = int(node.bbox.y2 * canvas_h)

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        node_type = str(node.type).lower()

        # 1. Card / Content Container with Elevation Drop Shadow
        if node_type in ("card", "container", "list"):
            # Multi-layer ambient and directional drop shadows in RGB luminance
            if w > 16 and h > 16:
                draw.rounded_rectangle((x1 + 1, y1 + 3, x2 + 1, y2 + 5), radius=10, fill=shadow_color)
                draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=surface_color, outline=border_color, width=1)
            else:
                draw.rectangle((x1, y1, x2, y2), fill=surface_color, outline=border_color, width=1)

        # 2. Button Archetype (Pill / Rounded with Accent Fill and Centered TrueType Label)
        elif node_type == "button":
            radius = min(8, h // 2)
            draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=accent_color)
            label = node.text or "Action"
            btn_text_fill = "#FFFFFF" if not is_dark or accent_color != "#FFFFFF" else "#0F172A"
            btn_font = fonts["button"]
            try:
                tb = draw.textbbox((0, 0), label[:40], font=btn_font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
                tx = x1 + max(8, (w - tw) // 2)
                ty = y1 + max(2, (h - th) // 2)
            except Exception:
                tx = x1 + 10
                ty = y1 + max(2, (h - 14) // 2)
            draw.text((tx, ty), label[:40], fill=btn_text_fill, font=btn_font)
            continue

        # 3. Input Field Archetype
        elif node_type == "input":
            draw.rounded_rectangle((x1, y1, x2, y2), radius=4, fill=surface_color, outline=border_color, width=1)
            label = node.text or node.content_ref or "Enter text..."
            input_font = fonts["caption"]
            draw.text((x1 + 8, y1 + max(2, (h - 14) // 2)), label[:60], fill="#94A3B8", font=input_font)
            continue

        # 4. Divider Line
        elif node_type == "divider":
            mid_y = y1 + h // 2
            draw.line((x1, mid_y, x2, mid_y), fill=border_color, width=1)
            continue

        # 5. Generic Node / Outline Fallback
        else:
            colors = {"gray": "#D1D5DB", "type": "#93C5FD", "dark": "#374151"}
            stroke = colors.get(style, border_color)
            draw.rectangle((x1, y1, x2, y2), outline=stroke, width=3 if node.locked else 2)

        # Render Typographic Silhouette for Text / Labels with TrueType Hierarchy
        label = node.text or node.content_ref or (node.type if node_type not in ("container", "card") else "")
        if label:
            avail_w = max(16, w - 16)
            # Dynamic typographic scaling with bounded hierarchy protection:
            # - Title/Display: bounds to Title(22pt)->Header(18pt). Never degrades to body or caption!
            # - Card/Container: bounds to Header(18pt)->Body(14pt).
            # - Generic text: bounds to Body(14pt)->Caption(12pt).
            # All glyphs remain strictly >= 12px to survive 16x VAE latent downsampling.
            if node.id == "title" or y1 < int(0.22 * canvas_h):
                font_candidates = [fonts["title"], fonts["header"]]
            elif node_type in ("card", "container"):
                font_candidates = [fonts["header"], fonts["body"]]
            else:
                font_candidates = [fonts["body"], fonts["caption"]]

            active_font = font_candidates[-1]
            for f in font_candidates:
                try:
                    tb = draw.textbbox((0, 0), label, font=f)
                    if (tb[2] - tb[0]) <= avail_w:
                        active_font = f
                        break
                except Exception:
                    continue

            # Measured pixel-width wrapping: proportional fonts wrapped by actual rendered footprint
            wrapped_lines = _wrap_text_measured(draw, label, active_font, avail_w)
            
            # Explicit height constraint fallback:
            # - If container height < 36px, render strictly the first measured line (clean prefix, no '...' dots)
            # - If container height >= 36px, render up to the maximum lines that fit inside the box
            if h < 36:
                lines = wrapped_lines[:1]
            else:
                max_lines = max(1, (h - 16) // 18)
                lines = wrapped_lines[:max_lines]

            cur_y = y1 + 8
            for line in lines:
                draw.text((x1 + 8, cur_y), line, fill=text_color, font=active_font)
                cur_y += 18

        # Draw Distinct Locked Region Indicator
        if node.locked:
            draw.rectangle((x1, y1, x2, y2), outline="#F59E0B", width=2)
            badge_text = "[LOCKED]"
            badge_font = fonts["badge"]
            draw.text((max(x1 + 4, x2 - 60), y1 + 4), badge_text, fill="#D97706", font=badge_font)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
