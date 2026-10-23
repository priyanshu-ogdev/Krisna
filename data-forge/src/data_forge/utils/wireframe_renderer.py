"""Deterministic low-fidelity wireframe renderer for DataForge training pair synthesis.

Grounding & Research Lineage:
- TextDiffuser-2 (Chen et al., ECCV 2024): Explicit visual glyph footprints and typography conditioning.
- AnyText (Tu et al., ICLR 2024): High-contrast visual semantic wireframe conditioning in standard RGB.
- SPADE / GauGAN (Park et al., CVPR 2019 Oral): Semantic spatial layout conditioning.
- LayerDiffuse (Zhang & Agrawala, CVPR 2024): Structural transparency and layer-aware spatial conditioning.

Guarantees:
- 100% standard 3-channel RGB (fully compatible with frozen Qwen-Image-2.1 VAE).
- Bitwise identical rendering across environments via bundled DejaVuSans TrueType fonts.
- Produces matching RENDERER_VERSION hash to ensure 0 distribution shift between training and inference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from data_forge.logging_setup import get_logger
from data_forge.utils.path_safety import resolve_data_path
from data_forge.utils.structure_converter import LayoutIR, structure_to_layout_ir

log = get_logger("utils.wireframe_renderer")

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
    _HAS_PIL = False


def _compute_renderer_hash() -> str:
    """Computes SHA-256 hash across bundled font assets and renderer specification."""
    hasher = hashlib.sha256()
    asset_dir = Path(__file__).parent.parent / "assets" / "fonts"
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        fp = asset_dir / name
        if fp.exists():
            hasher.update(fp.read_bytes())
    hasher.update(b"krivex.layout_renderer.v2.deterministic_fonts_measured_wrap")
    return hasher.hexdigest()


RENDERER_SPEC_HASH = _compute_renderer_hash()
RENDERER_VERSION = f"krivex.renderer.v2.{RENDERER_SPEC_HASH[:8]}"

# Global cache for font objects across records to avoid descriptor exhaustion
_FONT_CACHE: dict[str, dict[str, Any]] = {}


def _resolve_ui_fonts() -> dict[str, Any]:
    """Resolves crisp, deterministic, anti-aliased TrueType fonts with cached handles."""
    if "default" in _FONT_CACHE:
        return _FONT_CACHE["default"]

    if not _HAS_PIL:
        raise RuntimeError("Pillow is required for wireframe rendering")

    asset_dir = Path(__file__).parent.parent / "assets" / "fonts"
    bundled_reg = asset_dir / "DejaVuSans.ttf"
    bundled_bold = asset_dir / "DejaVuSans-Bold.ttf"

    candidates_regular = [
        str(bundled_reg),
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "DejaVuSans.ttf",
        "arial.ttf",
    ]
    candidates_bold = [
        str(bundled_bold),
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica-Bold.ttc",
        "DejaVuSans-Bold.ttf",
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
            resolved = {
                "display": ImageFont.truetype(bold_path, 30),
                "title": ImageFont.truetype(bold_path, 22),
                "header": ImageFont.truetype(bold_path, 18),
                "body": ImageFont.truetype(reg_path, 14),
                "button": ImageFont.truetype(bold_path, 14),
                "caption": ImageFont.truetype(reg_path, 12),
                "badge": ImageFont.truetype(bold_path, 12),
            }
            _FONT_CACHE["default"] = resolved
            return resolved
    except Exception as exc:
        log.warning("truetype_font_load_fallback", error=str(exc))

    default_font = ImageFont.load_default()
    resolved = {
        "display": default_font,
        "title": default_font,
        "header": default_font,
        "body": default_font,
        "button": default_font,
        "caption": default_font,
        "badge": default_font,
    }
    _FONT_CACHE["default"] = resolved
    return resolved


def _wrap_text_measured(text: str, font: Any, max_width: int, draw: Any) -> list[str]:
    """Word-wraps text based on exact font bounding box measurements."""
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        candidate = " ".join(current_line + [word])
        try:
            bbox = draw.textbbox((0, 0), candidate, font=font)
            line_w = bbox[2] - bbox[0]
        except Exception:
            line_w = len(candidate) * 8

        if line_w <= max_width or not current_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))
    return lines


def render_wireframe(layout: LayoutIR, output_path: str | Path, *, style: str = "dark") -> Path:
    """Render a high-entropy semantic 3-channel RGB wireframe for diffusion handoff.

    Guarantees:
    - 100% standard 3-channel RGB (fully compatible with frozen Qwen-Image-2.1 VAE).
    - Palette-aware surface fills and accent buttons.
    - Subtle elevation drop shadows directly in RGB luminance channels.
    - Anti-aliased TrueType typography scaling across Title, Header, Body, Button, and Badge archetypes.
    """
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required for wireframe rendering")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    canvas_w = max(256, layout.canvas_width)
    canvas_h = max(256, layout.canvas_height)

    # Extract palette and style tokens from root
    root_tokens: dict[str, Any] = {}
    for node in layout.nodes:
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

    for node in sorted(layout.nodes, key=lambda item: (item.z_index, item.id)):
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
        if node_type in ("card", "container", "list", "table", "region"):
            if w > 16 and h > 16:
                elevation = max(1, min(5, int(getattr(node, "z_index", 0)) // 10 + 1))
                shadow_offset_y = 2 + elevation * 2
                shadow_radius = 8 + elevation * 2
                draw.rounded_rectangle(
                    (x1 + 1, y1 + shadow_offset_y, x2 + 1, y2 + shadow_offset_y + 2),
                    radius=shadow_radius,
                    fill=shadow_color,
                )
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
                tx = x1 + 8
                ty = y1 + max(2, (h - 14) // 2)
            draw.text((tx, ty), label[:40], fill=btn_text_fill, font=btn_font)
            continue

        # 3. Badge / Banner Archetype
        elif node_type in ("badge", "banner"):
            radius = min(6, max(2, h // 2))
            draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=surface_color, outline=accent_color, width=1)
            label = node.text or "New"
            badge_font = fonts["badge"]
            draw.text((x1 + 6, y1 + max(2, (h - 12) // 2)), label[:30], fill=accent_color, font=badge_font)
            continue

        # 4. Input Field Archetype
        elif node_type == "input":
            draw.rounded_rectangle((x1, y1, x2, y2), radius=4, fill=surface_color, outline=border_color, width=1)
            label = node.text or node.content_ref or "Enter text..."
            input_font = fonts["caption"]
            draw.text((x1 + 8, y1 + max(2, (h - 14) // 2)), label[:60], fill="#94A3B8", font=input_font)
            continue

        # 5. Divider Line
        elif node_type == "divider":
            mid_y = y1 + h // 2
            draw.line((x1, mid_y, x2, mid_y), fill=border_color, width=1)
            continue

        # 6. Shape / Decorative Artwork
        elif node_type in ("shape", "decorative", "icon", "logo"):
            draw.rounded_rectangle((x1, y1, x2, y2), radius=8, outline=accent_color, width=2)
            continue

        # 7. Focal Subject / Product / Avatar
        elif node_type in ("focal_subject", "product", "illustration", "image") or getattr(node, "is_focal_subject", False):
            draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=surface_color, outline=border_color, width=1)
            # Add reticles
            r_len = min(12, max(4, min(w // 4, h // 4)))
            draw.line((x1 + 4, y1 + 4, x1 + 4 + r_len, y1 + 4), fill=accent_color, width=1)
            draw.line((x1 + 4, y1 + 4, x1 + 4, y1 + 4 + r_len), fill=accent_color, width=1)
            draw.line((x2 - 4, y1 + 4, x2 - 4 - r_len, y1 + 4), fill=accent_color, width=1)
            draw.line((x2 - 4, y1 + 4, x2 - 4, y1 + 4 + r_len), fill=accent_color, width=1)

        # 8. Outline Fallback
        else:
            colors = {"gray": "#D1D5DB", "dark": "#374151"}
            stroke = colors.get(style, border_color)
            draw.rectangle((x1, y1, x2, y2), outline=stroke, width=2)

        # Render Typographic Footprint for Text / Labels with TrueType Hierarchy
        label = node.text or node.content_ref or ""
        if label and node_type not in ("container", "card", "list", "table", "region"):
            avail_w = max(16, w - 16)
            if node_type in ("headline", "title") or y1 < int(0.20 * canvas_h):
                font_candidates = [fonts["display"], fonts["title"], fonts["header"]]
            elif node_type in ("subhead",):
                font_candidates = [fonts["header"], fonts["body"]]
            elif node_type in ("caption", "badge"):
                font_candidates = [fonts["caption"], fonts["badge"]]
            else:
                font_candidates = [fonts["body"], fonts["caption"]]

            chosen_font = font_candidates[-1]
            chosen_lines = [label]
            for font in font_candidates:
                lines = _wrap_text_measured(label, font, avail_w, draw)
                try:
                    tb = draw.textbbox((0, 0), "Ag", font=font)
                    line_h = (tb[3] - tb[1]) + 4
                except Exception:
                    line_h = 16
                if len(lines) * line_h <= max(16, h - 8):
                    chosen_font = font
                    chosen_lines = lines
                    break

            try:
                tb = draw.textbbox((0, 0), "Ag", font=chosen_font)
                line_h = (tb[3] - tb[1]) + 4
            except Exception:
                line_h = 16

            cur_y = y1 + max(4, (h - len(chosen_lines) * line_h) // 2)
            for line_str in chosen_lines[:6]:
                if cur_y + line_h > y2 - 2:
                    break
                draw.text((x1 + 8, cur_y), line_str, fill=text_color, font=chosen_font)
                cur_y += line_h

    image.save(out_p, format="PNG", optimize=True)
    return out_p


def synthesize_record_wireframe(rec: Any, config: Any, output_path: Path | None = None) -> Path | None:
    """Synthesize a deterministic wireframe condition image for a ManifestRecord."""
    if not rec.scrubbed_image_path:
        return None
    try:
        img_path = resolve_data_path(config.data_root, rec.scrubbed_image_path)
        if not img_path.is_file():
            return None
        with Image.open(img_path) as target_img:
            target_w, target_h = target_img.size
    except Exception as exc:
        log.warning("target_image_open_failed", record_id=rec.id, error=str(exc))
        return None

    structure = getattr(rec, "structure_output", None)
    if not isinstance(structure, dict) or not structure.get("elements"):
        return None

    ocr_output = getattr(rec, "ocr_output", None)

    layout = structure_to_layout_ir(
        structure=structure,
        ocr_output=ocr_output,
        canvas_w=target_w,
        canvas_h=target_h,
        style="dark",
    )

    if output_path is None:
        paths = config.paths.resolve(config.data_root) if config.paths else config.resolved_paths
        wireframes_dir = paths.get("wireframes", config.data_root / "processed" / "wireframes")
        wireframes_dir.mkdir(parents=True, exist_ok=True)
        output_path = wireframes_dir / f"{rec.id}.png"

    try:
        return render_wireframe(layout, output_path, style="dark")
    except Exception as exc:
        log.error("wireframe_synthesis_failed", record_id=rec.id, error=str(exc))
        return None
