"""Stage 5.5: PII Text Redaction — regex-based text-PII detection + visual redaction.

This is the "(Critical)" text-PII half of the original v13 "PII Scrub" design
that s03_5_pii_scrub.py's docstring promised but never actually performed:
that stage ran right after Stage 3 (Quality), before OCR extraction had ever
happened, so `rec.ocr_output` was always empty and the regex email/phone/SSN/
credit-card detection was dead code that never fired.

OCR text (`rec.ocr_output`) only becomes available after Stage 5's OCR
enrichment sub-stage (s05_ocr_enrichment) runs. This stage runs immediately
after that, so it actually has data to act on.

Unlike the old dead branch — which only *logged* a PII text match into
manifest metadata without touching the image — this stage draws an opaque
redaction box over the matched text region on the scrubbed image itself, so
"scrub" actually means the pixels are gone, not just that a detection was
recorded.
"""

import concurrent.futures
import math
import multiprocessing
import os
from pathlib import Path
import re
from typing import Any, ClassVar

from PIL import Image, ImageDraw

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.path_safety import resolve_data_path

log = get_logger("stages.s05_5")


@register_stage("s05_5_pii_text_redact")
class PIITextRedactStage(Stage):
    name = "s05_5_pii_text_redact"
    requires: ClassVar[tuple[str, ...]] = ("s05_ocr_enrichment",)

    async def run(
        self, manifest: Manifest, config: PipelineConfig,
        record_ids: list[str], engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s05_5_pii_text_redact")

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status in ("structured", "recaptioned")]
        if not records:
            return result

        regex_patterns = stage_cfg.get("regex_patterns", {})
        compiled_patterns: dict[str, re.Pattern] = {}  # type: ignore[type-arg]
        for name, pattern in regex_patterns.items():
            try:
                compiled_patterns[name] = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid configured PII pattern {name!r}: {e}") from e
        if not compiled_patterns:
            raise ValueError("Text-PII redaction requires at least one configured detection pattern")

        redaction_token = stage_cfg.get("redaction_token", "[REDACTED]")

        processed = failed = 0
        redacted_count = 0

        max_workers = min(32, max(4, multiprocessing.cpu_count()))

        def _sanitize(value):
            if isinstance(value, str):
                for pattern in compiled_patterns.values():
                    value = pattern.sub(redaction_token, value)
                return value
            if isinstance(value, list):
                return [_sanitize(item) for item in value]
            if isinstance(value, dict):
                return {key: _sanitize(item) for key, item in value.items()}
            return value

        def _contains_pii(value) -> bool:
            if isinstance(value, str):
                return any(pattern.search(value) for pattern in compiled_patterns.values())
            if isinstance(value, list):
                return any(_contains_pii(item) for item in value)
            if isinstance(value, dict):
                return any(_contains_pii(item) for item in value.values())
            return False
        
        def _process_redaction(rec):
            try:
                if not rec.ocr_output:
                    return {
                        "id": rec.id,
                        "error": "OCR output missing; text-PII scan could not run",
                    }
                if not rec.scrubbed_image_path:
                    return {"id": rec.id, "error": "Scrubbed image missing"}
                img_path = resolve_data_path(config.data_root, rec.scrubbed_image_path)
                if not img_path.is_file():
                    return {"id": rec.id, "error": "Scrubbed image file missing"}
                
                text_regions = (rec.ocr_output or {}).get("text_regions", [])
                new_detections: list[str] = []
                boxes_to_redact: list[list[float]] = []
                unredactable = False

                for region in text_regions:
                    text = region.get("text", "") if isinstance(region, dict) else ""
                    bbox = region.get("bbox") if isinstance(region, dict) else None
                    for pii_name, pattern in compiled_patterns.items():
                        if isinstance(text, str) and pattern.search(text):
                            new_detections.append(f"{pii_name}_in_ocr_text")
                            if bbox and len(bbox) == 4:
                                try:
                                    coords = [float(v) for v in bbox]
                                except (TypeError, ValueError):
                                    unredactable = True
                                    continue
                                if all(math.isfinite(v) for v in coords):
                                    boxes_to_redact.append(coords)
                                else:
                                    unredactable = True
                            else:
                                unredactable = True

                metadata_has_pii = any(_contains_pii(value) for value in (
                    rec.caption,
                    rec.source_caption,
                    rec.caption_output,
                    rec.structure_output,
                    rec.ocr_output,
                    rec.critique_output,
                ))
                if metadata_has_pii and (not boxes_to_redact or unredactable):
                    return {
                        "id": rec.id,
                        "error": "PII detected in text metadata without a usable image bounding box",
                    }
                with Image.open(img_path) as source_image:
                    image_format = source_image.format
                    img = source_image.convert("RGB")
                w, h = img.size
                draw = ImageDraw.Draw(img)
                for bbox in boxes_to_redact:
                    x1, y1, x2, y2 = bbox
                    if all(0.0 <= coordinate <= 1.0 for coordinate in bbox):
                        px1, py1, px2, py2 = x1 * w, y1 * h, x2 * w, y2 * h
                    else:
                        px1, py1, px2, py2 = x1, y1, x2, y2
                    px1 = max(0, min(w, px1))
                    py1 = max(0, min(h, py1))
                    px2 = max(0, min(w, px2))
                    py2 = max(0, min(h, py2))
                    if px2 <= px1 or py2 <= py1:
                        return {
                            "id": rec.id,
                            "error": "PII bounding box is empty or outside the image",
                        }
                    draw.rectangle([px1, py1, px2, py2], fill=(0, 0, 0))

                if boxes_to_redact:
                    if image_format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                        return {"id": rec.id, "error": f"Unsupported image format for redaction: {image_format}"}
                    temp_path = img_path.with_name(f".{img_path.name}.redacting")
                    save_options = {"quality": 95} if image_format == "JPEG" else {}
                    img.save(temp_path, format=image_format, **save_options)
                    os.replace(temp_path, img_path)
                
                return {
                    "id": rec.id,
                    "new_detections": new_detections,
                    "old_detections": list(rec.pii_detections or []),
                    "redacted_count": len(boxes_to_redact),
                    "ocr_output": _sanitize(rec.ocr_output),
                    "caption": _sanitize(rec.caption),
                    "source_caption": _sanitize(rec.source_caption),
                    "caption_output": _sanitize(rec.caption_output),
                    "structure_output": _sanitize(rec.structure_output),
                    "critique_output": _sanitize(rec.critique_output),
                }

            except Exception as e:
                log.error("pii_text_redact_failed", record_id=rec.id, error=str(e))
                return {"id": rec.id, "error": f"Text-PII redaction failed: {e}"}

        updates: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(_process_redaction, records):
                if res.get("error"):
                    updates.append({
                        "id": res["id"],
                        "new_status": "excluded_failed",
                        "reason": res["error"],
                        "exclusion_reason": "pii_text_redact_failed",
                        "pii_scrubbed": False,
                    })
                    failed += 1
                else:
                    merged_detections = list(dict.fromkeys(res["old_detections"] + res["new_detections"]))
                    updates.append({
                        "id": res["id"],
                        "ocr_output": res["ocr_output"],
                        "caption": res["caption"],
                        "source_caption": res["source_caption"],
                        "caption_output": res["caption_output"],
                        "structure_output": res["structure_output"],
                        "critique_output": res["critique_output"],
                        "pii_detections": merged_detections,
                        "pii_scrubbed": True,
                    })
                    redacted_count += res["redacted_count"]
                    processed += 1

                if len(updates) >= 500:
                    manifest.bulk_update_records(updates, stage="pii_text_redact")
                    updates.clear()

        if updates:
            manifest.bulk_update_records(updates, stage="pii_text_redact")

        result.records_processed = processed
        result.records_failed = failed
        result.metadata = {"redaction_token": redaction_token, "regions_redacted": redacted_count}
        log.info("pii_text_redact_complete", processed=processed, regions_redacted=redacted_count)
        return result
