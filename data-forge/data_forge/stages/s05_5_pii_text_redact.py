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

from __future__ import annotations

import re
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

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
        # OCR enrichment doesn't advance status — records are still
        # "structured" when this runs, with ocr_output now populated.
        records = [r for r in records if r.status == "structured" and r.ocr_output]
        if not records:
            return result

        regex_patterns = stage_cfg.get("regex_patterns", {})
        compiled_patterns: dict[str, re.Pattern] = {}  # type: ignore[type-arg]
        for name, pattern in regex_patterns.items():
            try:
                compiled_patterns[name] = re.compile(pattern)
            except re.error as e:
                log.warning("invalid_regex", name=name, error=str(e))

        redaction_token = stage_cfg.get("redaction_token", "[REDACTED]")

        processed = 0
        redacted_count = 0

        for rec in records:
            try:
                img_path = config.data_root / (rec.scrubbed_image_path or "")
                if not img_path.exists():
                    # Not a hard failure for this stage — s08_encoding is
                    # the hard gate on scrubbed_image_path existing.
                    continue

                text_regions = (rec.ocr_output or {}).get("text_regions", [])
                if not text_regions:
                    continue

                new_detections: list[str] = []
                boxes_to_redact: list[list[float]] = []

                for region in text_regions:
                    text = region.get("text", "")
                    bbox = region.get("bbox")
                    for pii_name, pattern in compiled_patterns.items():
                        if pattern.search(text):
                            new_detections.append(f"{pii_name}_in_ocr_text")
                            if bbox and len(bbox) == 4:
                                boxes_to_redact.append(bbox)

                if not new_detections:
                    continue

                from PIL import Image, ImageDraw

                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                draw = ImageDraw.Draw(img)
                for bbox in boxes_to_redact:
                    x1, y1, x2, y2 = bbox
                    # bbox is normalized 0-1 (see structured_output.TextRegion)
                    px1, py1, px2, py2 = x1 * w, y1 * h, x2 * w, y2 * h
                    draw.rectangle([px1, py1, px2, py2], fill=(0, 0, 0))

                img.save(img_path, quality=95)

                merged_detections = list(rec.pii_detections or []) + new_detections
                manifest.update_record(
                    rec.id, "pii_text_redact",
                    pii_detections=merged_detections,
                    pii_scrubbed=True,
                )
                redacted_count += len(boxes_to_redact)
                processed += 1

            except Exception as e:
                log.error("pii_text_redact_failed", record_id=rec.id, error=str(e))

        result.records_processed = processed
        result.metadata = {"redaction_token": redaction_token, "regions_redacted": redacted_count}
        log.info("pii_text_redact_complete", processed=processed, regions_redacted=redacted_count)
        return result
