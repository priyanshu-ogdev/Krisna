"""Stage 5: Recaption + OCR — dense captioning via Tier-1, text extraction via OCR model."""

from __future__ import annotations

from typing import Any

from data_forge.config import PipelineConfig
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s05")


@register_stage("s05_recaption")
class RecaptionStage(Stage):
    name = "s05_recaption"
    requires = ["s04_safety"]

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status == "safety_classified" and r.safety_tier == "safe"]
        if not records or engine is None:
            return result

        tier1 = Tier1Engine(engine, config)
        processed = failed = 0

        for rec in records:
            img_path = config.data_root / (rec.scrubbed_image_path or rec.image_path or "")
            if not img_path.exists():
                manifest.update_record(rec.id, "recaption", new_status="excluded_failed",
                                       reason="Image missing", exclusion_reason="image_missing")
                failed += 1; continue

            caption_out = await tier1.generate_caption(img_path)
            if caption_out is None:
                manifest.update_record(rec.id, "recaption", new_status="excluded_failed",
                                       reason="Caption inference failed", exclusion_reason="inference_failed")
                failed += 1; continue

            manifest.update_record(rec.id, "recaption", new_status="recaptioned",
                                   caption=caption_out.caption,
                                   caption_output=caption_out.model_dump())
            processed += 1

        result.records_processed = processed
        result.records_failed = failed
        return result


@register_stage("s05_ocr_enrichment")
class OCREnrichmentStage(Stage):
    """Sub-stage: OCR text extraction (runs in a separate model swap phase)."""
    name = "s05_ocr_enrichment"
    requires = ["s05_recaption"]

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status == "recaptioned"]
        if not records or engine is None:
            return result

        from data_forge.inference.ocr import OCREngine
        ocr = OCREngine(engine, config)
        processed = 0

        for rec in records:
            img_path = config.data_root / (rec.scrubbed_image_path or rec.image_path or "")
            if not img_path.exists():
                continue

            ocr_out = await ocr.extract_text(img_path)
            if ocr_out:
                manifest.update_record(rec.id, "ocr_enrichment",
                                       ocr_output=ocr_out.model_dump())
                processed += 1

        result.records_processed = processed
        return result
