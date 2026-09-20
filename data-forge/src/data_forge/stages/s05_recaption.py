"""Stage 5: Recaption + OCR — dense captioning via Tier-1, text extraction via OCR model."""

from __future__ import annotations

from typing import Any, ClassVar

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
    requires: ClassVar[tuple[str, ...]] = ("s04_safety",)

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
                failed += 1
                continue

            try:
                from PIL import UnidentifiedImageError
                from data_forge.utils.image_utils import load_image
                _ = load_image(img_path)
            except UnidentifiedImageError:
                manifest.update_record(rec.id, "recaption", new_status="excluded_failed",
                                       reason="Corrupt image", exclusion_reason="image_corrupt")
                failed += 1
                continue
            except Exception as e:
                manifest.update_record(rec.id, "recaption", new_status="excluded_failed",
                                       reason=f"Image load error: {e}", exclusion_reason="image_error")
                failed += 1
                continue

            caption_out = await tier1.generate_caption(img_path, source_caption_hint=rec.source_caption)
            if caption_out is None:
                manifest.update_record(rec.id, "recaption", new_status="excluded_failed",
                                       reason="Caption inference failed", exclusion_reason="inference_failed")
                failed += 1
                continue

            manifest.update_record(rec.id, "recaption", new_status="recaptioned",
                                   caption=caption_out.caption,
                                   caption_output=caption_out.model_dump())
            processed += 1

        result.records_processed = processed
        result.records_failed = failed
        return result


# Re-export for backward compatibility (moved to dedicated s05_ocr_enrichment.py)
from data_forge.stages.s05_ocr_enrichment import OCREnrichmentStage  # noqa: E402

__all__ = ["RecaptionStage", "OCREnrichmentStage"]
