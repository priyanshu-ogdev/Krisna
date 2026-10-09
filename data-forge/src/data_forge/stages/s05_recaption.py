"""Stage 5: Recaption + OCR — dense captioning via Tier-1, text extraction via OCR model."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from PIL import UnidentifiedImageError

from data_forge.config import PipelineConfig
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.image_utils import load_image

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

        async def _process(rec):
            raw_path = rec.scrubbed_image_path or rec.image_path
            if not raw_path:
                return {"id": rec.id, "status": "excluded_failed", "reason": "No image path specified", "exclusion_reason": "image_missing"}
            p = Path(raw_path)
            img_path = p if p.is_absolute() else (config.data_root / p)
            if not img_path.is_file():
                return {"id": rec.id, "status": "excluded_failed", "reason": "Image missing", "exclusion_reason": "image_missing"}

            try:
                caption_out = await tier1.generate_caption(img_path, source_caption_hint=rec.source_caption)
            except Exception as e:
                return {"id": rec.id, "status": "excluded_failed", "reason": f"Caption inference exception: {e}", "exclusion_reason": "inference_failed"}
            if caption_out is None:
                return {"id": rec.id, "status": "excluded_failed", "reason": "Caption inference failed", "exclusion_reason": "inference_failed"}

            return {"id": rec.id, "status": "recaptioned", "caption": caption_out.caption, "out": caption_out.model_dump()}

        stage_cfg = config.get_stage("s05_recaption")
        batch_concurrency = stage_cfg.get("batch_size", 64)
        sem = asyncio.Semaphore(batch_concurrency)
        async def _bounded_process(rec):
            async with sem:
                return await _process(rec)

        tasks = [asyncio.create_task(_bounded_process(rec)) for rec in records]
        results = await asyncio.gather(*tasks)

        updates: list[dict[str, Any]] = []
        for res in results:
            status = res["status"]
            update_item: dict[str, Any] = {
                "id": res["id"],
                "new_status": status,
                "caption": res.get("caption"),
                "caption_output": res.get("out"),
            }
            if status == "recaptioned":
                processed += 1
            else:
                update_item["reason"] = res.get("reason")
                update_item["exclusion_reason"] = res.get("exclusion_reason")
                failed += 1
            updates.append(update_item)

        if updates:
            manifest.bulk_update_records(updates, stage="recaption")

        result.records_processed = processed
        result.records_failed = failed
        return result


# Re-export for backward compatibility (moved to dedicated s05_ocr_enrichment.py)
from data_forge.stages.s05_ocr_enrichment import OCREnrichmentStage  # noqa: E402

__all__ = ["RecaptionStage", "OCREnrichmentStage"]
