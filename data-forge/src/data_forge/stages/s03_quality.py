"""Stage 3: Quality — aesthetic scoring and resolution filtering via Tier-1."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image

from data_forge.config import PipelineConfig
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s03")


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


@register_stage("s03_quality")
class QualityStage(Stage):
    name = "s03_quality"
    requires: ClassVar[tuple[str, ...]] = ("s02_dedup",)

    async def run(
        self, manifest: Manifest, config: PipelineConfig,
        record_ids: list[str], engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s03_quality")
        threshold = stage_cfg.get("aesthetic_score_threshold", 0.4)
        min_res = stage_cfg.get("min_resolution", [256, 256])
        max_res = stage_cfg.get("max_resolution", [8192, 8192])

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status == "deduped"]
        if not records or engine is None:
            return result

        tier1 = Tier1Engine(engine, config)
        processed = 0
        excluded = 0
        failed = 0

        async def _process(rec):
            if not rec.image_path:
                return {"id": rec.id, "status": "excluded_failed", "reason": "No image path specified", "exclusion_reason": "image_missing"}
            p = Path(rec.image_path)
            img_path = p if p.is_absolute() else (config.data_root / p)
            if not img_path.is_file():
                return {"id": rec.id, "status": "excluded_failed", "reason": "Image file not found", "exclusion_reason": "image_missing"}
            
            w, h = rec.image_width or 0, rec.image_height or 0
            if (w == 0 or h == 0) and img_path.is_file():
                try:
                    w, h = await asyncio.to_thread(_read_image_size, img_path)
                except Exception:
                    pass

            if w < min_res[0] or h < min_res[1]:
                return {"id": rec.id, "status": "excluded_low_quality", "reason": f"Below min resolution: {w}x{h}", "exclusion_reason": "below_min_resolution", "w": w, "h": h}
            if w > max_res[0] or h > max_res[1]:
                return {"id": rec.id, "status": "excluded_low_quality", "reason": f"Above max resolution: {w}x{h}", "exclusion_reason": "above_max_resolution", "w": w, "h": h}
            if max(w, h) / max(min(w, h), 1) > 4.0:
                return {"id": rec.id, "status": "excluded_low_quality", "reason": f"Extreme aspect ratio: {w}x{h} (>4:1)", "exclusion_reason": "extreme_aspect_ratio", "w": w, "h": h}

            try:
                quality_out = await tier1.score_quality(img_path)
            except Exception as e:
                return {"id": rec.id, "status": "excluded_failed", "reason": f"Quality scoring exception: {e}", "exclusion_reason": "inference_failed", "w": w, "h": h}
            if quality_out is None:
                return {"id": rec.id, "status": "excluded_failed", "reason": "Quality scoring failed", "exclusion_reason": "inference_failed", "w": w, "h": h}

            if quality_out.aesthetic_score is None or math.isnan(quality_out.aesthetic_score) or math.isinf(quality_out.aesthetic_score):
                return {"id": rec.id, "status": "excluded_failed", "reason": "Invalid aesthetic score (NaN/Inf)", "exclusion_reason": "inference_failed", "w": w, "h": h}

            if quality_out.aesthetic_score < threshold:
                return {"id": rec.id, "status": "excluded_low_quality", "reason": f"Aesthetic score {quality_out.aesthetic_score:.3f} < {threshold}", "exclusion_reason": "below_aesthetic_threshold", "score": quality_out.aesthetic_score, "out": quality_out.model_dump(), "w": w, "h": h}
            else:
                return {"id": rec.id, "status": "quality_scored", "score": quality_out.aesthetic_score, "out": quality_out.model_dump(), "w": w, "h": h}

        batch_concurrency = stage_cfg.get("batch_size", 128)
        sem = asyncio.Semaphore(batch_concurrency)
        async def _bounded_process(rec):
            async with sem:
                return await _process(rec)

        tasks = [asyncio.create_task(_bounded_process(rec)) for rec in records]
        results = await asyncio.gather(*tasks)

        updates: list[dict[str, Any]] = []
        for res in results:
            w, h = res.get("w"), res.get("h")
            status = res["status"]
            update_item: dict[str, Any] = {
                "id": res["id"],
                "new_status": status,
                "aesthetic_score": res.get("score"),
                "quality_output": res.get("out"),
                "image_width": w,
                "image_height": h,
            }
            if status == "quality_scored":
                processed += 1
            else:
                update_item["reason"] = res.get("reason")
                update_item["exclusion_reason"] = res.get("exclusion_reason")
                if status == "excluded_failed":
                    failed += 1
                else:
                    excluded += 1
            updates.append(update_item)

        if updates:
            manifest.bulk_update_records(updates, stage="quality")

        result.records_processed = processed
        result.records_excluded = excluded
        result.records_failed = failed
        return result
