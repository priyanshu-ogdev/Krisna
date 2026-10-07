"""Stage 4: Safety — NSFW/harmful content classification via Tier-1."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s04")


@register_stage("s04_safety")
class SafetyStage(Stage):
    name = "s04_safety"
    requires: ClassVar[tuple[str, ...]] = ("s03_5_pii_scrub",)

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s04_safety")
        conf_threshold = stage_cfg.get("confidence_threshold", 0.8)

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status == "pii_scrubbed"]
        if not records or engine is None:
            return result

        tier1 = Tier1Engine(engine, config)
        processed = excluded = failed = 0

        async def _process(rec):
            raw_path = rec.scrubbed_image_path or rec.image_path
            if not raw_path:
                return {"id": rec.id, "status": "excluded_failed", "reason": "No image path specified", "exclusion_reason": "image_missing"}
            p = Path(raw_path)
            img_path = p if p.is_absolute() else (config.data_root / p)
            if not img_path.is_file():
                return {"id": rec.id, "status": "excluded_failed", "reason": "Image missing", "exclusion_reason": "image_missing"}

            try:
                safety_out = await tier1.classify_safety(img_path)
            except Exception as e:
                return {"id": rec.id, "status": "excluded_failed", "reason": f"Safety inference exception: {e}", "exclusion_reason": "inference_failed"}
            if safety_out is None:
                return {"id": rec.id, "status": "excluded_failed", "reason": "Safety inference failed", "exclusion_reason": "inference_failed"}

            out_dict = safety_out.model_dump()

            if safety_out.tier == "unsafe":
                return {"id": rec.id, "status": "excluded_unsafe", "reason": safety_out.rationale, "tier": "unsafe", "out": out_dict, "exclusion_reason": "unsafe_content"}
            elif safety_out.tier == "borderline" or safety_out.confidence < conf_threshold:
                return {"id": rec.id, "status": "safety_classified", "tier": "borderline", "out": out_dict}
            else:
                return {"id": rec.id, "status": "safety_classified", "tier": "safe", "out": out_dict}

        batch_concurrency = stage_cfg.get("batch_size", 128)
        sem = asyncio.Semaphore(batch_concurrency)
        async def _bounded_process(rec):
            async with sem:
                return await _process(rec)

        chunk_window = max(batch_concurrency * 2, 256)
        for c_start in range(0, len(records), chunk_window):
            chunk = records[c_start : c_start + chunk_window]
            tasks = [asyncio.create_task(_bounded_process(rec)) for rec in chunk]
            results = await asyncio.gather(*tasks)

            updates: list[dict[str, Any]] = []
            for res in results:
                status = res["status"]
                update_item: dict[str, Any] = {
                    "id": res["id"],
                    "new_status": status,
                    "safety_tier": res.get("tier"),
                    "safety_output": res.get("out"),
                }
                if status == "safety_classified":
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
                manifest.bulk_update_records(updates, stage="safety")

        result.records_processed = processed
        result.records_excluded = excluded
        result.records_failed = failed
        return result
