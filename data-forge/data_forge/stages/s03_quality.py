"""Stage 3: Quality — aesthetic scoring and resolution filtering via Tier-1."""

from __future__ import annotations

from typing import Any

from data_forge.config import PipelineConfig
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s03")


@register_stage("s03_quality")
class QualityStage(Stage):
    name = "s03_quality"
    requires = ["s02_dedup"]

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

        for rec in records:
            # Resolution pre-filter (deterministic, no model needed)
            w, h = rec.image_width or 0, rec.image_height or 0
            if w < min_res[0] or h < min_res[1]:
                manifest.update_record(rec.id, "quality", new_status="excluded_low_quality",
                                       reason=f"Below min resolution: {w}x{h}", exclusion_reason="below_min_resolution")
                excluded += 1; continue
            if w > max_res[0] or h > max_res[1]:
                manifest.update_record(rec.id, "quality", new_status="excluded_low_quality",
                                       reason=f"Above max resolution: {w}x{h}", exclusion_reason="above_max_resolution")
                excluded += 1; continue

            # Aesthetic scoring via Tier-1
            img_path = config.data_root / (rec.image_path or "")
            if not img_path.exists():
                manifest.update_record(rec.id, "quality", new_status="excluded_failed",
                                       reason="Image file not found", exclusion_reason="image_missing")
                failed += 1; continue

            quality_out = await tier1.score_quality(img_path)
            if quality_out is None:
                manifest.update_record(rec.id, "quality", new_status="excluded_failed",
                                       reason="Quality scoring failed", exclusion_reason="inference_failed")
                failed += 1; continue

            if quality_out.aesthetic_score < threshold:
                manifest.update_record(rec.id, "quality", new_status="excluded_low_quality",
                                       reason=f"Aesthetic score {quality_out.aesthetic_score:.3f} < {threshold}",
                                       aesthetic_score=quality_out.aesthetic_score,
                                       quality_output=quality_out.model_dump(),
                                       exclusion_reason="below_aesthetic_threshold")
                excluded += 1
            else:
                manifest.update_record(rec.id, "quality", new_status="quality_scored",
                                       aesthetic_score=quality_out.aesthetic_score,
                                       quality_output=quality_out.model_dump())
                processed += 1

        result.records_processed = processed
        result.records_excluded = excluded
        result.records_failed = failed
        return result
