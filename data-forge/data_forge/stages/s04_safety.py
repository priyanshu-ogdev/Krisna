"""Stage 4: Safety — NSFW/harmful content classification via Tier-1."""

from __future__ import annotations

from typing import Any

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
    requires = ["s03_5_pii_scrub"]

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

        for rec in records:
            img_path = config.data_root / (rec.scrubbed_image_path or rec.image_path or "")
            if not img_path.exists():
                manifest.update_record(rec.id, "safety", new_status="excluded_failed",
                                       reason="Image missing", exclusion_reason="image_missing")
                failed += 1; continue

            safety_out = await tier1.classify_safety(img_path)
            if safety_out is None:
                manifest.update_record(rec.id, "safety", new_status="excluded_failed",
                                       reason="Safety inference failed", exclusion_reason="inference_failed")
                failed += 1; continue

            out_dict = safety_out.model_dump()

            if safety_out.tier == "unsafe":
                manifest.update_record(rec.id, "safety", new_status="excluded_unsafe",
                                       reason=safety_out.rationale, safety_tier="unsafe",
                                       safety_output=out_dict, exclusion_reason="unsafe_content")
                excluded += 1
            elif safety_out.tier == "borderline" or safety_out.confidence < conf_threshold:
                # Mark for Tier-2 escalation — stays in pipeline but flagged
                manifest.update_record(rec.id, "safety", new_status="safety_classified",
                                       safety_tier="borderline", safety_output=out_dict)
                processed += 1
            else:
                manifest.update_record(rec.id, "safety", new_status="safety_classified",
                                       safety_tier="safe", safety_output=out_dict)
                processed += 1

        result.records_processed = processed
        result.records_excluded = excluded
        result.records_failed = failed
        return result
