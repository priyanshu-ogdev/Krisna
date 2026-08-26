"""Stage 4.5: Escalation — Tier-2 second opinion on borderline records."""

from __future__ import annotations

from typing import Any

from data_forge.config import PipelineConfig
from data_forge.inference.tier2 import Tier2Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s04_5")


@register_stage("s04_5_escalation")
class EscalationStage(Stage):
    name = "s04_5_escalation"
    requires = ["s04_safety"]

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        records = manifest.get_records_by_ids(record_ids)
        borderline = [r for r in records if r.safety_tier == "borderline"]

        if not borderline or engine is None:
            return result

        tier2 = Tier2Engine(engine, config)
        resolved = escalated = 0

        for rec in borderline:
            img_path = config.data_root / (rec.scrubbed_image_path or rec.image_path or "")
            if not img_path.exists():
                continue

            tier1_output = rec.safety_output or {}
            t2_result = await tier2.reclassify_safety(img_path, tier1_output)
            if t2_result is None:
                # Can't get second opinion — leave as pending review
                manifest.update_record(rec.id, "escalation", new_status="excluded_pending_review",
                                       reason="Tier-2 inference failed", exclusion_reason="escalation_failed")
                escalated += 1; continue

            # Two-model agreement logic
            t1_tier = tier1_output.get("tier", "borderline")
            t2_tier = t2_result.tier

            if t2_tier == "safe":
                # Tier-2 says safe → override Tier-1's borderline → proceed
                manifest.update_record(rec.id, "escalation", safety_tier="safe",
                                       safety_output=t2_result.model_dump())
                resolved += 1
            elif t2_tier == "unsafe":
                # Both agree on unsafe direction → exclude
                manifest.update_record(rec.id, "escalation", new_status="excluded_unsafe",
                                       reason=f"Tier-2 confirmed unsafe: {t2_result.rationale}",
                                       safety_tier="unsafe", safety_output=t2_result.model_dump(),
                                       exclusion_reason="tier2_confirmed_unsafe")
                escalated += 1
            else:
                # Persistent disagreement (both say borderline) → pending review
                manifest.update_record(rec.id, "escalation", new_status="excluded_pending_review",
                                       reason="Persistent borderline after Tier-2 review",
                                       safety_output=t2_result.model_dump(),
                                       exclusion_reason="persistent_borderline")
                escalated += 1

        result.records_processed = resolved
        result.records_excluded = escalated
        log.info("escalation_complete", resolved=resolved, escalated=escalated)
        return result
