"""Stage 4.5: Escalation — Tier-2 second opinion on borderline records."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

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
    requires: ClassVar[tuple[str, ...]] = ("s04_safety",)

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        borderline_ids = manifest.get_borderline_ids(record_ids)
        if not borderline_ids or engine is None:
            return result

        records = manifest.get_records_by_ids(borderline_ids)
        borderline = [r for r in records if r.safety_tier == "borderline"]
        if not borderline:
            return result

        tier2 = Tier2Engine(engine, config)
        resolved = escalated = 0
        updates: list[dict[str, Any]] = []
        sem = asyncio.Semaphore(16)

        async def _eval_rec(rec):
            raw_path = rec.scrubbed_image_path or rec.image_path
            if not raw_path:
                return None
            p = Path(raw_path)
            img_path = p if p.is_absolute() else (config.data_root / p)
            if not img_path.is_file():
                return None
            tier1_output = rec.safety_output or {}
            async with sem:
                t2_result = await tier2.reclassify_safety(img_path, tier1_output)
            return rec, tier1_output, t2_result

        eval_tasks = [_eval_rec(r) for r in borderline]
        eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        for rec, res in zip(borderline, eval_results):
            if isinstance(res, Exception):
                log.error("tier2_escalation_exception", record_id=rec.id, error=str(res))
                updates.append({
                    "id": rec.id,
                    "new_status": "excluded_pending_review",
                    "reason": f"Tier-2 inference exception: {res}",
                    "exclusion_reason": "escalation_exception",
                })
                escalated += 1
                continue
            if not res:
                updates.append({
                    "id": rec.id,
                    "new_status": "excluded_pending_review",
                    "reason": "Tier-2 image missing or unreadable",
                    "exclusion_reason": "image_missing",
                })
                escalated += 1
                continue

            _, tier1_output, t2_result = res
            if t2_result is None:
                updates.append({
                    "id": rec.id,
                    "new_status": "excluded_pending_review",
                    "reason": "Tier-2 inference failed",
                    "exclusion_reason": "escalation_failed",
                })
                escalated += 1
                continue

            t2_tier = t2_result.tier
            if t2_tier == "safe":
                updates.append({
                    "id": rec.id,
                    "safety_tier": "safe",
                    "safety_output": t2_result.model_dump(),
                })
                resolved += 1
            elif t2_tier == "unsafe":
                updates.append({
                    "id": rec.id,
                    "new_status": "excluded_unsafe",
                    "reason": f"Tier-2 confirmed unsafe: {t2_result.rationale}",
                    "safety_tier": "unsafe",
                    "safety_output": t2_result.model_dump(),
                    "exclusion_reason": "tier2_confirmed_unsafe",
                })
                escalated += 1
            else:
                updates.append({
                    "id": rec.id,
                    "new_status": "excluded_pending_review",
                    "reason": "Persistent borderline after Tier-2 review",
                    "safety_output": t2_result.model_dump(),
                    "exclusion_reason": "persistent_borderline",
                })
                escalated += 1

        if updates:
            manifest.bulk_update_records(updates, stage="escalation")

        result.records_processed = resolved
        result.records_excluded = escalated
        log.info("escalation_complete", resolved=resolved, escalated=escalated)
        return result
