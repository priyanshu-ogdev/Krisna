"""Stage 7: Routing — domain tagging and ratio-enforced shard assignment."""

from __future__ import annotations

from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.data.domain_tagger import tag_domain
from data_forge.data.shard_router import ShardRouter
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s07")


@register_stage("s07_routing")
class RoutingStage(Stage):
    name = "s07_routing"
    requires: ClassVar[tuple[str, ...]] = ("s06_structure",)

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s07_routing")

        ratio = stage_cfg.get("ui_first_ratio", 0.70)
        overflow = stage_cfg.get("overflow_action", "exclude")
        shard_size = stage_cfg.get("records_per_shard", 5000)
        routed = excluded = 0
        ui_routed = general_routed = 0
        updates: list[dict[str, Any]] = []

        # Keep memory bounded: the full corpus can exceed one million
        # records, while the manifest materializes JSON payloads per row.
        for start in range(0, len(record_ids), 5000):
            records = manifest.get_records_by_ids(record_ids[start : start + 5000])
            records = [r for r in records if r.status in ("structured", "recaptioned")]
            if not records:
                continue

            domain_map: dict[str, str] = {}
            for rec in records:
                domain = tag_domain(rec)
                domain_map[rec.id] = domain
                rec.domain = domain

            router = ShardRouter(
                ui_first_ratio=ratio,
                overflow_action=overflow,
                records_per_shard=shard_size,
            )
            assignments = router.route(records)
            local_routed = 0

            for assignment in assignments:
                rec_id = assignment["record_id"]
                status = assignment["status"]
                global_shard = (routed + local_routed) // shard_size
                item: dict[str, Any] = {
                    "id": rec_id,
                    "new_status": status,
                    "domain": domain_map[rec_id],
                    "shard_id": f"shard_{global_shard:04d}",
                }
                local_routed += 1
                if domain_map[rec_id] == "ui_first":
                    ui_routed += 1
                else:
                    general_routed += 1
                updates.append(item)

                if len(updates) >= 500:
                    manifest.bulk_update_records(updates, stage="routing")
                    updates.clear()
            routed += local_routed

        if updates:
            manifest.bulk_update_records(updates, stage="routing")
        result.records_processed = routed
        result.records_excluded = excluded
        result.metadata = {
            "ui_routed": ui_routed,
            "general_routed": general_routed,
            "actual_ui_ratio": ui_routed / routed if routed else 0.0,
            "target_ui_ratio_for_training": ratio,
            "records_per_shard": shard_size,
        }
        return result
