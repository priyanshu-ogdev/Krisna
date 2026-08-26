"""Stage 7: Routing — domain tagging and ratio-enforced shard assignment."""

from __future__ import annotations

from typing import Any

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
    requires = ["s06_structure"]

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s07_routing")

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status == "structured"]
        if not records:
            return result

        # Tag domains
        for rec in records:
            domain = tag_domain(rec)
            manifest.update_record(rec.id, "routing", domain=domain)
            rec.domain = domain  # Update in-memory for router

        # Route with ratio enforcement
        ratio = stage_cfg.get("ui_first_ratio", 0.70)
        overflow = stage_cfg.get("overflow_action", "exclude")
        router = ShardRouter(ui_first_ratio=ratio, overflow_action=overflow)

        assignments = router.route(records)
        routed = excluded = 0

        for assignment in assignments:
            if assignment["status"] == "routed":
                manifest.update_record(assignment["record_id"], "routing",
                                       new_status="routed",
                                       shard_id=assignment["shard_id"])
                routed += 1
            elif assignment["status"] == "overflow_excluded":
                manifest.update_record(assignment["record_id"], "routing",
                                       new_status="overflow_excluded",
                                       exclusion_reason="ratio_overflow")
                excluded += 1

        result.records_processed = routed
        result.records_excluded = excluded
        return result
