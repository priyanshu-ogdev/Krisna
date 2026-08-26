"""Stage 9: Heldout Carve — stratified sampling for eval set."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s09")


@register_stage("s09_heldout")
class HeldoutStage(Stage):
    name = "s09_heldout"
    requires = ["s08_encoding"]

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s09_heldout")
        fraction = stage_cfg.get("heldout_fraction", 0.05)
        stratify_by = stage_cfg.get("stratify_by", ["domain", "source_dataset"])

        records = manifest.get_records_by_ids(record_ids)
        encoded = [r for r in records if r.status == "encoded"]
        if not encoded:
            return result

        # Stratified sampling
        strata: dict[str, list] = defaultdict(list)
        for rec in encoded:
            key_parts = []
            for field in stratify_by:
                val = getattr(rec, field, None) or "unknown"
                if field == "quality_score_bucket" and rec.aesthetic_score is not None:
                    val = f"q{int(rec.aesthetic_score * 10)}"
                key_parts.append(str(val))
            key = "|".join(key_parts)
            strata[key].append(rec)

        heldout_ids: set[str] = set()
        for key, stratum in strata.items():
            n_heldout = max(1, int(len(stratum) * fraction))
            selected = random.sample(stratum, min(n_heldout, len(stratum)))
            for rec in selected:
                heldout_ids.add(rec.id)

        # Update manifest
        training = heldout = 0
        for rec in encoded:
            if rec.id in heldout_ids:
                manifest.update_record(rec.id, "heldout", new_status="heldout")
                heldout += 1
            else:
                manifest.update_record(rec.id, "heldout", new_status="training_pool")
                training += 1

        result.records_processed = training + heldout
        result.metadata = {"training_pool": training, "heldout": heldout,
                           "strata_count": len(strata)}
        log.info("heldout_complete", training=training, heldout=heldout,
                 strata=len(strata), fraction=fraction)
        return result
