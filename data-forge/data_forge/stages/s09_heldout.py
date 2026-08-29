"""Stage 9: Heldout Carve — stratified sampling for eval set."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.completeness import is_encoding_complete, missing_artifacts

log = get_logger("stages.s09")


@register_stage("s09_heldout")
class HeldoutStage(Stage):
    name = "s09_heldout"
    requires: ClassVar[tuple[str, ...]] = ("s08_encoding",)

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s09_heldout")
        fraction = stage_cfg.get("heldout_fraction", 0.05)
        stratify_by = stage_cfg.get("stratify_by", ["domain", "source_dataset"])

        records = manifest.get_records_by_ids(record_ids)
        candidates = [r for r in records if r.status == "encoded"]
        if not candidates:
            return result

        # BUG FIX / COMPLETENESS GAP: status == "encoded" only means Stage
        # 8 produced AT LEAST ONE artifact for this record (see
        # s08_encoding.py's own "don't silently produce zero-artifact
        # records" rule) — not that every artifact its domain actually
        # requires is present. Without this check, a record missing (say)
        # its Qwen-Image-Edit-2511 latent due to a transient encode
        # failure would still flow into training_pool/heldout as if fully
        # processed, silently degrading whichever model's training loop
        # expects that artifact to exist. Uses the domain-aware predicate
        # in utils/completeness.py so general_design records (which
        # correctly lack vq_tokens — see s08_encoding.py's domain gate)
        # aren't wrongly excluded for "missing" an artifact they were
        # never supposed to have.
        encoded = []
        incomplete_count = 0
        for rec in candidates:
            if is_encoding_complete(rec):
                encoded.append(rec)
            else:
                incomplete_count += 1
                manifest.update_record(
                    rec.id, "heldout", new_status="excluded_failed",
                    reason=f"Incomplete encoding artifacts: missing {sorted(missing_artifacts(rec))}",
                    exclusion_reason="encoding_incomplete",
                )
        if incomplete_count:
            log.warning("heldout_excluded_incomplete", count=incomplete_count)
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
        for _key, stratum in strata.items():
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
