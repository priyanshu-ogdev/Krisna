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
from data_forge.stages.split_identity import source_split_identity
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
        stratify_by = [
            "source_dataset" if field == "source" else field
            for field in stage_cfg.get("stratify_by", ["domain", "source_dataset"])
        ]
        source_disjoint = bool(stage_cfg.get("source_disjoint", True))

        records = manifest.get_records_by_ids(record_ids)
        candidates = [r for r in records if r.status in ("encoded", "training_pool", "heldout")]
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
        # in utils/completeness.py. As of the maskgit_vq removal (sync
        # audit item #1), ui_first and general_design records require the
        # same artifact set — there is no longer a domain-specific extra
        # to exclude anyone for lacking.
        encoded = []
        incomplete_count = 0
        incomplete_updates: list[dict[str, Any]] = []
        for rec in candidates:
            if is_encoding_complete(rec):
                encoded.append(rec)
            else:
                incomplete_count += 1
                incomplete_updates.append({
                    "id": rec.id,
                    "new_status": "excluded_failed",
                    "reason": f"Incomplete encoding artifacts: missing {sorted(missing_artifacts(rec))}",
                    "exclusion_reason": "encoding_incomplete",
                })
        if incomplete_updates:
            manifest.bulk_update_records(incomplete_updates, stage="heldout")
        if incomplete_count:
            log.warning("heldout_excluded_incomplete", count=incomplete_count)
        if not encoded:
            return result

        # Source-disjoint splitting is the default for model-training data.
        # Sampling individual records from each source would put the same
        # upstream identity in both train and held-out sets, making leakage
        # possible even when the random seed is fixed.  Keep the legacy
        # record-level stratification as an explicit opt-out for experiments
        # that do not claim a source-disjoint evaluation split.
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

        seed = stage_cfg.get("random_seed", 42)
        rng = random.Random(seed)
        heldout_ids: set[str] = set()
        heldout_source_identities: list[str] = []
        if source_disjoint:
            by_source: dict[str, list] = defaultdict(list)
            for rec in encoded:
                source_identity = source_split_identity(rec)
                by_source[source_identity].append(rec)

            source_order = sorted(by_source)
            rng.shuffle(source_order)
            target_count = max(1, int(len(encoded) * fraction))
            selected_count = 0
            for source_identity in source_order:
                heldout_source_identities.append(source_identity)
                selected_count += len(by_source[source_identity])
                heldout_ids.update(rec.id for rec in by_source[source_identity])
                if selected_count >= target_count:
                    break
        else:
            for _key, stratum in strata.items():
                n_heldout = max(1, int(len(stratum) * fraction))
                selected = rng.sample(stratum, min(n_heldout, len(stratum)))
                for rec in selected:
                    heldout_ids.add(rec.id)

        # Update manifest via single atomic batch
        training = heldout = 0
        split_updates: list[dict[str, Any]] = []
        for rec in encoded:
            if rec.id in heldout_ids:
                split_updates.append({"id": rec.id, "new_status": "heldout"})
                heldout += 1
            else:
                split_updates.append({"id": rec.id, "new_status": "training_pool"})
                training += 1

        if split_updates:
            manifest.bulk_update_records(split_updates, stage="heldout")

        result.records_processed = training + heldout
        result.metadata = {"training_pool": training, "heldout": heldout,
                           "strata_count": len(strata),
                           "source_disjoint": source_disjoint,
                           "heldout_source_identities": heldout_source_identities}
        log.info("heldout_complete", training=training, heldout=heldout,
                 strata=len(strata), fraction=fraction,
                 source_disjoint=source_disjoint,
                 heldout_source_identities=heldout_source_identities)
        return result
