"""Stage 10.5: Critic Tier — Gemma 4 31B critique generation for DPO seeding.

New in the v10 PRD revision. Bootstraps `/ui_critique/*.parquet` at scale
using the Critic Tier (Gemma 4 31B Dense) as a UICrit-rubric VLM judge,
same job UICrit's original 983 human-annotated screens do, just generated
rather than hand-labeled — exactly the "Synthetic: Self-generated once
fine-tuned models exist" data source the PRD's §8.1 data table describes.

Scope note, stated plainly because it's easy to conflate the two: this
stage produces *scored critiques of single images* (`/ui_critique/`), not
*DPO preference pairs* (`/preference_pairs/`, ranked A-vs-B). Constructing
genuine preference pairs requires multiple candidate generations for the
same prompt/design to rank against each other — that only exists once the
product's own planner+renderer models are trained and sampling candidates,
which is downstream of and out of scope for this data-prep pipeline. What
this stage produces is the *input* DPO training will need once that loop
exists (a calibrated quality signal per image), not the pairs themselves.
Runs after s10_audit so it operates on the same audited training_pool
records, not the full unaudited pool.

Sampling safeguard: records already carrying real human ground truth
(`critique_output.critique_source == "uicrit_human"`, from
s01_5_uicrit_join) are excluded from this stage's sampling pool, so a
Gemma-4-generated critique can never silently overwrite a human one —
see the BUG FIX note in `run()` below.
"""

from __future__ import annotations

import json
import random
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.inference.critic import CriticEngine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.parquet_writer import write_records_parquet

log = get_logger("stages.s10_5")


def _has_human_critique(record: Any) -> bool:
    """Whether a record already carries real human ground truth from
    s01_5_uicrit_join, as opposed to no critique or a prior Gemma-4
    self-generated one.

    Extracted as its own function so the sampling-eligibility fix (see
    the module docstring's "Sampling safeguard" note) is directly unit-
    testable without spinning up a full async stage run + inference
    engine.
    """
    return bool(
        record.critique_output
        and record.critique_output.get("critique_source") == "uicrit_human"
    )


def _is_eligible_for_critic_sampling(record: Any) -> bool:
    """Whether a record should be considered for Gemma-4 critique
    generation in this stage.

    A record is eligible if it's reached the audited/training_pool stage
    AND does not already carry real human ground truth — see
    `_has_human_critique` and the module docstring's BUG FIX note.
    """
    return (
        record.status in ("audited", "training_pool")
        and not _has_human_critique(record)
    )


@register_stage("s10_5_critic_preference")
class CriticPreferenceStage(Stage):
    name = "s10_5_critic_preference"
    requires: ClassVar[tuple[str, ...]] = ("s10_audit",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s10_5_critic_preference")

        records = manifest.get_records_by_ids(record_ids)
        # Same "audited" filter pattern already established elsewhere in
        # this file family — only records that made it through s10_audit
        # are eligible, so critique data is generated on the same
        # already-quality-gated pool, not the raw unaudited training_pool.
        #
        # BUG FIX: this used to sample from every audited/training_pool
        # record with no regard for whether a record already carried real
        # human ground truth from s01_5_uicrit_join
        # (critique_output.critique_source == "uicrit_human"). Since
        # manifest.update_record() overwrites critique_output wholesale
        # (no merge — see manifest.py), any RICO/UICrit record randomly
        # selected here had its human label silently replaced with a
        # Gemma-4-generated one, destroying the exact "human calibration,
        # not self-distillation" signal s01_5_uicrit_join exists to
        # produce (see DATA_COMPLETENESS.md). `_is_eligible_for_critic_
        # sampling` excludes already-human-labeled records from the
        # sampling pool, preserving that signal; everything else in the
        # corpus still gets a Gemma-4 critique, since only ~983 UICrit-
        # joined records exist against a much larger training pool.
        audited_or_pool = [r for r in records if r.status in ("audited", "training_pool")]
        eligible = [r for r in audited_or_pool if _is_eligible_for_critic_sampling(r)]
        if not eligible or engine is None:
            return result

        excluded_human_labeled = len(audited_or_pool) - len(eligible)

        sample_rate = stage_cfg.get("sample_rate", 0.10)
        min_samples = stage_cfg.get("min_samples", 500)
        max_samples = stage_cfg.get("max_samples", 20000)

        target = max(int(len(eligible) * sample_rate), min(min_samples, len(eligible)))
        target = min(target, max_samples, len(eligible))
        sample = eligible if target >= len(eligible) else random.sample(eligible, target)

        log.info(
            "critic_sample_selected",
            eligible=len(eligible),
            sample_size=len(sample),
            sample_rate=sample_rate,
            excluded_human_labeled=excluded_human_labeled,
        )

        critic = CriticEngine(engine, config)
        parquet_rows: list[dict[str, Any]] = []
        succeeded = 0
        failed = 0

        for rec in sample:
            image_rel = rec.scrubbed_image_path or rec.image_path
            if not image_rel:
                failed += 1
                continue
            image_path = config.data_root / image_rel
            if not image_path.exists():
                log.warning("critique_image_missing", record_id=rec.id, path=str(image_path))
                failed += 1
                continue

            caption = rec.caption or ""
            structure_json = json.dumps(rec.structure_output) if rec.structure_output else "{}"

            try:
                critique = await critic.critique(image_path, caption, structure_json)
            except Exception as e:
                log.warning("critique_call_failed", record_id=rec.id, error=str(e))
                failed += 1
                continue

            if critique is None:
                failed += 1
                continue

            critique_dict = critique.model_dump()
            manifest.update_record(
                rec.id, "critic_preference", critique_output=critique_dict
            )

            parquet_rows.append(
                {
                    "record_id": rec.id,
                    "source_dataset": rec.source_dataset,
                    "image_path": image_rel,
                    "critique_source": "gemma4_31b",
                    **critique_dict,
                }
            )
            succeeded += 1

        # Write the accumulated shard in one call (see parquet_writer.py —
        # one shard per stage invocation, not one file per record).
        shard_path = write_records_parquet(
            parquet_rows,
            directory=config.resolved_paths["ui_critique"],
            shard_prefix="critique_gemma4",
        )

        result.records_processed = succeeded
        result.records_failed = failed
        result.metadata = {
            "sample_size": len(sample),
            "succeeded": succeeded,
            "failed": failed,
            "shard_path": str(shard_path) if shard_path else None,
        }

        log.info(
            "critic_preference_complete",
            succeeded=succeeded,
            failed=failed,
            shard=str(shard_path) if shard_path else "none (empty sample)",
        )
        return result
