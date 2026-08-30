"""Stage 1.5: UICrit Join — attach real human critique/ratings to RICO records.

Runs after s01_fetch (needs rico_core/rico_semantic already ingested and
the UICrit repo already cloned via `_fetch_github`'s annotation_only path
— see fetcher.py and data/uicrit_ingest.py for why UICrit never produces
its own image records).

UICrit's ~983 human ratings are the one real UI-critique ground-truth
signal in this pipeline. Records this stage successfully joins get
`critique_output.critique_source == "uicrit_human"`. Two consumers read
this: s12_model_data_export.py's `planner_rag_corpus/` export (the
planner ships frozen and retrieves this text at inference time rather
than being fine-tuned on it) and, historically, the now-removed Gemma-4
critic-tier calibration path — that model is a frozen, on-demand product
feature now, not trained by data-forge at all, so this stage's only live
consumer today is the RAG export. Kept as its own stage regardless of the
critic tier's removal, since the RAG corpus still needs the same
join/dedup treatment.
"""

from __future__ import annotations

from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.data.uicrit_ingest import parse_uicrit_annotations, to_critique_output_dict
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s01_5")

_RICO_SOURCE_DATASETS = ("rico_core", "rico_semantic")


@register_stage("s01_5_uicrit_join")
class UICritJoinStage(Stage):
    name = "s01_5_uicrit_join"
    requires: ClassVar[tuple[str, ...]] = ("s01_fetch",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)

        uicrit_spec = config.datasets.get("uicrit")
        if uicrit_spec is None or not uicrit_spec.annotation_only:
            log.info("uicrit_join_skipped", reason="uicrit not configured as annotation_only")
            return result

        repo_dir = config.data_root / "raw" / "uicrit" / "repo"
        if not repo_dir.exists():
            log.warning(
                "uicrit_repo_not_found",
                path=str(repo_dir),
                note="s01_fetch must run for the 'uicrit' dataset before this stage.",
            )
            return result

        annotations = parse_uicrit_annotations(repo_dir)
        if not annotations:
            log.error(
                "uicrit_join_no_annotations",
                note="parse_uicrit_annotations returned nothing — see its own "
                     "error logs above for which candidate file/columns failed. "
                     "This stage cannot proceed; the join is a hard stop, not a "
                     "silent zero-match success.",
            )
            return result

        # Build a filename-stem -> record_id lookup across both RICO
        # sources. UICrit is ~983 rows and RICO's ingested pool is at
        # most in the low hundred-thousands at this pipeline's target
        # scale — an in-memory dict is the right tool here, not another
        # per-row SQL query.
        stem_to_record: dict[str, str] = {}
        for source_dataset in _RICO_SOURCE_DATASETS:
            for rec in manifest.query_by_dataset(source_dataset):
                if not rec.source_file:
                    continue
                stem = rec.source_file.rsplit(".", 1)[0]
                stem_to_record[stem] = rec.id

        log.info(
            "uicrit_join_starting",
            annotations=len(annotations),
            rico_pool_size=len(stem_to_record),
        )

        matched = 0
        unmatched = 0
        for ann in annotations:
            join_key = ann["rico_join_key"]
            # Try the key as-is, then with common numeric-padding variants,
            # since RICO filenames and UICrit's ID column may not agree on
            # zero-padding or a leading identifier prefix. UNVERIFIED which
            # of these the real data needs — trying a small set of cheap
            # variants is safer than assuming one and failing silently.
            candidates = [join_key, join_key.lstrip("0") or "0", join_key.zfill(5)]
            record_id = next((stem_to_record[c] for c in candidates if c in stem_to_record), None)

            if record_id is None:
                unmatched += 1
                continue

            critique_dict = to_critique_output_dict(ann)
            manifest.update_record(record_id, "uicrit_join", critique_output=critique_dict)
            matched += 1

        result.records_processed = matched
        result.records_failed = unmatched
        result.metadata = {
            "annotations_total": len(annotations),
            "matched": matched,
            "unmatched": unmatched,
            "match_rate": round(matched / len(annotations), 3) if annotations else 0.0,
        }

        log.info(
            "uicrit_join_complete",
            matched=matched,
            unmatched=unmatched,
            match_rate=result.metadata["match_rate"],
        )
        if matched == 0:
            log.error(
                "uicrit_join_zero_matches",
                note="Every annotation failed to join against a RICO record. "
                     "The join-key assumption in this stage (filename stem "
                     "match, with light zero-padding variants) is very "
                     "likely wrong for the real data — confirm UICrit's "
                     "actual ID format against a live sample before "
                     "assuming RICO simply wasn't ingested.",
            )
        return result
