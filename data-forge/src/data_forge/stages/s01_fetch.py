"""Stage 1: Fetch & License — download datasets with inline license verification."""

from __future__ import annotations

from typing import Any, ClassVar

from data_forge.agents.license_agent import LicenseVerificationAgent
from data_forge.config import PipelineConfig
from data_forge.data.fetcher import DatasetFetcher
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s01")


@register_stage("s01_fetch")
class FetchStage(Stage):
    name = "s01_fetch"
    requires: ClassVar[tuple[str, ...]] = ("s00_manifest_planning",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        fetcher = DatasetFetcher(config)

        threshold = config.get_stage("s01_fetch").get("license_confidence_threshold", 0.85)
        license_agent = LicenseVerificationAgent(config, confidence_threshold=threshold)

        total_fetched = 0
        total_excluded = 0

        for ds_key, ds_spec in config.datasets.items():
            log.info("fetching_dataset", dataset=ds_key, source=ds_spec.source_type)

            # Fetch the dataset files
            records = await fetcher.fetch_dataset(ds_key, ds_spec)
            if not records:
                log.warning("no_records_fetched", dataset=ds_key)
                continue

            # BUG FIX: caption_join's path (b) — e.g. Screen2Words'
            # captions referencing another dataset's images by ID rather
            # than bundling its own — returns pseudo-records marked
            # `_join_only` (see fetcher.py::_join_captions_to_existing's
            # docstring), NOT real image records. Passing those straight
            # into bulk_create_records would insert garbage rows (no
            # source_file/image_path/hash — every one of those columns
            # would be None) rather than doing what this dataset actually
            # needs: writing `source_caption` onto an EXISTING record.
            # Detect and route separately instead.
            join_only = [r for r in records if r.get("_join_only")]
            real_records = [r for r in records if not r.get("_join_only")]

            if join_only:
                matched, unmatched = self._apply_caption_joins(manifest, ds_key, join_only)
                log.info(
                    "caption_join_applied",
                    dataset=ds_key,
                    matched=matched,
                    unmatched=unmatched,
                )
                total_fetched += matched
                # No new manifest records are created for a join-only
                # source — continue to the next dataset rather than
                # falling through to bulk_create_records with an empty
                # (or partial) real_records list.
                if not real_records:
                    continue

            records = real_records

            # Bulk insert into manifest
            inserted = manifest.bulk_create_records(records, source_dataset=ds_key)
            total_fetched += inserted

            # Inline license verification. Runs whenever a dataset is still
            # "unverified" — regardless of whether license_url is set. A
            # missing URL is not "skip the check", it's an automatic fail:
            # verify_dataset_license() already handles license_url=None by
            # returning verified=False with an explicit "No license URL
            # provided" reason, which correctly routes the dataset to
            # excluded_pending_review instead of silently passing it through.
            if ds_spec.license_status == "unverified":
                tier1 = None
                if engine is not None:
                    tier1 = Tier1Engine(engine, config)

                if tier1:
                    verification = await license_agent.verify_dataset_license(
                        dataset_key=ds_key,
                        license_url=ds_spec.license_url,
                        tier1_engine=tier1,
                    )

                    # Update all records from this dataset. Scoped at the SQL
                    # level (see manifest.query_by_status_and_dataset's
                    # docstring for why the old full-table-scan-then-filter
                    # approach was a real bottleneck at this pipeline's
                    # target corpus scale).
                    ds_records = manifest.query_by_status_and_dataset("fetched", ds_key)

                    for rec in ds_records:
                        if verification["verified"]:
                            manifest.update_record(
                                record_id=rec.id,
                                stage="fetch",
                                license_verified=True,
                                license_output=verification["output"],
                            )
                        else:
                            manifest.update_record(
                                record_id=rec.id,
                                stage="fetch",
                                new_status="excluded_pending_review",
                                reason=verification["reason"],
                                license_verified=False,
                                license_output=verification["output"],
                                exclusion_reason=verification["reason"],
                            )
                            total_excluded += 1
                else:
                    log.warning(
                        "license_check_skipped",
                        dataset=ds_key,
                        reason="No inference engine available (Stage 1 runs before Tier-1 is loaded in chunk-based flow)",
                    )

        result.records_processed = total_fetched
        result.records_excluded = total_excluded
        log.info(
            "fetch_complete",
            total_fetched=total_fetched,
            total_excluded=total_excluded,
        )
        return result

    @staticmethod
    def _apply_caption_joins(
        manifest: Manifest, ds_key: str, join_pairs: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """Match caption_join pseudo-records against an already-ingested
        dataset's records by filename stem, same join strategy as
        uicrit_ingest.py uses for UICrit's ratings against RICO.

        Returns (matched, unmatched) counts.
        """
        if not join_pairs:
            return 0, 0

        join_target = join_pairs[0]["_join_target_dataset"]
        stem_to_record: dict[str, str] = {}
        for rec in manifest.query_by_dataset(join_target):
            if rec.source_file:
                stem_to_record[rec.source_file.rsplit(".", 1)[0]] = rec.id

        matched = 0
        unmatched = 0
        for pair in join_pairs:
            join_key = pair["_join_key"]
            candidates = [join_key, join_key.lstrip("0") or "0", join_key.zfill(5)]
            record_id = next((stem_to_record[c] for c in candidates if c in stem_to_record), None)
            if record_id is None:
                unmatched += 1
                continue
            manifest.update_record(record_id, "fetch_caption_join", source_caption=pair["_source_caption"])
            matched += 1

        if matched == 0 and join_pairs:
            log.error(
                "caption_join_zero_matches",
                dataset=ds_key,
                join_target=join_target,
                note="Every caption failed to join — the filename-stem "
                     "join-key assumption is very likely wrong for this "
                     "dataset's real ID format. Confirm against a live "
                     "sample rather than assuming the target dataset "
                     "simply wasn't ingested.",
            )
        return matched, unmatched
