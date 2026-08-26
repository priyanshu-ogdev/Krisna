"""Stage 1: Fetch & License — download datasets with inline license verification."""

from __future__ import annotations

from typing import Any

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
    requires = ["s00_manifest_planning"]

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

            # Bulk insert into manifest
            inserted = manifest.bulk_create_records(records, source_dataset=ds_key)
            total_fetched += inserted

            # Inline license verification (if URL available and status unverified)
            if ds_spec.license_status == "unverified" and ds_spec.license_url:
                tier1 = None
                if engine is not None:
                    tier1 = Tier1Engine(engine, config)

                if tier1:
                    verification = await license_agent.verify_dataset_license(
                        dataset_key=ds_key,
                        license_url=ds_spec.license_url,
                        tier1_engine=tier1,
                    )

                    # Update all records from this dataset
                    ds_records = manifest.query_by_status("fetched")
                    ds_records = [r for r in ds_records if r.source_dataset == ds_key]

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
