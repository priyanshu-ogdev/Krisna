"""Stage 0: Manifest Planning — init DB, read registry watcher report, pre-flight checks."""

from __future__ import annotations

from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.data.storage import StorageManager
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s00")


@register_stage("s00_manifest_planning")
class ManifestPlanningStage(Stage):
    name = "s00_manifest_planning"
    requires: ClassVar[tuple[str, ...]] = ()

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)

        # 1. Read registry watcher report (if available)
        report_path = config.resolved_paths["registry_reports"] / "latest.json"
        if report_path.exists():
            import json
            report = json.loads(report_path.read_text(encoding="utf-8"))
            recommendations = report.get("recommendations", [])
            swaps = [r for r in recommendations if r.get("action") == "swap"]
            if swaps:
                log.warning(
                    "registry_swaps_pending",
                    count=len(swaps),
                    models=[s.get("model") for s in swaps],
                )
            else:
                log.info("registry_report_clean", recommendation_count=len(recommendations))
        else:
            log.info("no_registry_report", note="Run `data-forge registry check` to create one")

        # 2. Pre-flight storage check
        # BUG FIX: this used to sum raw `expected_record_count` (PD12M's
        # full 12.4M, CC12M's full 12.4M, etc.) with no regard for
        # `fetch_config.sample_size` actually capping downloads to 200K/
        # 150K, or for annotation_only/caption_join/preference_pair/
        # eval_reference sources never producing standalone image records
        # at all. That
        # projected ~26M records / ~33TB against the PRD's real ~100K-
        # 500K / ~3TB target (§8.3), which would false-fail
        # `pre_flight_check` before Stage 1 ever ran on any normal
        # workstation disk. `storage_relevant_record_count()` reconciles
        # both — see its docstring in config.py.
        storage = StorageManager(config)
        total_expected_raw = sum(
            ds.expected_record_count for ds in config.datasets.values()
        )
        total_expected_effective = sum(
            ds.storage_relevant_record_count() for ds in config.datasets.values()
        )
        total_preference_pairs = sum(
            ds.preference_pair_relevant_count() for ds in config.datasets.values()
        )
        log.info(
            "storage_projection_basis",
            raw_expected_record_count_sum=total_expected_raw,
            effective_storage_relevant_count=total_expected_effective,
            preference_pair_count=total_preference_pairs,
            note="Pre-flight check uses the effective (sample_size-capped, "
                 "image-record-only) count for the main corpus and a "
                 "separate preference-pair count (Pick-a-Pic v2/HPDv2/"
                 "DesignSense-10k/DesignPref) budgeted at its own, larger "
                 "per-item storage cost — not the raw sum of every "
                 "dataset's full corpus size, and not combined into one "
                 "count with a single per-record constant.",
        )
        storage.pre_flight_check(total_expected_effective, total_preference_pairs)

        # 3. Generate dataset version
        import time
        version_num = 1
        latest = manifest.get_latest_version()
        if latest:
            try:
                version_num = int(latest.split("_v")[-1]) + 1
            except (ValueError, IndexError):
                version_num = int(time.time())

        version_id = f"{config.dataset_version_prefix}{version_num:03d}"
        manifest.create_dataset_version(version_id, notes="Pipeline run started")

        log.info(
            "manifest_planning_complete",
            version=version_id,
            expected_records_effective=total_expected_effective,
            expected_records_raw=total_expected_raw,
            datasets=list(config.datasets.keys()),
        )

        result.records_processed = 1
        result.metadata = {
            "version": version_id,
            "expected_records": total_expected_effective,
            "expected_records_raw_corpus_sum": total_expected_raw,
        }
        return result
