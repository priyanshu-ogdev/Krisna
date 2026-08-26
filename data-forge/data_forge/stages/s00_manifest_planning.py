"""Stage 0: Manifest Planning — init DB, read registry watcher report, pre-flight checks."""

from __future__ import annotations

from typing import Any

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
    requires = []

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
        storage = StorageManager(config)
        total_expected = sum(
            ds.expected_record_count for ds in config.datasets.values()
        )
        storage.pre_flight_check(total_expected)

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
            expected_records=total_expected,
            datasets=list(config.datasets.keys()),
        )

        result.records_processed = 1
        result.metadata = {
            "version": version_id,
            "expected_records": total_expected,
        }
        return result
