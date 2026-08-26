"""Stage 11: Registry Watcher — reads external cron-generated report at startup."""

from __future__ import annotations

import json
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s11")


@register_stage("s11_registry_watcher")
class RegistryWatcherStage(Stage):
    """Reads the registry watcher report. Actual watching is done externally."""
    name = "s11_registry_watcher"
    requires = []

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)

        report_path = config.resolved_paths["registry_reports"] / "latest.json"
        if not report_path.exists():
            log.info("no_registry_report", note="Run `data-forge registry check`")
            return result

        report = json.loads(report_path.read_text(encoding="utf-8"))
        recommendations = report.get("recommendations", [])

        for rec in recommendations:
            action = rec.get("action", "hold")
            model = rec.get("model", "unknown")
            if action == "swap":
                log.warning("registry_swap_recommendation", model=model,
                            new_version=rec.get("new_version"),
                            reason=rec.get("reason"))
            elif action == "investigate":
                log.info("registry_investigate", model=model,
                         reason=rec.get("reason"))

        result.metadata = {"recommendations": len(recommendations)}
        result.records_processed = 1
        return result
