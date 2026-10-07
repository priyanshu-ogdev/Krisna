"""Stage 11: Registry Watcher — reads external cron-generated report at startup."""

from __future__ import annotations

import json
from typing import Any, ClassVar

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
    requires: ClassVar[tuple[str, ...]] = ()

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)

        reg_dir = config.resolved_paths.get("registry_reports") if config.resolved_paths else None
        report_path = (reg_dir or (config.data_root / "registry_reports")) / "latest.json"
        if not report_path.exists():
            log.info("no_registry_report", note="Run `data-forge registry check`")
            return result

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            recommendations = report.get("recommendations", [])
        except Exception as e:
            log.error("registry_report_parse_error", error=str(e))
            result.records_failed = 1
            return result

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
