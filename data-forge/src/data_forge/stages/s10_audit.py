"""Stage 10: Audit Pass — VLM-as-judge rubric evaluation replacing manual spot-check."""

from __future__ import annotations

import asyncio
import json
from typing import Any, ClassVar

from data_forge.agents.audit_agent import AuditAgent
from data_forge.config import PipelineConfig
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.audit_gate import (
    eligible_training_records,
    pipeline_config_fingerprint,
    training_pool_fingerprint,
)

log = get_logger("stages.s10")


@register_stage("s10_audit")
class AuditStage(Stage):
    name = "s10_audit"
    requires: ClassVar[tuple[str, ...]] = ("s09_heldout",)

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s10_audit")

        if record_ids:
            records = manifest.get_records_by_ids(record_ids)
            training_records = [r for r in records if r.status in ("training_pool", "audited")]
        else:
            training_records = eligible_training_records(manifest, config.data_root)
        if not training_records:
            stats = {
                "total_audited": 0,
                "passed": 0,
                "failed": 0,
                "errored": 0,
                "pass_rate": 0.0,
                "threshold": stage_cfg.get("pass_rate_threshold", 0.95),
                "pipeline_passes": False,
                "reason": "No export-eligible training records were available to audit",
            }
            result.records_failed = 1
            result.metadata = stats
            self._write_report(config, manifest, stats, "")
            return result
        if engine is None:
            result.records_failed = len(training_records)
            stats = {
                "total_audited": 0,
                "passed": 0,
                "failed": 0,
                "errored": len(training_records),
                "pass_rate": 0.0,
                "threshold": stage_cfg.get("pass_rate_threshold", 0.95),
                "pipeline_passes": False,
                "reason": "Audit engine unavailable",
            }
            result.metadata = stats
            self._write_report(config, manifest, stats, "")
            return result

        agent = AuditAgent(
            config,
            sample_rate=stage_cfg.get("sample_rate", 0.03),
            min_samples=stage_cfg.get("min_samples", 200),
            pass_rate_threshold=stage_cfg.get("pass_rate_threshold", 0.95),
            random_seed=stage_cfg.get("random_seed", 42),
        )

        # Select sample
        sample = agent.select_audit_sample(training_records)
        log.info("audit_sample_selected", total_pool=len(training_records),
                 sample_size=len(sample))

        tier1 = Tier1Engine(engine, config)
        audit_results: list[tuple[str, Any]] = []
        failed_ids: list[str] = []

        batch_concurrency = stage_cfg.get("batch_size", 64)
        sem = asyncio.Semaphore(batch_concurrency)

        async def _audit_one(rec):
            async with sem:
                audit_out = await agent.audit_record(rec, tier1, config.data_root)
                return rec, audit_out

        tasks = [_audit_one(rec) for rec in sample]
        results = await asyncio.gather(*tasks)

        updates: list[dict[str, Any]] = []
        for rec, audit_out in results:
            audit_results.append((rec.id, audit_out))
            if audit_out is None or not audit_out.overall_pass:
                failed_ids.append(rec.id)
                disagreement = audit_out is not None and agent.check_ensemble_disagreement(rec, audit_out)
                updates.append({
                    "id": rec.id,
                    "new_status": "excluded_pending_review",
                    "audit_output": audit_out.model_dump() if audit_out is not None else None,
                    "reason": "Audit inference failed" if audit_out is None else "Audit failed",
                    "exclusion_reason": "audit_inference_failed" if audit_out is None else (
                        "audit_disagreement" if disagreement else "audit_failed"
                    ),
                })
            else:
                updates.append({
                    "id": rec.id,
                    "new_status": "audited",
                    "audit_output": audit_out.model_dump(),
                })

        if updates:
            manifest.bulk_update_records(updates, stage="audit")

        # Compute stats
        stats = agent.compute_audit_stats(audit_results)
        remaining_records = eligible_training_records(manifest, config.data_root)
        pool_fingerprint = training_pool_fingerprint(remaining_records, config.data_root)

        # Save audit report
        self._write_report(config, manifest, stats, pool_fingerprint)

        if failed_ids:
            log.info("audit_records_excluded", count=len(failed_ids))

        if not stats["pipeline_passes"]:
            log.error("AUDIT_THRESHOLD_FAILED", **stats)

        result.records_processed = len(sample)
        result.records_excluded = len(failed_ids)
        result.records_failed = stats["errored"]
        result.metadata = stats
        return result

    @staticmethod
    def _write_report(
        config: PipelineConfig,
        manifest: Manifest,
        stats: dict[str, Any],
        pool_fingerprint: str,
    ) -> None:
        report_path = config.resolved_paths["audit_reports"] / "latest_audit.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            **stats,
            "dataset_version": manifest.get_latest_version(),
            "training_pool_fingerprint": pool_fingerprint,
            "pipeline_fingerprint": pipeline_config_fingerprint(config),
            "sample_rate": config.get_stage("s10_audit").get("sample_rate", 0.03),
            "min_samples": config.get_stage("s10_audit").get("min_samples", 200),
        }
        temp_report = report_path.with_suffix(".json.tmp")
        temp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temp_report.replace(report_path)
