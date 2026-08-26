"""Stage 10: Audit Pass — VLM-as-judge rubric evaluation replacing manual spot-check."""

from __future__ import annotations

import json
from typing import Any

from data_forge.agents.audit_agent import AuditAgent
from data_forge.config import PipelineConfig
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s10")


@register_stage("s10_audit")
class AuditStage(Stage):
    name = "s10_audit"
    requires = ["s09_heldout"]

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s10_audit")

        records = manifest.get_records_by_ids(record_ids)
        training_records = [r for r in records if r.status == "training_pool"]
        if not training_records or engine is None:
            return result

        agent = AuditAgent(
            config,
            sample_rate=stage_cfg.get("sample_rate", 0.03),
            min_samples=stage_cfg.get("min_samples", 200),
            pass_rate_threshold=stage_cfg.get("pass_rate_threshold", 0.95),
        )

        # Select sample
        sample = agent.select_audit_sample(training_records)
        log.info("audit_sample_selected", total_pool=len(training_records),
                 sample_size=len(sample))

        tier1 = Tier1Engine(engine, config)
        audit_results: list[tuple[str, Any]] = []
        escalation_ids: list[str] = []

        for rec in sample:
            audit_out = await agent.audit_record(rec, tier1, config.data_root)
            audit_results.append((rec.id, audit_out))

            if audit_out:
                manifest.update_record(rec.id, "audit", new_status="audited",
                                       audit_output=audit_out.model_dump())

                # Check for ensemble disagreement
                if not audit_out.overall_pass:
                    if agent.check_ensemble_disagreement(rec, audit_out):
                        escalation_ids.append(rec.id)

        # Compute stats
        stats = agent.compute_audit_stats(audit_results)

        # Save audit report
        report_path = config.resolved_paths["audit_reports"] / "latest_audit.json"
        report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        # Handle escalation (Tier-2 re-audit would be done in a separate model swap)
        if escalation_ids:
            log.info("audit_escalation_needed", count=len(escalation_ids))
            for eid in escalation_ids:
                manifest.update_record(eid, "audit",
                                       new_status="excluded_pending_review",
                                       reason="Audit ensemble disagreement",
                                       exclusion_reason="audit_disagreement")

        if not stats["pipeline_passes"]:
            log.error("AUDIT_THRESHOLD_FAILED", **stats)
            # Don't halt pipeline — log the failure for human review

        result.records_processed = len(sample)
        result.records_excluded = len(escalation_ids)
        result.metadata = stats
        return result
