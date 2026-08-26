"""Automated Audit Agent — replaces manual spot-check (v12 Stage 10).

Runs every sampled record through a fixed rubric:
- Does the caption match the image?
- Does the structural JSON match the image?
- Any quality/safety issues the earlier gates might have missed?

Implements ensemble disagreement escalation to Tier-2.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.inference.structured_output import AuditOutput
from data_forge.logging_setup import get_logger
from data_forge.manifest import ManifestRecord

log = get_logger("agents.audit")


class AuditAgent:
    """Automated audit pass with ensemble disagreement escalation."""

    def __init__(
        self,
        config: PipelineConfig,
        sample_rate: float = 0.03,
        min_samples: int = 200,
        pass_rate_threshold: float = 0.95,
    ) -> None:
        self._config = config
        self._sample_rate = sample_rate
        self._min_samples = min_samples
        self._pass_threshold = pass_rate_threshold

    def select_audit_sample(
        self, records: list[ManifestRecord]
    ) -> list[ManifestRecord]:
        """Select records for audit, respecting sample rate and minimum."""
        target = max(
            int(len(records) * self._sample_rate),
            min(self._min_samples, len(records)),
        )
        if target >= len(records):
            return list(records)
        return random.sample(records, target)

    async def audit_record(
        self,
        record: ManifestRecord,
        tier1_engine: Any,
        data_root: Path,
    ) -> AuditOutput | None:
        """Audit a single record using Tier-1 model."""
        image_path = data_root / (record.scrubbed_image_path or record.image_path or "")
        if not image_path.exists():
            log.warning("audit_image_missing", record_id=record.id, path=str(image_path))
            return None

        caption = record.caption or ""
        structure_json = json.dumps(record.structure_output) if record.structure_output else "{}"

        result = await tier1_engine.audit_record(
            image_path=image_path,
            caption=caption,
            structure_json=structure_json,
        )
        return result

    def check_ensemble_disagreement(
        self,
        record: ManifestRecord,
        audit_result: AuditOutput,
    ) -> bool:
        """Check if the audit result disagrees with earlier pipeline stages.

        Returns True if there's a disagreement that needs Tier-2 escalation.
        """
        disagreements = []

        # Audit says caption doesn't match but Stage 5 produced it with high confidence
        if not audit_result.caption_matches_image:
            if record.caption_output:
                stage5_conf = record.caption_output.get("confidence", 0)
                if stage5_conf > 0.8:
                    disagreements.append("caption_confidence_mismatch")

        # Audit says structure doesn't match but Stage 6 produced it
        if not audit_result.structure_matches_image:
            if record.structure_output:
                disagreements.append("structure_mismatch")

        # Audit found safety issues but Stage 4 said "safe"
        if audit_result.safety_issues and record.safety_tier == "safe":
            disagreements.append("safety_mismatch")

        if disagreements:
            log.info(
                "ensemble_disagreement",
                record_id=record.id,
                disagreements=disagreements,
            )
            return True

        return False

    def compute_audit_stats(
        self, results: list[tuple[str, AuditOutput | None]]
    ) -> dict[str, Any]:
        """Compute aggregate audit statistics.

        Returns stats dict and whether the pipeline passes the threshold.
        """
        total = len(results)
        passed = sum(
            1 for _, r in results if r is not None and r.overall_pass
        )
        failed = sum(
            1 for _, r in results if r is not None and not r.overall_pass
        )
        errored = sum(1 for _, r in results if r is None)

        pass_rate = passed / total if total > 0 else 0.0

        stats = {
            "total_audited": total,
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "pass_rate": round(pass_rate, 4),
            "threshold": self._pass_threshold,
            "pipeline_passes": pass_rate >= self._pass_threshold,
        }

        if not stats["pipeline_passes"]:
            log.error(
                "audit_threshold_failed",
                pass_rate=stats["pass_rate"],
                threshold=self._pass_threshold,
            )
        else:
            log.info("audit_passed", **stats)

        return stats
