"""Tier-2 escalation engine — Qwen3.6-27B dense.

Handles: borderline safety re-evaluation, ambiguous license classification,
and cross-check on Tier-1 audit disagreements. Only invoked for escalated cases.
"""

from __future__ import annotations

from pathlib import Path

from data_forge.config import PipelineConfig
from data_forge.inference.client import InferenceClient
from data_forge.inference.engine import ModelEngine
from data_forge.inference.structured_output import (
    AuditOutput,
    LicenseOutput,
    SafetyOutput,
)
from data_forge.logging_setup import get_logger

log = get_logger("inference.tier2")


class Tier2Engine:
    """High-level API for Tier-2 escalation inference.

    Tier-2 provides a higher-capability second opinion on borderline cases.
    Volume is small by design (only escalated records).
    """

    def __init__(self, engine: ModelEngine, config: PipelineConfig) -> None:
        self._engine = engine
        self._config = config
        self._client = InferenceClient(
            http_client=engine.vllm_client,
            model_id=config.models["tier2"].model_id,
            max_concurrent=8,  # Lower concurrency — Tier-2 is higher quality, lower volume
        )

    async def reclassify_safety(
        self, image_path: Path, tier1_output: dict
    ) -> SafetyOutput | None:
        """Re-evaluate a borderline safety classification with Tier-2.

        Includes Tier-1's original assessment in the prompt so Tier-2
        can make an informed second opinion.
        """
        prompt = self._config.get_prompt("safety_classification")
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"## Context: Tier-1 Assessment (for reference)\n"
            f"The initial classifier marked this as 'borderline' with the "
            f"following rationale: {tier1_output.get('rationale', 'N/A')}\n"
            f"Flags: {tier1_output.get('flags', [])}\n\n"
            f"Please provide your independent assessment."
        )
        result = await self._client.complete(
            prompt=enhanced_prompt,
            image_path=image_path,
            schema=SafetyOutput,
            max_tokens=512,
            temperature=0.05,  # Very low temp for escalation decisions
        )
        return result if isinstance(result, SafetyOutput) else None

    async def reclassify_license(
        self, license_text: str, source_url: str, tier1_output: dict
    ) -> LicenseOutput | None:
        """Re-evaluate an ambiguous license classification with Tier-2."""
        prompt = self._config.get_prompt("license_verification")
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"## Context: Initial Assessment (for reference)\n"
            f"The initial classifier returned confidence "
            f"{tier1_output.get('confidence', 'N/A')} with classification: "
            f"{tier1_output.get('license_type', 'N/A')}\n"
            f"Summary: {tier1_output.get('summary', 'N/A')}\n\n"
            f"## Source URL\n{source_url}\n\n"
            f"## License Page Content\n{license_text}\n\n"
            f"Please provide your independent assessment."
        )
        result = await self._client.complete(
            prompt=enhanced_prompt,
            schema=LicenseOutput,
            max_tokens=1024,
            temperature=0.05,
        )
        return result if isinstance(result, LicenseOutput) else None

    async def re_audit(
        self,
        image_path: Path,
        caption: str,
        structure_json: str,
        tier1_audit: dict,
    ) -> AuditOutput | None:
        """Re-audit a record where Tier-1 audit disagreed with earlier stages."""
        caption_prompt = self._config.get_prompt("audit_caption")
        structure_prompt = self._config.get_prompt("audit_structure")

        enhanced_prompt = (
            f"{caption_prompt}\n\n"
            f"## Caption to Verify\n{caption}\n\n"
            f"---\n\n"
            f"{structure_prompt}\n\n"
            f"## Structural JSON to Verify\n{structure_json}\n\n"
            f"---\n\n"
            f"## Context: Initial Audit Assessment\n"
            f"The first audit returned: overall_pass={tier1_audit.get('overall_pass')}, "
            f"confidence={tier1_audit.get('confidence')}\n"
            f"Rationale: {tier1_audit.get('rationale', 'N/A')}\n"
            f"Issues: {tier1_audit.get('accuracy_issues', [])}\n\n"
            f"Please provide your independent assessment."
        )
        result = await self._client.complete(
            prompt=enhanced_prompt,
            image_path=image_path,
            schema=AuditOutput,
            max_tokens=1024,
            temperature=0.05,
        )
        return result if isinstance(result, AuditOutput) else None
