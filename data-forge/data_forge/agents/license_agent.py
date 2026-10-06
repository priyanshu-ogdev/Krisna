"""License Verification Agent — automated license text fetching, classification, and compliance brief generation.

Replaces the manual "must be checked directly, not assumed" gate from v10 §1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.inference.structured_output import LicenseOutput
from data_forge.logging_setup import get_logger
from data_forge.utils.web_fetch import fetch_page_text

log = get_logger("agents.license")


class LicenseVerificationAgent:
    """Fetches, analyzes, and classifies dataset license terms.

    Pipeline:
    1. web_fetch the canonical license URL
    2. Pass extracted text to Tier-1 model with structured extraction prompt
    3. Parse LicenseOutput
    4. High-confidence compatible → license_verified: true
    5. Low-confidence or incompatible → excluded_pending_review

    Generates a human-readable compliance brief for the review queue.
    """

    def __init__(
        self,
        config: PipelineConfig,
        confidence_threshold: float = 0.85,
    ) -> None:
        self._config = config
        self._threshold = confidence_threshold
        self._briefs_dir = config.resolved_paths.get("compliance_briefs")

    async def verify_dataset_license(
        self,
        dataset_key: str,
        license_url: str | None,
        tier1_engine: Any,
    ) -> dict[str, Any]:
        """Verify a dataset's license and return verification result.

        Returns:
            {
                "verified": bool,
                "output": LicenseOutput dict,
                "brief_path": str or None,
                "reason": str,
            }
        """
        if not license_url:
            log.warning("no_license_url", dataset=dataset_key)
            return {
                "verified": False,
                "output": None,
                "brief_path": None,
                "reason": "No license URL provided",
            }

        # Step 1: Fetch the license page
        log.info("fetching_license_page", dataset=dataset_key, url=license_url)
        page_text = await fetch_page_text(license_url)

        if not page_text:
            log.warning("license_page_fetch_failed", dataset=dataset_key, url=license_url)
            return {
                "verified": False,
                "output": {"license_type": "Not Found", "confidence": 0.0},
                "brief_path": None,
                "reason": f"Could not fetch license page: {license_url}",
            }

        # Step 2: Send to Tier-1 model for structured extraction
        log.info("analyzing_license_text", dataset=dataset_key, text_length=len(page_text))
        result = await tier1_engine.verify_license(
            license_text=page_text,
            source_url=license_url,
        )

        if result is None:
            return {
                "verified": False,
                "output": None,
                "brief_path": None,
                "reason": "Model inference failed",
            }

        output_dict = result.model_dump() if isinstance(result, LicenseOutput) else result

        # Step 3: Classification decision
        confidence = output_dict.get("confidence", 0.0)
        commercial_ok = output_dict.get("commercial_use_allowed", False)
        redistribution_ok = output_dict.get("redistribution_allowed", False)
        research_only = output_dict.get("research_only", False)

        verified = (
            confidence >= self._threshold
            and commercial_ok
            and redistribution_ok
            and not research_only
        )

        reason = self._build_reason(output_dict, verified, confidence)

        # Step 4: Generate compliance brief
        brief_path = None
        if self._briefs_dir:
            brief_path = self._generate_brief(dataset_key, license_url, output_dict, verified, reason)

        log.info(
            "license_verification_complete",
            dataset=dataset_key,
            verified=verified,
            confidence=confidence,
            license_type=output_dict.get("license_type"),
        )

        return {
            "verified": verified,
            "output": output_dict,
            "brief_path": str(brief_path) if brief_path else None,
            "reason": reason,
        }

    def _build_reason(self, output: dict, verified: bool, confidence: float) -> str:
        if verified:
            return f"License verified: {output.get('license_type')} (confidence: {confidence:.2f})"

        reasons = []
        if confidence < self._threshold:
            reasons.append(f"Low confidence ({confidence:.2f} < {self._threshold})")
        if not output.get("commercial_use_allowed"):
            reasons.append("Commercial use not allowed")
        if not output.get("redistribution_allowed"):
            reasons.append("Redistribution not allowed")
        if output.get("research_only"):
            reasons.append("Research-only restriction")

        return "; ".join(reasons) if reasons else "Unknown"

    def _generate_brief(
        self,
        dataset_key: str,
        license_url: str,
        output: dict,
        verified: bool,
        reason: str,
    ) -> Path:
        """Generate a human-readable compliance brief."""
        brief_path = self._briefs_dir / f"{dataset_key}_license_brief.md"
        brief_path.parent.mkdir(parents=True, exist_ok=True)

        status = "✅ VERIFIED" if verified else "⚠️ PENDING REVIEW"

        brief = f"""# License Compliance Brief: {dataset_key}

**Status**: {status}
**Reason**: {reason}

## Source
- **URL**: {license_url}
- **License Type**: {output.get('license_type', 'Unknown')}
- **Confidence**: {output.get('confidence', 0.0):.2f}

## Permissions
| Permission | Allowed |
|------------|---------|
| Commercial Use | {'✅' if output.get('commercial_use_allowed') else '❌'} |
| Redistribution | {'✅' if output.get('redistribution_allowed') else '❌'} |
| Attribution Required | {'⚠️ Yes' if output.get('attribution_required') else 'No'} |
| Research Only | {'⚠️ Yes' if output.get('research_only') else 'No'} |

## Key Restrictions
{chr(10).join(f'- {r}' for r in output.get('key_restrictions', [])) or '- None identified'}

## Summary
{output.get('summary', 'No summary available.')}

## Source Citation
> {output.get('source_citation', 'No citation available.')}

---
*Generated by data-forge License Verification Agent. This brief is for review —
a human must sign off on excluded_pending_review records before inclusion in
any shipped or published product.*
"""
        brief_path.write_text(brief, encoding="utf-8")
        log.info("compliance_brief_generated", path=str(brief_path))
        return brief_path
