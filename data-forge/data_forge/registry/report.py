"""Registry watcher report generator."""

from __future__ import annotations

import time
from typing import Any


def generate_report(
    recommendations: list[dict[str, Any]],
    duration_seconds: float,
) -> dict[str, Any]:
    """Generate a structured watcher report."""
    return {
        "report_version": "1.0",
        "generated_at": time.time(),
        "check_duration_seconds": round(duration_seconds, 2),
        "total_checks": len(recommendations),
        "swaps": [r for r in recommendations if r.get("action") == "swap"],
        "holds": [r for r in recommendations if r.get("action") == "hold"],
        "investigations": [r for r in recommendations if r.get("action") == "investigate"],
        "recommendations": recommendations,
        "summary": _build_summary(recommendations),
    }


def _build_summary(recommendations: list[dict[str, Any]]) -> str:
    swaps = sum(1 for r in recommendations if r.get("action") == "swap")
    investigations = sum(1 for r in recommendations if r.get("action") == "investigate")

    if swaps:
        return f"{swaps} model(s) recommended for swap. Review before next pipeline run."
    if investigations:
        return f"{investigations} model(s) need investigation. No immediate action required."
    return "All models up to date. No action needed."
