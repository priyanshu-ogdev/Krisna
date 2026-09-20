"""Model & Source Registry Watcher — the automated version of manual model discovery.

Designed to be triggered by external cron (`data-forge registry check`).
Checks for new model releases, runs eval comparisons, and generates
swap/hold/investigate recommendations.
"""

from __future__ import annotations

import time
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.registry.hf_checker import check_model_updates
from data_forge.registry.report import generate_report

log = get_logger("registry.watcher")


class RegistryWatcher:
    """Polls HuggingFace and GitHub for model/dataset updates."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    async def check_all(self) -> dict[str, Any]:
        """Run all registry checks and return a structured report."""
        log.info("registry_check_starting")
        start = time.monotonic()

        recommendations: list[dict[str, Any]] = []

        # Check each pinned model for updates
        for key, spec in self._config.models.items():
            if spec.role in ("product_planner",):
                continue  # Don't check models not managed by data-forge

            try:
                update_info = await check_model_updates(
                    model_id=spec.model_id,
                    current_revision=spec.revision,
                )

                if update_info["has_update"]:
                    recommendations.append({
                        "model": key,
                        "model_id": spec.model_id,
                        "action": "investigate",
                        "current_revision": spec.revision,
                        "new_revision": update_info.get("latest_revision"),
                        "new_version": update_info.get("latest_tag"),
                        "reason": update_info.get("reason", "New version available"),
                        "checked_at": time.time(),
                    })
                else:
                    log.info("model_up_to_date", model=key, model_id=spec.model_id)

            except Exception as e:
                log.warning("model_check_failed", model=key, error=str(e))
                recommendations.append({
                    "model": key,
                    "model_id": spec.model_id,
                    "action": "investigate",
                    "reason": f"Check failed: {e}",
                    "checked_at": time.time(),
                })

        # Check encoder models too
        for key, spec in self._config.encoders.items():
            try:
                update_info = await check_model_updates(
                    model_id=spec.model_id,
                    current_revision=spec.revision,
                )
                if update_info["has_update"]:
                    recommendations.append({
                        "model": f"encoder_{key}",
                        "model_id": spec.model_id,
                        "action": "investigate",
                        "current_revision": spec.revision,
                        "new_revision": update_info.get("latest_revision"),
                        "reason": update_info.get("reason", "New encoder version"),
                        "checked_at": time.time(),
                    })
            except Exception as e:
                log.warning("encoder_check_failed", encoder=key, error=str(e))

        duration = time.monotonic() - start
        report = generate_report(recommendations, duration)

        log.info(
            "registry_check_complete",
            duration_s=round(duration, 2),
            recommendations=len(recommendations),
            swaps=sum(1 for r in recommendations if r["action"] == "swap"),
        )

        return report
