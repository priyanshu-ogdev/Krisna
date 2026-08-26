"""Storage budget calculator and disk space management.

Pre-flight and mid-flight quota enforcement to prevent ENOSPC disasters
during Tri-Path encoding (Stage 8).
"""

from __future__ import annotations

import shutil
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger

log = get_logger("data.storage")


class StorageQuotaExceeded(Exception):
    """Raised when projected storage exceeds safe limit."""


class StorageManager:
    """Pre-flight and mid-flight storage quota enforcement."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._data_root = config.data_root
        self._safety_margin = config.storage.safety_margin
        self._estimates = config.storage.per_record_estimates

    def pre_flight_check(self, manifest_count: int) -> dict[str, Any]:
        """Calculate projected storage needs and validate against available space.

        Raises StorageQuotaExceeded if projected > (available * (1 - safety_margin)).
        """
        projected = self.calculate_projected_size(manifest_count)
        available = self.get_available_bytes()
        safe_limit = int(available * (1.0 - self._safety_margin))

        report = {
            "manifest_count": manifest_count,
            "projected_bytes": projected,
            "projected_gb": round(projected / 1e9, 2),
            "available_bytes": available,
            "available_gb": round(available / 1e9, 2),
            "safe_limit_bytes": safe_limit,
            "safe_limit_gb": round(safe_limit / 1e9, 2),
            "safety_margin": self._safety_margin,
            "pass": projected <= safe_limit,
        }

        log.info("storage_preflight", **report)

        if not report["pass"]:
            raise StorageQuotaExceeded(
                f"Projected size {report['projected_gb']:.2f}GB exceeds safe limit "
                f"{report['safe_limit_gb']:.2f}GB "
                f"(available: {report['available_gb']:.2f}GB, "
                f"margin: {self._safety_margin:.0%}). "
                f"Records: {manifest_count}"
            )

        return report

    def mid_flight_check(self) -> dict[str, Any]:
        """Check current disk usage during pipeline execution."""
        usage = shutil.disk_usage(str(self._data_root))
        used_pct = usage.used / usage.total if usage.total > 0 else 0.0
        free_pct = usage.free / usage.total if usage.total > 0 else 0.0

        report = {
            "total_gb": round(usage.total / 1e9, 2),
            "used_gb": round(usage.used / 1e9, 2),
            "free_gb": round(usage.free / 1e9, 2),
            "used_pct": round(used_pct * 100, 1),
            "free_pct": round(free_pct * 100, 1),
            "safe": free_pct > self._safety_margin,
        }

        if not report["safe"]:
            log.warning("storage_low", **report)

        return report

    def calculate_projected_size(self, record_count: int) -> int:
        """Estimate total bytes needed for record_count records through all stages."""
        per_record = sum(self._estimates.values()) if self._estimates else 1_277_000
        return record_count * per_record

    def get_available_bytes(self) -> int:
        """Get available disk space at DATA_ROOT."""
        self._data_root.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(str(self._data_root)).free

    def get_directory_sizes(self) -> dict[str, float]:
        """Calculate actual sizes of each pipeline subdirectory in GB."""
        sizes: dict[str, float] = {}
        for name, path in self._config.resolved_paths.items():
            if path.exists():
                total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                sizes[name] = round(total / 1e9, 3)
            else:
                sizes[name] = 0.0
        return sizes
