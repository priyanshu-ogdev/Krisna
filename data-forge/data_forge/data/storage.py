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

    # Which per_record_estimates keys apply to a single image-manifest
    # record vs. a single preference pair — kept as two disjoint sets
    # deliberately. calculate_projected_size() sums each against its own
    # count rather than summing every key in per_record_estimates against
    # one combined count, which would multiply preference-pair storage
    # (two source images + two latents, a materially larger per-item cost)
    # by the image-record count instead of the pair count — silently
    # inflating the projection by orders of magnitude at this corpus's
    # scale and false-failing pre_flight_check on a perfectly safe disk.
    _PER_IMAGE_ESTIMATE_KEYS = frozenset({
        "raw_image_bytes", "scrubbed_image_bytes", "z_image_latent_bytes",
        "vq_tokens_bytes", "control_map_bytes", "metadata_bytes",
    })
    _PER_PREFERENCE_PAIR_ESTIMATE_KEYS = frozenset({
        "preference_pair_source_bytes", "preference_pair_latent_bytes",
        "preference_pair_metadata_bytes",
    })

    # Windows classic MAX_PATH. (Windows 10 1607+ can lift this via a
    # long-paths opt-in registry key / app manifest, but that's a per-machine
    # setting this pipeline can't assume — stay conservative against the
    # unmodified default.)
    _WINDOWS_MAX_PATH = 260
    # Safety buffer subtracted from the raw limit — leaves room for a
    # filesystem/tooling quirk (e.g. a backup or sync tool re-nesting paths)
    # without living exactly on the edge of the actual OS limit.
    _WINDOWS_PATH_SAFETY_BUFFER = 20

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._data_root = config.data_root
        self._safety_margin = config.storage.safety_margin
        self._estimates = config.storage.per_record_estimates

    def _check_windows_max_path(self) -> None:
        """Validate DATA_ROOT against Windows' MAX_PATH, using the pipeline's
        own actual worst-case relative path — not an arbitrary root-length
        guess.

        BUG FIX: the previous check rejected any DATA_ROOT >= 50 characters,
        regardless of whether the *actual* deepest path this pipeline ever
        constructs would come anywhere near MAX_PATH=260. That threshold was
        roughly 3.5x more conservative than necessary — it would falsely
        reject a perfectly safe path like
        "D:\\Users\\SomeUser\\Projects\\ML\\krisna_data_root" (well over 50
        chars, nowhere close to 260 once you add this pipeline's actual
        deepest subpath) while providing no real guarantee for a case that
        WOULD overflow if this pipeline's directory layout got deeper later.
        This computes the real worst case instead: the longest configured
        subdirectory (from PathsConfig) plus a representative worst-case
        filename (a uuid4-based shard name with the longest extension this
        pipeline writes, ".safetensors").
        """
        from data_forge.config import PathsConfig

        longest_subdir = max(len(v) for v in vars(PathsConfig()).values())
        # Representative worst-case filename this pipeline actually writes,
        # e.g. "processed/dpo_latents/<source>/<32-char-uuid4-hex>.safetensors"
        worst_case_filename_len = len("shard_prefix_") + 32 + len(".safetensors")
        worst_case_relative = longest_subdir + 1 + worst_case_filename_len  # +1 for path sep

        worst_case_total = len(str(self._data_root)) + 1 + worst_case_relative
        limit = self._WINDOWS_MAX_PATH - self._WINDOWS_PATH_SAFETY_BUFFER

        if worst_case_total >= limit:
            max_safe_root_len = limit - 1 - worst_case_relative
            raise StorageQuotaExceeded(
                f"Windows MAX_PATH risk: DATA_ROOT '{self._data_root}' "
                f"({len(str(self._data_root))} chars) plus this pipeline's worst-case "
                f"subpath ({worst_case_relative} chars) projects to "
                f"{worst_case_total} total characters, at or above the "
                f"{limit}-char safe limit (MAX_PATH={self._WINDOWS_MAX_PATH}, minus a "
                f"{self._WINDOWS_PATH_SAFETY_BUFFER}-char safety buffer). "
                f"Use a DATA_ROOT no longer than ~{max(max_safe_root_len, 0)} characters, "
                "or enable Windows long-path support (Local Group Policy / registry "
                "LongPathsEnabled) if you control this machine."
            )

    def pre_flight_check(self, manifest_count: int, preference_pair_count: int = 0) -> dict[str, Any]:
        """Calculate projected storage needs and validate against available space.

        Raises StorageQuotaExceeded if projected > (available * (1 - safety_margin)).
        """
        self._check_windows_max_path()

        projected = self.calculate_projected_size(manifest_count, preference_pair_count)
        available = self.get_available_bytes()
        safe_limit = int(available * (1.0 - self._safety_margin))

        report = {
            "manifest_count": manifest_count,
            "preference_pair_count": preference_pair_count,
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
                f"Records: {manifest_count}, preference pairs: {preference_pair_count}"
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

    def calculate_projected_size(self, record_count: int, preference_pair_count: int = 0) -> int:
        """Estimate total bytes needed for record_count image records plus
        preference_pair_count DPO pairs, through all stages.

        Each count is multiplied against its own disjoint estimate-key set
        (see _PER_IMAGE_ESTIMATE_KEYS / _PER_PREFERENCE_PAIR_ESTIMATE_KEYS)
        rather than summing every configured estimate against one count.
        """
        per_image = sum(
            v for k, v in self._estimates.items() if k in self._PER_IMAGE_ESTIMATE_KEYS
        ) if self._estimates else 546_000  # fallback: raw+scrubbed+z_latent+vq+control+meta defaults
        per_pair = sum(
            v for k, v in self._estimates.items() if k in self._PER_PREFERENCE_PAIR_ESTIMATE_KEYS
        )
        return record_count * per_image + preference_pair_count * per_pair

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
