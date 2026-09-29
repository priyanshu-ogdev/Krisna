"""VAE config assertion — deterministic channel count verification (v13 §4.2).

This was only "manual" because nobody had written the two-line config check yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_forge.logging_setup import get_logger

log = get_logger("agents.vae_checker")


class VAEConfigError(Exception):
    """Raised when VAE config doesn't match expectations."""


def check_vae_config(
    model_dir: Path | str,
    expected_channels: int = 64,
    expected_downsample: int = 16,
) -> dict[str, Any]:
    """Assert VAE config matches expected values.

    Reads the channel count directly from the VAE's own config file,
    asserts against the storage-budget calculation, and fails loudly
    if the assumption drifts.

    Returns the parsed config on success.
    Raises VAEConfigError on mismatch.
    """
    model_dir = Path(model_dir)

    # Search for config files in common locations
    config_candidates = [
        model_dir / "config.json",
        model_dir / "vae" / "config.json",
        model_dir / "vae_config.json",
    ]

    config_data: dict[str, Any] | None = None
    config_path: Path | None = None

    for candidate in config_candidates:
        if candidate.exists():
            config_data = json.loads(candidate.read_text(encoding="utf-8"))
            config_path = candidate
            break

    if config_data is None:
        raise VAEConfigError(
            f"No VAE config found. Searched: {[str(c) for c in config_candidates]}"
        )

    log.info("vae_config_found", path=str(config_path))

    # Extract channel count — check common config key names
    channels = (
        config_data.get("latent_channels")
        or config_data.get("out_channels")
        or config_data.get("z_channels")
    )

    if channels is None:
        raise VAEConfigError(
            f"Could not find channel count in VAE config. "
            f"Available keys: {list(config_data.keys())}"
        )

    # Extract downsample ratio if available
    downsample = config_data.get("spatial_downsample_ratio")

    # Assertions
    if channels != expected_channels:
        raise VAEConfigError(
            f"VAE channel count mismatch: expected {expected_channels}, "
            f"got {channels}. Storage budget calculations may be wrong."
        )

    if downsample is not None and downsample != expected_downsample:
        raise VAEConfigError(
            f"VAE downsample ratio mismatch: expected {expected_downsample}, "
            f"got {downsample}."
        )

    log.info(
        "vae_config_verified",
        channels=channels,
        downsample=downsample,
        config_path=str(config_path),
    )

    return {
        "channels": channels,
        "downsample": downsample,
        "config_path": str(config_path),
        "verified": True,
    }
