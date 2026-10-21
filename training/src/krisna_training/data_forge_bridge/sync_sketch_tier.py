"""Sketch tier synchronization bridge — RETIRED / DETERMINISTIC ENGINE.

The Sketch tier standardizes on the deterministic neuro-symbolic `krisna.layout.v1`
engine (0 MB VRAM, <5ms CPU latency). Discrete VQ token codebooks and MaskGIT training
shards are completely retired and obsolete.

This module is preserved as a sealed stub to prevent runtime import crashes while
authoritatively informing callers that no token shard synthesis is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
import warnings

log = logging.getLogger("krisna_training.data_forge_bridge.sync_sketch_tier")


def sync(
    data_forge_model_data_dir: str | Path,
    output_dir: str | Path,
    tokenizer: object | None = None,
    image_size: int = 256,
    expected_codebook_size: int | None = 16384,
) -> Path | None:
    """No-op sync for the deterministic Sketch tier.

    Because the production Sketch tier (krisna.layout.v1) generates layouts
    via deterministic Simplex constraint solving rather than learned discrete tokens,
    data-forge VQ token manifests are not consumed during training.
    """
    warnings.warn(
        "sync_sketch_tier is retired: Sketch tier is 100% deterministic Layout IR (0 MB VRAM). "
        "No training tokens are generated or consumed.",
        DeprecationWarning,
        stacklevel=2,
    )
    log.info("Sketch tier is deterministic (krisna.layout.v1); bypassing token shard sync.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    if not manifest.exists():
        manifest.write_text("", encoding="utf-8")
    return manifest
