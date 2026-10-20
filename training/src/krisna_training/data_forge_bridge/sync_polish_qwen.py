"""Bridges data-forge's `model_data/polish_qwen_image21/` export into the
exact input format expected by Qwen-Image-2.1 training (`training/polish/prepare_dataset.py`).

Validates the strict source-disjoint split assignments, non-identical
condition/target pairs, and image byte hashes before materializing the
HuggingFace `Dataset` layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

from krisna_training.polish.prepare_dataset import prepare_qwen_pairs_from_manifest

log = logging.getLogger("krisna_training.data_forge_bridge.sync_polish_qwen")


def sync(
    data_forge_model_data_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Synchronize Data-Forge's Qwen pair export to training dataset format.

    Args:
        data_forge_model_data_dir: Path to data-forge's `model_data/` root
            or `model_data/polish_qwen_image21/`.
        output_dir: Destination path to materialize the prepared dataset.

    Returns:
        Path to the prepared dataset directory.
    """
    root = Path(data_forge_model_data_dir)
    manifest_path = root / "manifest.jsonl" if (root / "manifest.jsonl").is_file() else root / "polish_qwen_image21" / "manifest.jsonl"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Qwen-Image-2.1 pair manifest not found at {manifest_path}. "
            "Ensure data-forge's stage s12_model_data_export has executed with polish_qwen_image21 enabled."
        )

    log.info("syncing_qwen_pairs manifest=%s output=%s", manifest_path, output_dir)
    return prepare_qwen_pairs_from_manifest(manifest_path, output_dir)
