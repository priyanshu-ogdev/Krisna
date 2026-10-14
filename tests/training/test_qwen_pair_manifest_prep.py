from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from krisna_training.polish.prepare_dataset import prepare_qwen_pairs_from_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_qwen_pairs_preserves_receipts_and_splits(tmp_path: Path) -> None:
    root = tmp_path / "pairs"
    conditions = root / "conditions"
    targets = root / "targets"
    conditions.mkdir(parents=True)
    targets.mkdir(parents=True)
    condition = conditions / "pair.png"
    target = targets / "pair.png"
    Image.new("RGB", (16, 16), (10, 10, 10)).save(condition)
    Image.new("RGB", (16, 16), (240, 240, 240)).save(target)
    row = {
        "record_id": "pair-1",
        "condition_image": "conditions/pair.png",
        "target_image": "targets/pair.png",
        "caption": "Refine the UI",
        "split": "train",
        "source_dataset": "test-ui",
        "source_revision": "a" * 40,
        "source_provenance": {"url": "https://example.test/ui"},
        "condition_sha256": _sha(condition),
        "target_sha256": _sha(target),
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output = prepare_qwen_pairs_from_manifest(manifest, tmp_path / "prepared")
    metadata = json.loads((output / "metadata.jsonl").read_text().strip())
    receipt = json.loads((output / "dataset_receipt.json").read_text())
    assert metadata["pair_id"] == "pair-1"
    assert metadata["split"] == "train"
    assert metadata["source_provenance"]["url"].startswith("https://")
    assert receipt["split_counts"] == {"train": 1, "validation": 0, "test": 0}


def test_prepare_qwen_pairs_rejects_unassigned_split(tmp_path: Path) -> None:
    root = tmp_path / "pairs"
    root.mkdir()
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps({"record_id": "p", "split": "unassigned"}) + "\n")
    with pytest.raises(ValueError, match="split assignment"):
        prepare_qwen_pairs_from_manifest(manifest, tmp_path / "prepared")
