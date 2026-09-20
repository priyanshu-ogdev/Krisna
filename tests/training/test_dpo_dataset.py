from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
from PIL import Image

from krisna_inference.backends.common import BlobStore
from krisna_training.polish.dpo_dataset import PreferencePairDataset
from krisna_training.preference import PreferencePair, PreferenceStore


def test_dpo_dataset_loads_and_transforms(tmp_path):
    db_path = tmp_path / "pref.db"
    store = PreferenceStore(db_path=db_path)

    blob_dir = tmp_path / "blobs"
    blobs = BlobStore(root=blob_dir)

    im1 = Image.new("RGB", (128, 128), color="green")
    im2 = Image.new("RGB", (128, 128), color="yellow")
    ref_chosen = blobs.save_image(im1)
    ref_rejected = blobs.save_image(im2)

    store.add(
        PreferencePair(
            prompt="test prompt",
            chosen_ref=ref_chosen,
            rejected_ref=ref_rejected,
            source="pickapic_v2",
        )
    )

    dataset = PreferencePairDataset(
        db_path=str(db_path),
        blob_root=str(blob_dir),
        sources=["pickapic_v2"],
        resolution=256,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert "chosen_pixel_values" in sample
    assert "rejected_pixel_values" in sample
    assert sample["prompt"] == "test prompt"
    assert sample["chosen_pixel_values"].shape == (3, 256, 256)
    assert sample["rejected_pixel_values"].shape == (3, 256, 256)


def test_dpo_dataset_empty_raises(tmp_path):
    db_path = tmp_path / "empty.db"
    store = PreferenceStore(db_path=db_path)
    with pytest.raises(ValueError, match="No preference pairs found"):
        PreferencePairDataset(db_path=str(db_path), blob_root=str(tmp_path))
