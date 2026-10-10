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


def test_dpo_dataset_strict_disjoint_train_val(tmp_path):
    db_path = tmp_path / "pref_split.db"
    store = PreferenceStore(db_path=db_path)
    blob_dir = tmp_path / "blobs_split"
    blobs = BlobStore(root=blob_dir)

    for i in range(20):
        im1 = Image.new("RGB", (64, 64), color="blue")
        im2 = Image.new("RGB", (64, 64), color="red")
        ref_c = blobs.save_image(im1)
        ref_r = blobs.save_image(im2)
        store.add(
            PreferencePair(
                prompt=f"prompt variant {i}",
                chosen_ref=ref_c,
                rejected_ref=ref_r,
                source="pickapic_v2",
            )
        )

    train_ds = PreferencePairDataset(
        db_path=str(db_path),
        blob_root=str(blob_dir),
        sources=["pickapic_v2"],
        split="train",
        val_fraction=0.2,
    )
    val_ds = PreferencePairDataset(
        db_path=str(db_path),
        blob_root=str(blob_dir),
        sources=["pickapic_v2"],
        split="val",
        val_fraction=0.2,
    )

    train_ids = {p.id for p in train_ds.pairs}
    val_ids = {p.id for p in val_ds.pairs}

    assert len(train_ids) > 0
    assert len(val_ids) > 0
    assert train_ids.isdisjoint(val_ids), "Train and Val splits leaked duplicate pairs!"
    assert train_ids | val_ids == {p.id for p in store.list()}


def test_dpo_dataset_no_prompt_or_session_leakage(tmp_path):
    db_path = tmp_path / "pref_groups.db"
    store = PreferenceStore(db_path=db_path)
    blob_dir = tmp_path / "blobs_groups"
    blobs = BlobStore(root=blob_dir)

    # 10 pairs sharing prompt A, 10 pairs sharing prompt B
    for i in range(10):
        im1 = Image.new("RGB", (64, 64), color="green")
        im2 = Image.new("RGB", (64, 64), color="yellow")
        store.add(
            PreferencePair(
                prompt="Prompt A: dashboard dark mode",
                chosen_ref=blobs.save_image(im1),
                rejected_ref=blobs.save_image(im2),
                source="pickapic_v2",
            )
        )
        store.add(
            PreferencePair(
                prompt="Prompt B: e-commerce product card",
                chosen_ref=blobs.save_image(im1),
                rejected_ref=blobs.save_image(im2),
                source="pickapic_v2",
            )
        )

    # Also add pairs sharing a session_id
    for i in range(5):
        im1 = Image.new("RGB", (64, 64), color="green")
        im2 = Image.new("RGB", (64, 64), color="yellow")
        store.add(
            PreferencePair(
                prompt=f"Session prompt step {i}",
                session_id="session-user-12345",
                chosen_ref=blobs.save_image(im1),
                rejected_ref=blobs.save_image(im2),
                source="pickapic_v2",
            )
        )

    train_ds = PreferencePairDataset(
        db_path=str(db_path),
        blob_root=str(blob_dir),
        split="train",
        val_fraction=0.5,
    )
    val_ds = PreferencePairDataset(
        db_path=str(db_path),
        blob_root=str(blob_dir),
        split="val",
        val_fraction=0.5,
    )

    train_prompts = {p.prompt.strip().lower() for p in train_ds.pairs}
    val_prompts = {p.prompt.strip().lower() for p in val_ds.pairs}
    assert train_prompts.isdisjoint(val_prompts), "Prompt leakage detected between train and val!"

    train_sessions = {p.session_id for p in train_ds.pairs if p.session_id}
    val_sessions = {p.session_id for p in val_ds.pairs if p.session_id}
    assert train_sessions.isdisjoint(val_sessions), "Session leakage detected between train and val!"


def test_dpo_dataset_single_pair_zero_overlap(tmp_path):
    db_path = tmp_path / "pref_single.db"
    store = PreferenceStore(db_path=db_path)
    blob_dir = tmp_path / "blobs_single"
    blobs = BlobStore(root=blob_dir)

    im1 = Image.new("RGB", (64, 64), color="green")
    im2 = Image.new("RGB", (64, 64), color="yellow")
    store.add(
        PreferencePair(
            prompt="single prompt",
            chosen_ref=blobs.save_image(im1),
            rejected_ref=blobs.save_image(im2),
            source="pickapic_v2",
        )
    )

    train_ds = PreferencePairDataset(
        db_path=str(db_path),
        blob_root=str(blob_dir),
        split="train",
        val_fraction=0.1,
    )
    assert len(train_ds) == 1

    # Val split must raise ValueError rather than copying the single pair from train
    with pytest.raises(ValueError, match="No preference pairs found"):
        PreferencePairDataset(
            db_path=str(db_path),
            blob_root=str(blob_dir),
            split="val",
            val_fraction=0.1,
        )
