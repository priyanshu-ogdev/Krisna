from __future__ import annotations

import json

import pytest
from PIL import Image

from krisna_training.dpo.preference_store import PreferenceStore
from krisna_inference.backends.common import BlobStore
from krisna_training.data_forge_bridge.sync_dpo_pairs import sync


@pytest.fixture
def pref_store(tmp_path) -> PreferenceStore:
    s = PreferenceStore(db_path=tmp_path / "prefs.db")
    yield s
    s.close()


@pytest.fixture
def blob_store(tmp_path) -> BlobStore:
    return BlobStore(root=tmp_path / "blobs")


def _write_pair(source_dir, pair_id, preferred="a", dedup_status="unique", prompt="a poster"):
    source_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color="red").save(source_dir / f"{pair_id}_a.png")
    Image.new("RGB", (4, 4), color="blue").save(source_dir / f"{pair_id}_b.png")
    meta = {
        "pair_id": pair_id, "image_a": f"{pair_id}_a.png", "image_b": f"{pair_id}_b.png",
        "preferred": preferred, "dedup_status": dedup_status, "prompt": prompt,
    }
    (source_dir / f"{pair_id}.json").write_text(json.dumps(meta))


def test_sync_imports_pairs_with_correct_chosen_rejected(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    _write_pair(data_root / "preference_pairs" / "pickapic_v2", "p1", preferred="a")

    counts = sync(data_root, pref_store, blob_store, sources=["pickapic_v2"])
    assert counts["pickapic_v2"] == 1

    pairs = pref_store.list(source="pickapic_v2")
    assert len(pairs) == 1
    assert pairs[0].chosen_ref.startswith("blob://pickapic_v2_a_")
    assert pairs[0].rejected_ref.startswith("blob://pickapic_v2_b_")
    assert pairs[0].prompt == "a poster"


def test_sync_preferred_b_swaps_chosen_rejected(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    _write_pair(data_root / "preference_pairs" / "hpdv2", "p1", preferred="b")

    sync(data_root, pref_store, blob_store, sources=["hpdv2"])
    pairs = pref_store.list(source="hpdv2")
    assert pairs[0].chosen_ref.startswith("blob://hpdv2_b_")
    assert pairs[0].rejected_ref.startswith("blob://hpdv2_a_")


def test_sync_skips_non_deduped_pairs(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    _write_pair(data_root / "preference_pairs" / "designsense_10k", "p1", dedup_status="pending")

    counts = sync(data_root, pref_store, blob_store, sources=["designsense_10k"])
    assert counts["designsense_10k"] == 0
    assert counts["skipped_not_deduped"] == 1
    assert pref_store.count() == 0


def test_sync_skips_pairs_missing_preferred_label(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    source_dir = data_root / "preference_pairs" / "designpref"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(source_dir / "p1_a.png")
    Image.new("RGB", (4, 4)).save(source_dir / "p1_b.png")
    (source_dir / "p1.json").write_text(json.dumps({
        "pair_id": "p1", "image_a": "p1_a.png", "image_b": "p1_b.png",
        "dedup_status": "unique",
        # no "preferred" key
    }))

    counts = sync(data_root, pref_store, blob_store, sources=["designpref"])
    assert counts["designpref"] == 0
    assert pref_store.count() == 0


def test_sync_skips_missing_image_files(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    source_dir = data_root / "preference_pairs" / "pickapic_v2"
    source_dir.mkdir(parents=True)
    (source_dir / "p1.json").write_text(json.dumps({
        "pair_id": "p1", "image_a": "missing_a.png", "image_b": "missing_b.png",
        "dedup_status": "unique", "preferred": "a",
    }))

    counts = sync(data_root, pref_store, blob_store, sources=["pickapic_v2"])
    assert counts["skipped_missing_images"] == 1
    assert pref_store.count() == 0


def test_sync_missing_source_dir_returns_zero_not_error(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    (data_root / "preference_pairs").mkdir(parents=True)
    counts = sync(data_root, pref_store, blob_store, sources=["hpdv2"])
    assert counts["hpdv2"] == 0


def test_sync_missing_preference_pairs_root_raises(tmp_path, pref_store, blob_store):
    with pytest.raises(FileNotFoundError):
        sync(tmp_path / "nonexistent", pref_store, blob_store)


def test_sync_unknown_source_key_raises(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    (data_root / "preference_pairs").mkdir(parents=True)
    with pytest.raises(KeyError):
        sync(data_root, pref_store, blob_store, sources=["totally_unknown_source"])


def test_sync_multiple_pairs_same_source(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    source_dir = data_root / "preference_pairs" / "hpdv2"
    _write_pair(source_dir, "p1")
    _write_pair(source_dir, "p2")

    counts = sync(data_root, pref_store, blob_store, sources=["hpdv2"])
    assert counts["hpdv2"] == 2
    assert pref_store.count(source="hpdv2") == 2


def test_sync_default_sources_covers_all_four(tmp_path, pref_store, blob_store):
    data_root = tmp_path / "data_root"
    (data_root / "preference_pairs").mkdir(parents=True)
    counts = sync(data_root, pref_store, blob_store)
    for key in ("pickapic_v2", "hpdv2", "designsense_10k", "designpref"):
        assert key in counts
