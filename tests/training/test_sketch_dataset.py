from __future__ import annotations

import json

import pytest

from krisna_training.sketch.dataset import SketchTokenDataset


def test_reads_inline_tokens(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"tokens": list(range(4)), "caption": "a login screen"}) + "\n"
    )
    ds = SketchTokenDataset(manifest, grid_h=2, grid_w=2)
    assert len(ds) == 1
    assert ds[0] == {"tokens": [0, 1, 2, 3], "caption": "a login screen"}


def test_reads_multiple_records(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    lines = [
        json.dumps({"tokens": [1, 2, 3, 4], "caption": "a"}),
        json.dumps({"tokens": [5, 6, 7, 8], "caption": "b"}),
    ]
    manifest.write_text("\n".join(lines) + "\n")
    ds = SketchTokenDataset(manifest, grid_h=2, grid_w=2)
    assert len(ds) == 2
    assert ds[1]["caption"] == "b"


def test_missing_caption_defaults_to_empty_string(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"tokens": [1, 2, 3, 4]}) + "\n")
    ds = SketchTokenDataset(manifest, grid_h=2, grid_w=2)
    assert ds[0]["caption"] == ""


def test_wrong_token_count_raises(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"tokens": [1, 2, 3], "caption": "x"}) + "\n")  # only 3, expected 4
    ds = SketchTokenDataset(manifest, grid_h=2, grid_w=2)
    with pytest.raises(ValueError, match="expected 4"):
        ds[0]


def test_reads_from_tokens_path(tmp_path):
    np = pytest.importorskip("numpy")
    tokens_dir = tmp_path / "tokens"
    tokens_dir.mkdir()
    np.save(tokens_dir / "0000.npy", np.array([1, 2, 3, 4], dtype="int32"))

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"tokens_path": "tokens/0000.npy", "caption": "c"}) + "\n")
    ds = SketchTokenDataset(manifest, grid_h=2, grid_w=2)
    assert ds[0]["tokens"] == [1, 2, 3, 4]


class TestConsolidatedShardFormat:
    """UPGRADE (docs/review/33_model_review_2_data_training_connection.md):
    tokens_shard/tokens_index is the current, default format both
    sync_sketch_tier.py and prepare_dataset.py write — replacing one
    .npy file per image (a real I/O bottleneck at PRD §8.3's target
    corpus scale) with consolidated, memory-mapped shard files."""

    def test_reads_from_consolidated_shard(self, tmp_path):
        np = pytest.importorskip("numpy")
        tokens_dir = tmp_path / "tokens"
        tokens_dir.mkdir()
        shard = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype="int32")
        np.save(tokens_dir / "shard_00000.npy", shard)

        manifest = tmp_path / "manifest.jsonl"
        lines = [
            json.dumps({"tokens_shard": "tokens/shard_00000.npy", "tokens_index": 0, "caption": "a"}),
            json.dumps({"tokens_shard": "tokens/shard_00000.npy", "tokens_index": 2, "caption": "c"}),
        ]
        manifest.write_text("\n".join(lines) + "\n")
        ds = SketchTokenDataset(manifest, grid_h=2, grid_w=2)

        assert ds[0]["tokens"] == [1, 2, 3, 4]
        assert ds[0]["caption"] == "a"
        assert ds[1]["tokens"] == [9, 10, 11, 12]  # tokens_index=2, not 1 — order-independent lookup
        assert ds[1]["caption"] == "c"

    def test_shard_opened_once_not_per_getitem_call(self, tmp_path, monkeypatch):
        """The whole point of the upgrade: the shard file must be opened
        ONCE (per distinct shard referenced), not once per __getitem__
        call — that's the actual fix for the many-small-files I/O
        bottleneck this format exists to solve."""
        np = pytest.importorskip("numpy")
        tokens_dir = tmp_path / "tokens"
        tokens_dir.mkdir()
        shard = np.array([[1, 2, 3, 4]] * 5, dtype="int32")
        np.save(tokens_dir / "shard_00000.npy", shard)

        manifest = tmp_path / "manifest.jsonl"
        lines = [
            json.dumps({"tokens_shard": "tokens/shard_00000.npy", "tokens_index": i, "caption": str(i)})
            for i in range(5)
        ]
        manifest.write_text("\n".join(lines) + "\n")
        ds = SketchTokenDataset(manifest, grid_h=2, grid_w=2)

        load_calls = []
        real_load = np.load

        def counting_load(*args, **kwargs):
            load_calls.append(args)
            return real_load(*args, **kwargs)

        monkeypatch.setattr(np, "load", counting_load)

        for i in range(5):
            _ = ds[i]["tokens"]

        assert len(load_calls) == 1  # one np.load call total, not five

    def test_multiple_distinct_shards_each_opened_once(self, tmp_path):
        np = pytest.importorskip("numpy")
        tokens_dir = tmp_path / "tokens"
        tokens_dir.mkdir()
        np.save(tokens_dir / "shard_00000.npy", np.array([[1, 2, 3, 4]], dtype="int32"))
        np.save(tokens_dir / "shard_00001.npy", np.array([[5, 6, 7, 8]], dtype="int32"))

        manifest = tmp_path / "manifest.jsonl"
        lines = [
            json.dumps({"tokens_shard": "tokens/shard_00000.npy", "tokens_index": 0, "caption": "a"}),
            json.dumps({"tokens_shard": "tokens/shard_00001.npy", "tokens_index": 0, "caption": "b"}),
        ]
        manifest.write_text("\n".join(lines) + "\n")
        ds = SketchTokenDataset(manifest, grid_h=2, grid_w=2)

        assert ds[0]["tokens"] == [1, 2, 3, 4]
        assert ds[1]["tokens"] == [5, 6, 7, 8]
        assert len(ds._shard_arrays) == 2

    def test_wrong_token_count_still_detected_in_shard_mode(self, tmp_path):
        np = pytest.importorskip("numpy")
        tokens_dir = tmp_path / "tokens"
        tokens_dir.mkdir()
        np.save(tokens_dir / "shard_00000.npy", np.array([[1, 2, 3]], dtype="int32"))  # 3, not 4

        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            json.dumps({"tokens_shard": "tokens/shard_00000.npy", "tokens_index": 0, "caption": "x"}) + "\n"
        )
        ds = SketchTokenDataset(manifest, grid_h=2, grid_w=2)
        with pytest.raises(ValueError, match="expected 4"):
            ds[0]
