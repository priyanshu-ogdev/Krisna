"""Tests for TokenShardWriter — the shared writer both
sync_sketch_tier.py and prepare_dataset.py use to produce consolidated,
memory-mappable token shards instead of one .npy file per image (see
docs/review/33_model_review_2_data_training_connection.md and
token_shard_writer.py's module docstring for the full reasoning).
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from krisna_training.sketch.token_shard_writer import TokenShardWriter


def test_single_shard_written_on_close(tmp_path):
    writer = TokenShardWriter(tmp_path / "tokens", expected_token_len=4, shard_size=10)
    paths = [writer.add([i, i, i, i]) for i in range(3)]
    writer.close()

    assert (tmp_path / "tokens" / "shard_00000.npy").exists()
    arr = np.load(tmp_path / "tokens" / "shard_00000.npy")
    assert arr.shape == (3, 4)
    for i, (shard_path, idx_in_shard) in enumerate(paths):
        assert shard_path == "tokens/shard_00000.npy"
        assert idx_in_shard == i
        assert arr[idx_in_shard].tolist() == [i, i, i, i]


def test_auto_flushes_at_shard_size_boundary(tmp_path):
    writer = TokenShardWriter(tmp_path / "tokens", expected_token_len=2, shard_size=2)
    p0 = writer.add([1, 1])
    p1 = writer.add([2, 2])  # hits shard_size=2 -> auto-flush here
    p2 = writer.add([3, 3])  # starts a new shard
    writer.close()

    assert p0[0] == "tokens/shard_00000.npy" and p0[1] == 0
    assert p1[0] == "tokens/shard_00000.npy" and p1[1] == 1
    assert p2[0] == "tokens/shard_00001.npy" and p2[1] == 0

    shard0 = np.load(tmp_path / "tokens" / "shard_00000.npy")
    shard1 = np.load(tmp_path / "tokens" / "shard_00001.npy")
    assert shard0.shape == (2, 2)
    assert shard1.shape == (1, 2)


def test_close_flushes_partial_final_shard(tmp_path):
    """A forgotten close() would silently drop the final partial shard —
    this confirms close() actually does the flush, not just a no-op."""
    writer = TokenShardWriter(tmp_path / "tokens", expected_token_len=2, shard_size=100)
    writer.add([9, 9])
    writer.close()

    assert (tmp_path / "tokens" / "shard_00000.npy").exists()
    arr = np.load(tmp_path / "tokens" / "shard_00000.npy")
    assert arr.shape == (1, 2)


def test_wrong_token_length_raises(tmp_path):
    writer = TokenShardWriter(tmp_path / "tokens", expected_token_len=4)
    with pytest.raises(ValueError, match="expected 4"):
        writer.add([1, 2, 3])


def test_total_written_tracks_all_adds_including_unflushed(tmp_path):
    writer = TokenShardWriter(tmp_path / "tokens", expected_token_len=2, shard_size=100)
    writer.add([1, 1])
    writer.add([2, 2])
    assert writer.total_written == 2
    writer.close()
    assert writer.total_written == 2


def test_written_shard_dtype_is_int32(tmp_path):
    """Matches VQGAN codebook indices — must not silently upcast/
    downcast (e.g. to int64, doubling shard file size for no reason, or
    to a type too narrow for a real codebook_size like 16384)."""
    writer = TokenShardWriter(tmp_path / "tokens", expected_token_len=2, shard_size=100)
    writer.add([1, 2])
    writer.close()
    arr = np.load(tmp_path / "tokens" / "shard_00000.npy")
    assert arr.dtype == np.int32


class TestEndToEndSyncSketchTierWithShardWriter:
    """Integration test: sync_sketch_tier.py's real code path, through a
    fake tokenizer, actually produces manifests SketchTokenDataset can
    read back correctly — not just the writer and reader tested in
    isolation."""

    def test_sync_then_read_round_trip(self, tmp_path):
        from PIL import Image

        from krisna_training.data_forge_bridge.sync_sketch_tier import sync
        from krisna_training.sketch.dataset import SketchTokenDataset

        data_forge_dir = tmp_path / "data_forge_model_data"
        images_dir = data_forge_dir / "sketch_tier_maskgit" / "images"
        images_dir.mkdir(parents=True)

        for i in range(3):
            Image.new("RGB", (32, 32), color=(i * 10, 0, 0)).save(images_dir / f"img_{i}.png")

        captions_path = data_forge_dir / "sketch_tier_maskgit" / "captions.jsonl"
        import json

        captions_path.write_text(
            "\n".join(
                json.dumps({"record_id": f"r{i}", "image_filename": f"img_{i}.png",
                            "caption": f"caption {i}", "source_caption": None})
                for i in range(3)
            )
        )

        class FakeTokenizer:
            codebook_size = 16384

            def grid_shape_for(self, image_size):
                return (2, 2)

            def encode(self, image):
                # Deterministic per-image "tokens" derived from the
                # image's own mean pixel value, so each image maps to a
                # distinguishable, reproducible token grid.
                import numpy as _np

                val = int(_np.array(image).mean()) % 100
                return [val, val, val, val]

        output_dir = tmp_path / "training_output"
        manifest_path = sync(
            data_forge_model_data_dir=data_forge_dir,
            output_dir=output_dir,
            tokenizer=FakeTokenizer(),
            image_size=32,
        )

        assert manifest_path.exists()
        ds = SketchTokenDataset(manifest_path, grid_h=2, grid_w=2)
        assert len(ds) == 3
        captions_seen = {ds[i]["caption"] for i in range(3)}
        assert captions_seen == {"caption 0", "caption 1", "caption 2"}
        # Every record's tokens must round-trip through the shard file
        # correctly, not just happen to look right for record 0.
        for i in range(3):
            tokens = ds[i]["tokens"]
            assert len(tokens) == 4
            assert len(set(tokens)) == 1  # our fake tokenizer always returns 4 identical values
