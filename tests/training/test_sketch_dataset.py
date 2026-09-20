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
