from __future__ import annotations

import json
from pathlib import Path
import pytest
from PIL import Image

from krisna_training.sketch.prepare_dataset import prepare


class DummyTokenizer:
    def grid_shape_for(self, image_size: int) -> tuple[int, int]:
        grid_dim = image_size // 16
        return (grid_dim, grid_dim)

    def encode(self, image: Image.Image) -> list[int]:
        w, h = image.size
        grid_dim = w // 16
        return [1] * (grid_dim * grid_dim)


def test_prepare_dataset_with_json_dict(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    im = Image.new("RGB", (64, 64), color="blue")
    im.save(img_dir / "test1.png")

    captions_file = tmp_path / "captions.json"
    captions_file.write_text(json.dumps({"test1.png": "a blue card"}))

    out_dir = tmp_path / "out"
    tok = DummyTokenizer()
    manifest = prepare(img_dir, out_dir, tok, image_size=256, captions_path=captions_file)

    assert manifest.exists()
    lines = manifest.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["caption"] == "a blue card"
    assert "tokens_path" in rec


def test_prepare_dataset_with_jsonl(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    im = Image.new("RGB", (64, 64), color="red")
    im.save(img_dir / "ui_card.png")

    captions_file = tmp_path / "captions.jsonl"
    captions_file.write_text(json.dumps({"image_filename": "ui_card.png", "caption": "a red card"}) + "\n")

    out_dir = tmp_path / "out"
    tok = DummyTokenizer()
    manifest = prepare(img_dir, out_dir, tok, image_size=256, captions_path=captions_file)

    assert manifest.exists()
    lines = manifest.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["caption"] == "a red card"
