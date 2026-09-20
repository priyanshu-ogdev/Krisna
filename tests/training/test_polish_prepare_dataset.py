from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

from krisna_training.polish.prepare_dataset import count_prepared, load_captions, prepare
from krisna_training.polish.dataset_prep import prepare as legacy_prepare


def test_prepare_with_dict(tmp_path):
    img_dir = tmp_path / "src"
    img_dir.mkdir()
    im = Image.new("RGB", (64, 64), color="blue")
    im.save(img_dir / "card.png")

    out_dir = tmp_path / "out"
    prepare(img_dir, out_dir, captions={"card.png": "a blue UI card"})

    meta = out_dir / "metadata.jsonl"
    assert meta.exists()
    assert count_prepared(out_dir) == 1
    rec = json.loads(meta.read_text(encoding="utf-8").strip())
    assert rec["file_name"] == "card.png"
    assert rec["text"] == "a blue UI card"


def test_prepare_with_jsonl(tmp_path):
    img_dir = tmp_path / "src"
    img_dir.mkdir()
    im = Image.new("RGB", (64, 64), color="red")
    im.save(img_dir / "nav.png")

    jsonl_path = tmp_path / "captions.jsonl"
    jsonl_path.write_text(json.dumps({"image_filename": "nav.png", "caption": "navigation bar"}) + "\n")

    out_dir = tmp_path / "out"
    prepare(img_dir, out_dir, captions=jsonl_path)

    meta = out_dir / "metadata.jsonl"
    assert meta.exists()
    assert count_prepared(out_dir) == 1
    rec = json.loads(meta.read_text(encoding="utf-8").strip())
    assert rec["file_name"] == "nav.png"
    assert rec["text"] == "navigation bar"


def test_legacy_wrapper_works(tmp_path):
    img_dir = tmp_path / "src"
    img_dir.mkdir()
    im = Image.new("RGB", (64, 64), color="green")
    im.save(img_dir / "icon.png")

    out_dir = tmp_path / "out"
    legacy_prepare(img_dir, out_dir, captions={"icon.png": "an icon"})
    assert count_prepared(out_dir) == 1
