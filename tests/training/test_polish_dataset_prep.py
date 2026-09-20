from __future__ import annotations

import json

import pytest
from PIL import Image

from krisna_training.polish.dataset_prep import count_prepared, prepare


def _make_images(image_dir, names):
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.new("RGB", (4, 4), color="red").save(image_dir / name)


def test_prepare_copies_images_and_writes_metadata(tmp_path):
    src = tmp_path / "src"
    _make_images(src, ["a.png", "b.jpg"])

    out = prepare(src, tmp_path / "out", captions={"a.png": "a login screen", "b.jpg": "a signup form"})

    assert (out / "a.png").exists()
    assert (out / "b.jpg").exists()
    lines = [json.loads(l) for l in (out / "metadata.jsonl").read_text().strip().split("\n")]
    by_name = {r["file_name"]: r["text"] for r in lines}
    assert by_name == {"a.png": "a login screen", "b.jpg": "a signup form"}


def test_prepare_raises_on_empty_dir(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    with pytest.raises(ValueError, match="No images found"):
        prepare(src, tmp_path / "out")


def test_prepare_falls_back_to_shared_instance_prompt(tmp_path):
    src = tmp_path / "src"
    _make_images(src, ["a.png", "b.png"])

    out = prepare(src, tmp_path / "out", captions={"a.png": "custom caption"}, use_shared_instance_prompt="a UI screen")

    lines = [json.loads(l) for l in (out / "metadata.jsonl").read_text().strip().split("\n")]
    by_name = {r["file_name"]: r["text"] for r in lines}
    assert by_name["a.png"] == "custom caption"
    assert by_name["b.png"] == "a UI screen"


def test_prepare_empty_caption_when_no_fallback(tmp_path):
    src = tmp_path / "src"
    _make_images(src, ["a.png"])
    out = prepare(src, tmp_path / "out")
    lines = [json.loads(l) for l in (out / "metadata.jsonl").read_text().strip().split("\n")]
    assert lines[0]["text"] == ""


def test_prepare_ignores_non_image_files(tmp_path):
    src = tmp_path / "src"
    _make_images(src, ["a.png"])
    (src / "readme.txt").write_text("not an image")

    out = prepare(src, tmp_path / "out")
    assert count_prepared(out) == 1


def test_count_prepared_zero_when_no_metadata(tmp_path):
    assert count_prepared(tmp_path / "does_not_exist") == 0


def test_prepare_recurses_into_subdirectories(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(src / "sub" / "nested.png")

    out = prepare(src, tmp_path / "out")
    assert count_prepared(out) == 1
    assert (out / "nested.png").exists()
