from __future__ import annotations

import json

import pytest
from PIL import Image

from krisna_training.data_forge_bridge.sync_sketch_tier import sync


class FakeVQTokenizer:
    """Deterministic stand-in for training.sketch.vq_tokenizer.VQTokenizer
    — same testing strategy used throughout this project for anything
    that would otherwise need a downloaded checkpoint. Implements just the
    two methods sync_sketch_tier.py actually calls."""

    def grid_shape_for(self, image_size: int):
        return (2, 2)

    def encode(self, image):
        return [1, 2, 3, 4]


def _make_data_forge_export(root, records):
    """records: list of (record_id, image_filename, caption).

    BUG FIX: this fixture used to name each image file after its
    record_id (f"{record_id}.png") and omit image_filename from
    captions.jsonl entirely — i.e. it baked the exact false assumption
    (filename stem == record_id) that sync_sketch_tier.py's old code
    made, so every test in this file passed while testing an unrealistic
    mock rather than real data-forge output. A real data-forge export
    names linked images after the ORIGINAL scrubbed-image filename
    (e.g. "rico_core_0000042.png"), completely unrelated to the uuid4
    record_id, and captions.jsonl entries include an explicit
    image_filename field precisely because that mismatch exists. This
    fixture now reflects that reality by construction — record_id and
    image_filename are deliberately different strings for every record.
    """
    sketch_dir = root / "sketch_tier_maskgit"
    images_dir = sketch_dir / "images"
    images_dir.mkdir(parents=True)
    captions = []
    for record_id, image_filename, caption in records:
        Image.new("RGB", (8, 8), color="purple").save(images_dir / image_filename)
        captions.append({"record_id": record_id, "caption": caption, "image_filename": image_filename})
    (sketch_dir / "captions.jsonl").write_text("\n".join(json.dumps(c) for c in captions))
    return root


def test_sync_writes_manifest_with_correct_captions(tmp_path):
    model_data = _make_data_forge_export(
        tmp_path / "model_data",
        [("a1b2c3d4-uuid-not-a-filename", "rico_core_0000042.png", "a signup form")],
    )
    manifest_path = sync(model_data, tmp_path / "prepared", tokenizer=FakeVQTokenizer())

    lines = [json.loads(l) for l in manifest_path.read_text().strip().split("\n")]
    assert len(lines) == 1
    assert lines[0]["caption"] == "a signup form"
    assert "tokens_path" in lines[0]


def test_sync_writes_real_token_files_from_tokenizer(tmp_path):
    model_data = _make_data_forge_export(
        tmp_path / "model_data", [("uuid-rec1", "webui_0000001.png", "x")]
    )
    manifest_path = sync(model_data, tmp_path / "prepared", tokenizer=FakeVQTokenizer())

    lines = [json.loads(l) for l in manifest_path.read_text().strip().split("\n")]
    import numpy as np

    tokens = np.load(manifest_path.parent / lines[0]["tokens_path"])
    assert tokens.tolist() == [1, 2, 3, 4]


def test_sync_missing_images_dir_raises_clear_error(tmp_path):
    model_data = tmp_path / "model_data"
    (model_data / "sketch_tier_maskgit").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="s12_model_data_export"):
        sync(model_data, tmp_path / "prepared", tokenizer=FakeVQTokenizer())


def test_sync_empty_images_dir_raises(tmp_path):
    model_data = tmp_path / "model_data"
    (model_data / "sketch_tier_maskgit" / "images").mkdir(parents=True)
    with pytest.raises(ValueError, match="No images found"):
        sync(model_data, tmp_path / "prepared", tokenizer=FakeVQTokenizer())


def test_sync_missing_caption_falls_back_to_empty_string(tmp_path):
    sketch_dir = tmp_path / "model_data" / "sketch_tier_maskgit"
    images_dir = sketch_dir / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(images_dir / "rico_core_0000001.png")
    # No captions.jsonl.

    manifest_path = sync(tmp_path / "model_data", tmp_path / "prepared", tokenizer=FakeVQTokenizer())
    lines = [json.loads(l) for l in manifest_path.read_text().strip().split("\n")]
    assert lines[0]["caption"] == ""


def test_sync_multiple_images(tmp_path):
    model_data = _make_data_forge_export(
        tmp_path / "model_data",
        [
            ("uuid-1", "rico_core_0000001.png", "a"),
            ("uuid-2", "rico_core_0000002.png", "b"),
            ("uuid-3", "clay_0000003.png", "c"),
        ],
    )
    manifest_path = sync(model_data, tmp_path / "prepared", tokenizer=FakeVQTokenizer())
    lines = manifest_path.read_text().strip().split("\n")
    assert len(lines) == 3


class TestRecordIdVsFilenameBugFix:
    """The actual regression test for the bug: record_id must never be
    derivable from the linked filename, and captions must still match
    correctly via the explicit image_filename join key.
    """

    def test_captions_match_via_image_filename_not_record_id_guess(self, tmp_path):
        # record_id is a real-looking uuid4; image_filename is a
        # realistic data-forge fetch-time name. Nothing about the
        # filename can be derived from the record_id or vice versa.
        model_data = _make_data_forge_export(
            tmp_path / "model_data",
            [("f47ac10b-58cc-4372-a567-0e02b2c3d479", "webui_0004821.png", "a pricing table")],
        )
        manifest_path = sync(model_data, tmp_path / "prepared", tokenizer=FakeVQTokenizer())
        lines = [json.loads(l) for l in manifest_path.read_text().strip().split("\n")]
        assert lines[0]["caption"] == "a pricing table"

    def test_legacy_captions_format_without_image_filename_warns_and_degrades(self, tmp_path, caplog):
        """A pre-fix data-forge export (captions.jsonl with no
        image_filename field at all) must not crash — it degrades to the
        old (broken) stem-matching behavior, but logs a loud warning
        explaining why captions are probably empty, instead of failing
        silently the way the original bug did.
        """
        sketch_dir = tmp_path / "model_data" / "sketch_tier_maskgit"
        images_dir = sketch_dir / "images"
        images_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(images_dir / "some_real_filename.png")
        # Legacy shape: no image_filename key at all.
        (sketch_dir / "captions.jsonl").write_text(
            json.dumps({"record_id": "uuid-abc", "caption": "will not match"})
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="krisna_training.data_forge_bridge.sync_sketch_tier"):
            manifest_path = sync(tmp_path / "model_data", tmp_path / "prepared", tokenizer=FakeVQTokenizer())

        assert any("legacy" in r.message.lower() for r in caplog.records)
        lines = [json.loads(l) for l in manifest_path.read_text().strip().split("\n")]
        # Degrades to the old broken behavior (empty caption) — documented,
        # not silent, and not a crash.
        assert lines[0]["caption"] == ""

    def test_sync_codebook_size_mismatch_raises_value_error(self, tmp_path):
        """C2: Mismatched codebook size must raise ValueError to prevent silent corruption."""
        model_data = _make_data_forge_export(
            tmp_path / "model_data",
            [("rec1", "test.png", "a button")],
        )
        bad_tok = FakeVQTokenizer()
        bad_tok.codebook_size = 8192  # Expected is 16384

        with pytest.raises(ValueError, match="does not match expected MaskGIT vocab_size"):
            sync(model_data, tmp_path / "prepared", tokenizer=bad_tok, expected_codebook_size=16384)

    def test_sync_codebook_size_match_succeeds(self, tmp_path):
        """C2: Matching codebook size succeeds without error."""
        model_data = _make_data_forge_export(
            tmp_path / "model_data",
            [("rec1", "test.png", "a button")],
        )
        good_tok = FakeVQTokenizer()
        good_tok.codebook_size = 16384

        manifest_path = sync(model_data, tmp_path / "prepared", tokenizer=good_tok, expected_codebook_size=16384)
        assert manifest_path.exists()
