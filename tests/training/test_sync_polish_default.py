from __future__ import annotations

import json

import pytest
from PIL import Image

from krisna_training.data_forge_bridge.sync_polish_default import sync


def _make_data_forge_export(root, records):
    """records: list of (record_id, image_filename, caption).

    BUG FIX: this fixture used to name each image file after its
    record_id and include no image_filename field in captions.jsonl —
    baking in the exact false stem==record_id assumption sync_polish_
    default.py's old code made. Fixed the same way as
    test_sync_sketch_tier.py's fixture: record_id and image_filename are
    deliberately different, unrelated strings, matching what a real
    data-forge export actually produces.
    """
    zimage_dir = root / "polish_zimage_turbo"
    images_dir = zimage_dir / "images"
    images_dir.mkdir(parents=True)
    captions = []
    for record_id, image_filename, caption in records:
        Image.new("RGB", (4, 4), color="green").save(images_dir / image_filename)
        captions.append({"record_id": record_id, "caption": caption, "image_filename": image_filename})
    (zimage_dir / "captions.jsonl").write_text("\n".join(json.dumps(c) for c in captions))
    return root


def test_sync_bridges_record_keyed_captions_to_filename_keyed(tmp_path):
    model_data = _make_data_forge_export(
        tmp_path / "model_data",
        [
            ("uuid-aaa-111", "pd12m_0000042.png", "a minimalist login screen"),
            ("uuid-bbb-222", "cc12m_0000091.png", "a bold poster"),
        ],
    )
    out = sync(model_data, tmp_path / "prepared")

    metadata_lines = [json.loads(l) for l in (out / "metadata.jsonl").read_text().strip().split("\n")]
    by_name = {r["file_name"]: r["text"] for r in metadata_lines}
    assert by_name == {
        "pd12m_0000042.png": "a minimalist login screen",
        "cc12m_0000091.png": "a bold poster",
    }
    assert (out / "pd12m_0000042.png").exists()
    assert (out / "cc12m_0000091.png").exists()


def test_sync_missing_images_dir_raises_clear_error(tmp_path):
    model_data = tmp_path / "model_data"
    (model_data / "polish_zimage_turbo").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="s12_model_data_export"):
        sync(model_data, tmp_path / "prepared")


def test_sync_falls_back_to_shared_prompt_when_caption_missing(tmp_path):
    zimage_dir = tmp_path / "model_data" / "polish_zimage_turbo"
    images_dir = zimage_dir / "images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(images_dir / "webui_0000007.png")
    # No captions.jsonl at all.

    out = sync(tmp_path / "model_data", tmp_path / "prepared")
    metadata_lines = [json.loads(l) for l in (out / "metadata.jsonl").read_text().strip().split("\n")]
    assert metadata_lines[0]["text"] == "a UI design"


class TestRecordIdVsFilenameBugFix:
    """The actual regression test: captions must join correctly via the
    explicit image_filename field, not a filename-equals-record_id guess
    that never holds for real data-forge output."""

    def test_captions_match_via_image_filename_not_record_id_guess(self, tmp_path):
        model_data = _make_data_forge_export(
            tmp_path / "model_data",
            [("f47ac10b-58cc-4372-a567-0e02b2c3d479", "rico_semantic_0009981.png", "a settings panel")],
        )
        out = sync(model_data, tmp_path / "prepared")
        metadata_lines = [json.loads(l) for l in (out / "metadata.jsonl").read_text().strip().split("\n")]
        by_name = {r["file_name"]: r["text"] for r in metadata_lines}
        assert by_name["rico_semantic_0009981.png"] == "a settings panel"

    def test_legacy_captions_format_without_image_filename_warns_and_degrades(self, tmp_path, caplog):
        zimage_dir = tmp_path / "model_data" / "polish_zimage_turbo"
        images_dir = zimage_dir / "images"
        images_dir.mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(images_dir / "some_real_filename.png")
        (zimage_dir / "captions.jsonl").write_text(
            json.dumps({"record_id": "uuid-abc", "caption": "will not match"})
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="krisna_training.data_forge_bridge.sync_polish_default"):
            out = sync(tmp_path / "model_data", tmp_path / "prepared")

        assert any("legacy" in r.message.lower() for r in caplog.records)
        metadata_lines = [json.loads(l) for l in (out / "metadata.jsonl").read_text().strip().split("\n")]
        # Degrades to the shared-instance-prompt fallback, not a crash.
        assert metadata_lines[0]["text"] == "a UI design"
