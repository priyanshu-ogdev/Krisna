"""Tests for the images/ linking fix in s12_model_data_export.py.

Both _export_sketch_tier and _export_zimage previously linked only their
processed artifacts (vq_tokens/, latents/) and never the raw scrubbed
images — meaning neither exported folder had a working consumer for the
tooling that actually needs raw images (the real Sketch tokenizer,
boris/vqgan_f16_16384, run by training/data_forge_bridge; the official
diffusers train_dreambooth_lora_z_image.py script, which takes
--instance_data_dir of raw images and computes its own latents
internally).

UPDATED (sync audit item #1): data-forge's own vq_tokens/ export has
since been removed entirely — its producer (the maskgit_vq encoder) never
worked, and nothing consumed vq_tokens/ even when the try/except silently
swallowed that failure. _export_sketch_tier now only links images/.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from data_forge.stages.s12_model_data_export import ModelDataExportStage


def _make_ui_first_record(manifest, data_root, sample_image, i: int):
    rel_path = str(sample_image.relative_to(data_root))
    rec = manifest.create_record(
        source_dataset="test_dataset", source_file=f"ui_{i}.png", image_path=rel_path,
    )
    manifest.update_record(
        rec.id, "test_setup", new_status="training_pool",
        domain="ui_first",
        scrubbed_image_path=rel_path,
        caption=f"a UI screen {i}",
        encoding_paths={
            "z_image_latent": rel_path, "control_map": rel_path,
        },
    )
    return rec.id


def _make_general_design_record(manifest, data_root, sample_image, i: int):
    rel_path = str(sample_image.relative_to(data_root))
    rec = manifest.create_record(
        source_dataset="test_dataset", source_file=f"gd_{i}.png", image_path=rel_path,
    )
    manifest.update_record(
        rec.id, "test_setup", new_status="training_pool",
        domain="general_design",
        scrubbed_image_path=rel_path,
        caption=f"general image {i}",
        encoding_paths={"z_image_latent": rel_path, "control_map": rel_path},
    )
    return rec.id


class TestSketchTierImagesLinked:
    def test_images_linked_no_vq_tokens_folder(self, manifest, config, data_root, sample_image):
        _make_ui_first_record(manifest, data_root, sample_image, 1)
        stage = ModelDataExportStage()
        result = asyncio.run(stage.run(manifest, config, []))

        model_dir = config.resolved_paths["model_data_root"] / "sketch_tier_maskgit"
        assert not (model_dir / "vq_tokens").exists()
        assert (model_dir / "images").exists()
        assert len(list((model_dir / "images").iterdir())) == 1

        summary = json.loads((model_dir / "manifest_summary.json").read_text())
        assert summary["images_linked"] == 1
        assert "vq_tokens_linked" not in summary


class TestZImageImagesLinked:
    def test_images_linked_alongside_latents(self, manifest, config, data_root, sample_image):
        # Two DISTINCT source images — reusing sample_image for both records
        # would give them the same filename and collapse to one linked file,
        # which isn't what this test is checking.
        from PIL import Image

        second_image = data_root / "raw" / "test_dataset" / "second_image.png"
        Image.new("RGB", (512, 512), color=(10, 20, 30)).save(second_image)

        _make_general_design_record(manifest, data_root, sample_image, 1)
        rec2_id = manifest.create_record(
            source_dataset="test_dataset", source_file="ui_2.png",
            image_path=str(second_image.relative_to(data_root)),
        ).id
        manifest.update_record(
            rec2_id, "test_setup", new_status="training_pool",
            domain="ui_first",
            scrubbed_image_path=str(second_image.relative_to(data_root)),
            caption="a UI screen 2",
            encoding_paths={
                "z_image_latent": str(second_image.relative_to(data_root)),
                "control_map": str(second_image.relative_to(data_root)),
            },
        )

        stage = ModelDataExportStage()
        asyncio.run(stage.run(manifest, config, []))

        model_dir = config.resolved_paths["model_data_root"] / "polish_zimage_turbo"
        assert (model_dir / "latents").exists()
        assert (model_dir / "images").exists()
        # Both domains feed Z-Image-Turbo (unlike the sketch tier, which is ui_first only).
        assert len(list((model_dir / "images").iterdir())) == 2

        summary = json.loads((model_dir / "manifest_summary.json").read_text())
        assert summary["images_linked"] == 2
        assert summary["latents_linked"] == 2

    def test_note_documents_which_folder_official_script_uses(self, manifest, config, data_root, sample_image):
        _make_general_design_record(manifest, data_root, sample_image, 1)
        stage = ModelDataExportStage()
        asyncio.run(stage.run(manifest, config, []))

        model_dir = config.resolved_paths["model_data_root"] / "polish_zimage_turbo"
        summary = json.loads((model_dir / "manifest_summary.json").read_text())
        assert "images/" in summary["note"]
        assert "official" in summary["note"].lower()


class TestZeroRecordsCreatesOutputDir:
    """Regression test for the actual FileNotFoundError bug — the two
    tests above only ever exercise the "at least one matching record"
    path, which never touches the mkdir-before-write ordering issue at
    all (link_or_copy() creates its own parent dirs when it runs, masking
    the gap). This test is the one that would have failed before the
    `model_dir.mkdir(parents=True, exist_ok=True)` fix at the top of
    _export_sketch_tier/_export_zimage: zero matching records means
    link_or_copy() never runs even once, so nothing creates model_dir
    before captions.jsonl's write — which used to raise FileNotFoundError
    on a fresh corpus or a domain with no qualifying records yet.
    """

    def test_sketch_tier_empty_ui_first_pool_does_not_crash(self, manifest, config, data_root, sample_image):
        # Only a general_design record exists — zero ui_first records,
        # so _export_sketch_tier's `records` list is empty.
        _make_general_design_record(manifest, data_root, sample_image, 1)
        stage = ModelDataExportStage()
        result = asyncio.run(stage.run(manifest, config, []))  # must not raise

        model_dir = config.resolved_paths["model_data_root"] / "sketch_tier_maskgit"
        assert model_dir.exists()
        assert (model_dir / "captions.jsonl").exists()
        assert (model_dir / "captions.jsonl").read_text() == ""
        summary = json.loads((model_dir / "manifest_summary.json").read_text())
        assert summary["records"] == 0

    def test_zimage_empty_training_pool_does_not_crash(self, manifest, config, data_root, sample_image):
        # No records reach training_pool status at all.
        stage = ModelDataExportStage()
        result = asyncio.run(stage.run(manifest, config, []))  # must not raise

        model_dir = config.resolved_paths["model_data_root"] / "polish_zimage_turbo"
        assert model_dir.exists()
        assert (model_dir / "captions.jsonl").exists()
        assert (model_dir / "captions.jsonl").read_text() == ""
        summary = json.loads((model_dir / "manifest_summary.json").read_text())
        assert summary["records"] == 0
        assert summary["images_linked"] == 0


class TestCaptionsIncludeImageFilename:
    """Regression test for a real bug found integrating with
    krisna-orchestrator's sync bridge: record_id is a random uuid4
    (manifest.py's create_record), completely unrelated to the linked
    image's on-disk filename (named after the original fetch-time file,
    e.g. "rico_core_0000042.png"). A downstream consumer joining
    captions.jsonl back to images/ by assuming filename.stem == record_id
    gets a 100% cache-miss on any real export. captions.jsonl must carry
    the actual linked filename as its own field so consumers don't have
    to guess.
    """

    def test_sketch_tier_captions_include_real_filename_not_derivable_from_record_id(
        self, manifest, config, data_root, sample_image
    ):
        rec_id = _make_ui_first_record(manifest, data_root, sample_image, 1)
        stage = ModelDataExportStage()
        asyncio.run(stage.run(manifest, config, []))

        model_dir = config.resolved_paths["model_data_root"] / "sketch_tier_maskgit"
        lines = [json.loads(l) for l in (model_dir / "captions.jsonl").read_text().splitlines()]
        assert len(lines) == 1
        entry = lines[0]
        assert entry["record_id"] == rec_id
        assert entry["image_filename"] == sample_image.name
        # The actual bug: record_id must NOT be derivable from the filename.
        assert entry["record_id"] != Path(entry["image_filename"]).stem

    def test_zimage_captions_include_real_filename(self, manifest, config, data_root, sample_image):
        rec_id = _make_general_design_record(manifest, data_root, sample_image, 1)
        stage = ModelDataExportStage()
        asyncio.run(stage.run(manifest, config, []))

        model_dir = config.resolved_paths["model_data_root"] / "polish_zimage_turbo"
        lines = [json.loads(l) for l in (model_dir / "captions.jsonl").read_text().splitlines()]
        assert len(lines) == 1
        assert lines[0]["image_filename"] == sample_image.name
        assert lines[0]["record_id"] == rec_id

    def test_image_filename_is_none_when_no_image_was_linked(self, manifest, config, data_root, sample_image):
        """A record with encoding artifacts but no scrubbed_image_path
        (shouldn't normally happen, but the field must degrade cleanly
        rather than crash) gets image_filename: null, not a KeyError."""
        rel_path = str(sample_image.relative_to(data_root))
        rec = manifest.create_record(
            source_dataset="test_dataset", source_file="no_image.png", image_path=rel_path,
        )
        manifest.update_record(
            rec.id, "test_setup", new_status="training_pool",
            domain="general_design",
            scrubbed_image_path=None,
            caption="orphan caption",
            encoding_paths={"z_image_latent": rel_path, "control_map": rel_path},
        )
        stage = ModelDataExportStage()
        asyncio.run(stage.run(manifest, config, []))

        model_dir = config.resolved_paths["model_data_root"] / "polish_zimage_turbo"
        lines = [json.loads(l) for l in (model_dir / "captions.jsonl").read_text().splitlines()]
        assert lines[0]["image_filename"] is None
