"""Focused checks for the Qwen-Image-2.1 paired export contract."""

from __future__ import annotations

import json
from pathlib import Path

from data_forge.config import DatasetSpec, StageConfig
from data_forge.stages.s12_model_data_export import ModelDataExportStage
from data_forge.utils.hashing import sha256_file
from PIL import Image


def _structure() -> dict:
    return {
        "elements": [{"type": "button", "bbox": [0.1, 0.1, 0.8, 0.2], "label": "Save"}],
        "layout_type": "form",
        "hierarchy_depth": 1,
        "background_style": "solid_light",
    }


def _make_record(manifest, data_root: Path, target: Path, *, condition: Path | None, domain: str = "ui_first"):
    target_rel = str(target.relative_to(data_root))
    condition_rel = str(condition.relative_to(data_root)) if condition else None
    record = manifest.create_record(
        source_dataset="test_ui",
        source_file="screen-01.png",
        image_path=target_rel,
    )
    paths = {"condition_image": condition_rel} if condition_rel else {}
    manifest.update_record(
        record.id,
        "test_setup",
        new_status="training_pool",
        domain=domain,
        scrubbed_image_path=target_rel,
        license_verified=True,
        pii_scrubbed=True,
        safety_tier="safe",
        caption="Polish this form screen into a production UI.",
        structure_output=_structure(),
        encoding_paths=paths,
        license_output={"repo_id": "example/test-ui", "source_revision": "a" * 40},
    )
    return manifest.get_record(record.id)


def _configure_test_source(config) -> None:
    config.stages["s09_heldout"] = StageConfig(
        enabled=True,
        params={"source_disjoint": True},
    )
    config.datasets["test_ui"] = DatasetSpec(
        display_name="Test UI",
        source_type="huggingface",
        repo_id="example/test-ui",
        category="web_ui",
        training_eligible=True,
        synthetic_status="real",
        provenance_url="https://example.test/test-ui",
        quality_tier="silver",
        training_targets=["polish_qwen_image21"],
        license_status="verified",
        revision="a" * 40,
    )


class TestQwenImage21Export:
    def test_exports_distinct_pair_with_hashes_and_provenance(
        self, manifest, config, data_root, sample_image
    ):
        _configure_test_source(config)
        condition = data_root / "processed" / "wireframes" / "condition.png"
        condition.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), color=(240, 240, 240)).save(condition)
        record = _make_record(manifest, data_root, sample_image, condition=condition)

        summary = ModelDataExportStage()._export_qwen_image21(
            [record], config, config.resolved_paths["model_data_root"]
        )

        model_dir = config.resolved_paths["model_data_root"] / "polish_qwen_image21"
        entry = json.loads((model_dir / "manifest.jsonl").read_text().strip())
        assert summary["pairs_exported"] == 1
        assert entry["record_id"] == record.id
        assert entry["caption"] == entry["prompt"]
        assert entry["image"] == entry["target_image"]
        assert entry["source_dataset"] == "test_ui"
        assert entry["split"] == "train"
        assert entry["source_revision"] == "a" * 40
        assert entry["source_provenance"]["repo_id"] == "example/test-ui"

        condition_path = model_dir / entry["condition_image"]
        target_path = model_dir / entry["target_image"]
        assert condition_path.is_file()
        assert target_path.is_file()
        assert entry["condition_sha256"] == sha256_file(condition_path)
        assert entry["target_sha256"] == sha256_file(target_path)
        assert entry["content_hashes"] == {
            "condition_image": entry["condition_sha256"],
            "target_image": entry["target_sha256"],
        }

    def test_excludes_general_visual_record_even_if_it_has_structure_and_condition(
        self, manifest, config, data_root, sample_image
    ):
        _configure_test_source(config)
        condition = data_root / "processed" / "wireframes" / "condition.png"
        condition.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), color=(240, 240, 240)).save(condition)
        record = _make_record(
            manifest, data_root, sample_image, condition=condition, domain="general_design"
        )
        manifest.update_record(record.id, "test_setup", source_dataset="pd12m")
        record = manifest.get_record(record.id)

        ModelDataExportStage()._export_qwen_image21(
            [record], config, config.resolved_paths["model_data_root"]
        )

        model_dir = config.resolved_paths["model_data_root"] / "polish_qwen_image21"
        assert (model_dir / "manifest.jsonl").read_text() == ""
        exclusion = json.loads((model_dir / "exclusions.jsonl").read_text().strip())
        assert exclusion["record_id"] == record.id
        assert exclusion["reason"] == "not_explicit_ui_source"

    def test_records_renderer_unavailable_explicitly(
        self, manifest, config, data_root, sample_image
    ):
        _configure_test_source(config)
        record = _make_record(manifest, data_root, sample_image, condition=None)

        ModelDataExportStage()._export_qwen_image21(
            [record], config, config.resolved_paths["model_data_root"]
        )

        model_dir = config.resolved_paths["model_data_root"] / "polish_qwen_image21"
        summary = json.loads((model_dir / "manifest_summary.json").read_text())
        assert summary["pairs_exported"] == 0
        assert summary["condition_renderer"]["available"] is False
        assert summary["exclusions_by_reason"]["condition_renderer_unavailable"] == 1
        assert "exclusions.jsonl" in summary["note"]

    def test_rejects_policy_unapproved_record_before_pairing(
        self, manifest, config, data_root, sample_image
    ):
        _configure_test_source(config)
        condition = data_root / "processed" / "wireframes" / "condition.png"
        condition.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), color=(240, 240, 240)).save(condition)
        record = _make_record(manifest, data_root, sample_image, condition=condition)
        manifest.update_record(record.id, "test_setup", license_verified=False)
        record = manifest.get_record(record.id)

        ModelDataExportStage()._export_qwen_image21(
            [record], config, config.resolved_paths["model_data_root"]
        )

        model_dir = config.resolved_paths["model_data_root"] / "polish_qwen_image21"
        summary = json.loads((model_dir / "manifest_summary.json").read_text())
        assert summary["pairs_exported"] == 0
        assert summary["exclusions_by_reason"]["source_license_not_approved"] == 1

    def test_requires_stage9_source_disjoint_split(self, manifest, config, data_root, sample_image):
        _configure_test_source(config)
        config.stages["s09_heldout"] = StageConfig(
            enabled=True,
            params={"source_disjoint": False},
        )
        condition = data_root / "processed" / "wireframes" / "condition.png"
        condition.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), color=(240, 240, 240)).save(condition)
        record = _make_record(manifest, data_root, sample_image, condition=condition)

        ModelDataExportStage()._export_qwen_image21(
            [record], config, config.resolved_paths["model_data_root"]
        )

        model_dir = config.resolved_paths["model_data_root"] / "polish_qwen_image21"
        summary = json.loads((model_dir / "manifest_summary.json").read_text())
        assert summary["pairs_exported"] == 0
        assert summary["exclusions_by_reason"]["source_disjoint_split_not_configured"] == 1

    def test_exports_instruction_editing_pair_with_freeform_structure(
        self, manifest, config, data_root, sample_image
    ):
        _configure_test_source(config)
        config.datasets["magicbrush"] = DatasetSpec(
            display_name="MagicBrush",
            source_type="huggingface",
            repo_id="osunlp/MagicBrush",
            category="instruction_image_editing",
            training_eligible=True,
            synthetic=False,
            synthetic_status="real",
            provenance_url="https://example.test/magicbrush",
            quality_tier="gold",
            training_targets=["polish_qwen_image21"],
            license_status="verified",
            revision="a" * 40,
        )
        condition = data_root / "processed" / "source_photos" / "source.png"
        condition.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), color=(240, 240, 240)).save(condition)
        record = _make_record(
            manifest, data_root, sample_image, condition=condition, domain="ui_first"
        )
        manifest.update_record(
            record.id,
            "test_setup",
            source_dataset="magicbrush",
            structure_output={"layout_type": "freeform", "elements": []},
        )
        record = manifest.get_record(record.id)

        summary = ModelDataExportStage()._export_qwen_image21(
            [record], config, config.resolved_paths["model_data_root"]
        )
        model_dir = config.resolved_paths["model_data_root"] / "polish_qwen_image21"
        exclusions_text = (model_dir / "exclusions.jsonl").read_text() if (model_dir / "exclusions.jsonl").is_file() else ""
        assert summary["pairs_exported"] == 1, f"Export failed. Summary: {summary}, Exclusions: {exclusions_text}"
        entry = json.loads((model_dir / "manifest.jsonl").read_text().strip())
        assert entry["record_id"] == record.id
        assert entry["source_dataset"] == "magicbrush"

