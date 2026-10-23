"""Comprehensive tests for Structure-to-LayoutIR conversion and deterministic wireframe synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from data_forge.config import DatasetSpec, PipelineConfig, StageConfig
from data_forge.stages.s12_model_data_export import ModelDataExportStage
from data_forge.utils.structure_converter import (
    TAXONOMY_MAP,
    _sanitize_bbox,
    structure_to_layout_ir,
)
from data_forge.utils.wireframe_renderer import (
    RENDERER_SPEC_HASH,
    RENDERER_VERSION,
    render_wireframe,
    synthesize_record_wireframe,
)


class TestStructureConverter:
    def test_ocr_verbatim_fusion_replaces_vlm_generic_labels(self):
        """Verify OCR verbatim text replaces VLM generic purpose strings."""
        structure = {
            "layout_type": "form",
            "hierarchy_depth": 1,
            "elements": [
                {
                    "type": "button",
                    "bbox": [0.1, 0.7, 0.4, 0.8],
                    "label": "Primary submit action button",
                },
                {
                    "type": "text_input",
                    "bbox": [0.1, 0.3, 0.9, 0.45],
                    "label": "Search query input",
                },
            ],
        }
        ocr_output = {
            "text_regions": [
                {
                    "text": "Deploy to Production",
                    "bbox": [0.12, 0.72, 0.38, 0.78],
                },
                {
                    "text": "Search repositories...",
                    "bbox": [0.15, 0.33, 0.60, 0.42],
                },
            ],
        }

        layout = structure_to_layout_ir(structure, ocr_output, canvas_w=800, canvas_h=600)

        nodes_by_type = {n.type: n for n in layout.nodes if n.id != "root"}
        assert "button" in nodes_by_type
        assert "input" in nodes_by_type  # normalized from text_input

        # OCR text fused directly into the button node
        assert nodes_by_type["button"].text == "Deploy to Production"
        assert nodes_by_type["input"].text == "Search repositories..."

    def test_depth_aware_z_indexing_prevents_container_occlusion(self):
        """Verify nested children get higher z-indices than enclosing cards/containers."""
        structure = {
            "layout_type": "dashboard",
            "hierarchy_depth": 2,
            "elements": [
                {
                    "type": "container",
                    "bbox": [0.05, 0.05, 0.95, 0.95],
                    "children": [
                        {
                            "type": "card",
                            "bbox": [0.1, 0.1, 0.9, 0.5],
                            "children": [
                                {
                                    "type": "button",
                                    "bbox": [0.15, 0.35, 0.45, 0.45],
                                    "label": "Confirm",
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        layout = structure_to_layout_ir(structure, canvas_w=1024, canvas_h=768)
        non_root_nodes = [n for n in layout.nodes if n.id != "root"]
        assert len(non_root_nodes) == 3

        container_node = [n for n in non_root_nodes if n.type == "container"][0]
        card_node = [n for n in non_root_nodes if n.type == "card"][0]
        button_node = [n for n in non_root_nodes if n.type == "button"][0]

        # Strictly ascending z-index hierarchy
        assert container_node.z_index < card_node.z_index
        assert card_node.z_index < button_node.z_index
        assert button_node.parent == card_node.id
        assert card_node.parent == container_node.id

    def test_rectifies_inverted_and_degenerate_boxes(self):
        """Verify inverted coordinates and zero-area boxes are sanitized cleanly."""
        # Inverted box
        bbox_inv = _sanitize_bbox([0.8, 0.9, 0.2, 0.3])
        assert bbox_inv is not None
        x1, y1, x2, y2 = bbox_inv
        assert x1 == 0.2 and x2 == 0.8
        assert y1 == 0.3 and y2 == 0.9

        # Zero-area point/line rejected
        assert _sanitize_bbox([0.5, 0.5, 0.5, 0.5]) is None
        assert _sanitize_bbox([0.2, 0.3, 0.2, 0.8]) is None

        # Out-of-bounds clamped
        bbox_oob = _sanitize_bbox([-0.1, -0.2, 1.5, 1.2])
        assert bbox_oob is not None
        assert bbox_oob[0] >= 0.0 and bbox_oob[1] >= 0.0
        assert bbox_oob[2] <= 1.0 and bbox_oob[3] <= 1.0

    def test_taxonomy_normalization(self):
        """Verify fine-grained VLM types map into canonical renderable archetypes."""
        assert TAXONOMY_MAP["text_input"] == "input"
        assert TAXONOMY_MAP["navigation_bar"] == "container"
        assert TAXONOMY_MAP["tab_bar"] == "container"
        assert TAXONOMY_MAP["list_item"] == "card"
        assert TAXONOMY_MAP["text_label"] == "text"


class TestWireframeRenderer:
    def test_renders_high_contrast_3channel_rgb(self, tmp_path):
        """Verify wireframe renderer outputs standard 3-channel RGB image with correct size."""
        structure = {
            "layout_type": "form",
            "hierarchy_depth": 1,
            "elements": [
                {"type": "card", "bbox": [0.05, 0.05, 0.95, 0.95]},
                {"type": "button", "bbox": [0.1, 0.7, 0.5, 0.85], "label": "Save Changes"},
                {"type": "text_input", "bbox": [0.1, 0.2, 0.9, 0.35], "label": "Full Name"},
            ],
        }
        layout = structure_to_layout_ir(structure, canvas_w=720, canvas_h=1280, style="dark")
        out_png = tmp_path / "test_wireframe.png"

        result = render_wireframe(layout, out_png, style="dark")
        assert result.is_file()
        assert RENDERER_SPEC_HASH is not None
        assert RENDERER_VERSION.startswith("krivex.renderer.v2.")

        with Image.open(result) as img:
            assert img.mode == "RGB"
            assert img.size == (720, 1280)

    def test_synthesize_record_wireframe_matches_target_dimensions(self, tmp_path):
        """Verify synthesize_record_wireframe extracts target image dimensions exactly."""
        data_root = tmp_path / "data_root"
        data_root.mkdir(parents=True, exist_ok=True)
        img_p = data_root / "scrubbed_target.png"
        Image.new("RGB", (640, 480), color=(30, 30, 30)).save(img_p)

        class FakeRecord:
            id = "rec_001"
            scrubbed_image_path = "scrubbed_target.png"
            structure_output = {
                "layout_type": "form",
                "hierarchy_depth": 1,
                "elements": [{"type": "button", "bbox": [0.2, 0.6, 0.8, 0.8], "label": "Login"}],
            }
            ocr_output = {
                "text_regions": [{"text": "Log In", "bbox": [0.25, 0.65, 0.75, 0.75]}],
            }

        config = PipelineConfig()
        config.data_root = data_root

        out_wf = synthesize_record_wireframe(FakeRecord(), config)
        assert out_wf is not None
        assert out_wf.is_file()

        with Image.open(out_wf) as img:
            assert img.mode == "RGB"
            assert img.size == (640, 480)


class TestS12ExportWithSynthesizedWireframe:
    def test_exports_with_zero_renderer_exclusions_when_wireframe_present(
        self, manifest, config, data_root, sample_image
    ):
        """Verify end-to-end export has 100% pass rate when condition wireframe is synthesized."""
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

        target_rel = str(sample_image.relative_to(data_root))
        record = manifest.create_record(
            source_dataset="test_ui",
            source_file="screen-01.png",
            image_path=target_rel,
        )
        manifest.update_record(
            record.id,
            "test_setup",
            new_status="training_pool",
            domain="ui_first",
            scrubbed_image_path=target_rel,
            license_verified=True,
            pii_scrubbed=True,
            safety_tier="safe",
            caption="Polish this form into a production dashboard.",
            structure_output={
                "elements": [{"type": "button", "bbox": [0.1, 0.1, 0.8, 0.2], "label": "Submit"}],
                "layout_type": "form",
                "hierarchy_depth": 1,
                "background_style": "solid_light",
            },
            ocr_output={
                "text_regions": [{"text": "Submit Form", "bbox": [0.15, 0.12, 0.75, 0.18]}],
            },
            license_output={"repo_id": "example/test-ui", "source_revision": "a" * 40},
        )
        rec = manifest.get_record(record.id)

        # Synthesize wireframe condition image
        wf_path = synthesize_record_wireframe(rec, config)
        assert wf_path is not None
        assert wf_path.is_file()

        # Update manifest record with the condition image path
        manifest.update_record(
            rec.id,
            "s08_encoding",
            encoding_paths={"condition_image": str(wf_path.relative_to(data_root))},
        )
        rec = manifest.get_record(rec.id)

        # Export for Qwen Image 2.1
        summary = ModelDataExportStage()._export_qwen_image21(
            [rec], config, config.resolved_paths["model_data_root"]
        )

        assert summary["pairs_exported"] == 1
        assert summary["condition_renderer"]["available"] is True
        assert summary["exclusions_by_reason"].get("condition_renderer_unavailable", 0) == 0

        model_dir = config.resolved_paths["model_data_root"] / "polish_qwen_image21"
        entry = json.loads((model_dir / "manifest.jsonl").read_text().strip())
        assert entry["record_id"] == rec.id
        assert Path(model_dir / entry["condition_image"]).is_file()
        assert Path(model_dir / entry["target_image"]).is_file()
