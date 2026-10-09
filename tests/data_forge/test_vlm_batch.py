"""Tests for VLMRecordContext, VLMUnifiedPassCoordinator, and unified Phase 4 execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from data_forge.config import PathsConfig, PipelineConfig, StageConfig
from data_forge.inference.client import ImagePayloadCache
from data_forge.inference.structured_output import CaptionOutput, OCROutput, StructureOutput, TextRegion, UIElement
from data_forge.inference.vlm_batch import VLMRecordContext, VLMUnifiedPassCoordinator
from data_forge.manifest import Manifest, ManifestRecord
from data_forge.orchestrator import Orchestrator


@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "test_screen.png"
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_tier1():
    t1 = MagicMock()
    t1.generate_caption = AsyncMock(
        return_value=CaptionOutput(
            caption="A modern blue login screen for user authentication",
            ui_elements_mentioned=["button", "input"],
            confidence=0.98,
        )
    )
    t1.extract_structure = AsyncMock(
        return_value=StructureOutput(
            elements=[
                UIElement(
                    type="button",
                    label="Submit",
                    bbox=[0.1, 0.1, 0.3, 0.4],
                )
            ],
            layout_type="form",
            hierarchy_depth=1,
        )
    )
    return t1


@pytest.fixture
def mock_ocr():
    ocr = MagicMock()
    ocr.extract_text = AsyncMock(
        return_value=OCROutput(
            text_regions=[
                TextRegion(text="Submit", bbox=[0.1, 0.1, 0.3, 0.4], role="button_label")
            ],
            confidence=0.95,
        )
    )
    return ocr


@pytest.mark.asyncio
async def test_vlm_record_context_and_coordinator_success(
    tmp_path: Path,
    dummy_image: Path,
    mock_tier1,
    mock_ocr,
):
    config = PipelineConfig()
    config.data_root = tmp_path
    cache = ImagePayloadCache(max_size=10)

    coordinator = VLMUnifiedPassCoordinator(
        config=config,
        tier1_engine=mock_tier1,
        ocr_engine=mock_ocr,
        image_cache=cache,
    )

    manifest_rec = ManifestRecord(
        id="rec_001",
        source_dataset="test_ui",
        image_path=dummy_image.name,
        safety_tier="safe",
        status="safety_classified",
    )

    ctx = VLMRecordContext(record_id="rec_001", manifest_record=manifest_rec)
    res_ctx = await coordinator.process_record(ctx, run_recaption=True, run_structure=True, run_ocr=True)

    assert res_ctx.status == "structured"
    assert res_ctx.caption_output is not None
    assert res_ctx.caption_output.caption == "A modern blue login screen for user authentication"
    assert res_ctx.structure_output is not None
    assert len(res_ctx.structure_output.elements) == 1
    assert res_ctx.ocr_output is not None
    assert len(res_ctx.ocr_output.text_regions) == 1

    # Image payload was encoded and cached
    assert res_ctx.image_payload is not None
    mime, b64 = res_ctx.image_payload
    assert mime == "image/png"
    assert len(b64) > 0

    # Verify coordinator passed image_payload to tier1 and ocr
    mock_tier1.generate_caption.assert_awaited_once_with(
        res_ctx.image_path,
        source_caption_hint=None,
        image_payload=res_ctx.image_payload,
    )
    mock_tier1.extract_structure.assert_awaited_once_with(
        res_ctx.image_path,
        image_payload=res_ctx.image_payload,
    )
    mock_ocr.extract_text.assert_awaited_once_with(
        res_ctx.image_path,
        image_payload=res_ctx.image_payload,
    )

    # Convert to manifest updates
    updates = coordinator.to_manifest_updates([res_ctx], run_recaption=True, run_structure=True, run_ocr=True)
    assert len(updates) == 1
    up = updates[0]
    assert up["id"] == "rec_001"
    assert up["new_status"] == "structured"
    assert up["caption"] == "A modern blue login screen for user authentication"
    assert "structure_output" in up
    assert "ocr_output" in up


@pytest.mark.asyncio
async def test_vlm_coordinator_missing_image(tmp_path: Path, mock_tier1, mock_ocr):
    config = PipelineConfig()
    config.data_root = tmp_path

    coordinator = VLMUnifiedPassCoordinator(
        config=config,
        tier1_engine=mock_tier1,
        ocr_engine=mock_ocr,
    )

    manifest_rec = ManifestRecord(
        id="rec_missing",
        source_dataset="test_ui",
        image_path="non_existent.png",
        safety_tier="safe",
        status="safety_classified",
    )

    ctx = VLMRecordContext(record_id="rec_missing", manifest_record=manifest_rec)
    res_ctx = await coordinator.process_record(ctx, run_recaption=True, run_structure=True, run_ocr=True)

    assert res_ctx.status == "excluded_failed"
    assert res_ctx.exclusion_reason == "image_missing"

    updates = coordinator.to_manifest_updates([res_ctx])
    assert updates[0]["new_status"] == "excluded_failed"
    assert updates[0]["exclusion_reason"] == "image_missing"


@pytest.mark.asyncio
async def test_vlm_coordinator_subchunk_concurrency(
    tmp_path: Path,
    dummy_image: Path,
    mock_tier1,
    mock_ocr,
):
    config = PipelineConfig()
    config.data_root = tmp_path

    coordinator = VLMUnifiedPassCoordinator(
        config=config,
        tier1_engine=mock_tier1,
        ocr_engine=mock_ocr,
    )

    records = [
        ManifestRecord(
            id=f"rec_{i:03d}",
            source_dataset="test_ui",
            image_path=dummy_image.name,
            safety_tier="safe",
            status="safety_classified",
        )
        for i in range(5)
    ]

    contexts = await coordinator.process_subchunk(
        records,
        run_recaption=True,
        run_structure=True,
        run_ocr=False,
        concurrency=2,
    )

    assert len(contexts) == 5
    assert all(c.status == "structured" for c in contexts)
    assert mock_tier1.generate_caption.await_count == 5
    assert mock_tier1.extract_structure.await_count == 5


@pytest.mark.asyncio
async def test_orchestrator_phase4_unified_pass(
    tmp_path: Path,
    dummy_image: Path,
    monkeypatch,
):
    config = PipelineConfig()
    config.data_root = tmp_path
    config.paths = PathsConfig(checkpoints="checkpoints")
    config.chunk_size = 10
    config.checkpoint_enabled = True
    from data_forge.config import ModelSpec
    config.models = {
        "tier1": ModelSpec(model_id="test-tier1"),
    }
    config.stages = {
        "s05_recaption": StageConfig(enabled=True),
        "s06_structure": StageConfig(enabled=True),
    }

    manifest = Manifest(tmp_path / "manifest.db")
    rec = manifest.create_record(
        source_dataset="test_ui",
        image_path=dummy_image.name,
    )
    manifest.update_record(
        rec.id,
        "test_init",
        status="safety_classified",
        safety_tier="safe",
    )

    orch = Orchestrator(config, manifest)

    # Mock tier1 engine inside vllm_session
    mock_engine = MagicMock()
    mock_engine.vllm_client = MagicMock()

    class FakeVLMCoordinator:
        def __init__(self, cfg, t1, ocr):
            self.cfg = cfg

        async def process_subchunk(self, records, run_recaption=True, run_structure=True, run_ocr=False):
            return [
                VLMRecordContext(
                    record_id=r.id,
                    manifest_record=r,
                    status="structured",
                    caption_output=CaptionOutput(
                        caption="Unified test caption with details",
                        ui_elements_mentioned=["form"],
                        confidence=0.99,
                    ),
                    structure_output=StructureOutput(elements=[], layout_type="form", hierarchy_depth=1),
                )
                for r in records
            ]

        def to_manifest_updates(self, contexts, run_recaption=True, run_structure=True, run_ocr=False):
            return VLMUnifiedPassCoordinator.to_manifest_updates(
                contexts, run_recaption, run_structure, run_ocr
            )

    import data_forge.inference.vlm_batch as vlm_batch_mod
    monkeypatch.setattr(vlm_batch_mod, "VLMUnifiedPassCoordinator", FakeVLMCoordinator)

    # Run phase 4 via execute_pipeline
    import contextlib

    @contextlib.asynccontextmanager
    async def fake_session(*a, **k):
        yield mock_engine

    import data_forge.inference.engine as engine_module
    monkeypatch.setattr(engine_module.ModelEngine, "vllm_session", staticmethod(fake_session))

    results = await orch.execute_pipeline(stages_filter=["s05_recaption", "s06_structure"])

    updated_rec = manifest.get_record(rec.id)
    assert updated_rec.status == "structured"
    assert updated_rec.caption == "Unified test caption with details"
    assert updated_rec.structure_output is not None

    # Check that stage results were appended for both stages
    assert len(results) == 1
    stage_results = results[0].stage_results
    assert len(stage_results) == 2
    assert stage_results[0].stage_name == "s05_recaption"
    assert stage_results[0].records_processed == 1
    assert stage_results[1].stage_name == "s06_structure"
    assert stage_results[1].records_processed == 1

    # Checkpoints were marked
    assert orch._is_stage_complete("s05_recaption", "chunk_0000_0", [rec.id]) is True
    assert orch._is_stage_complete("s06_structure", "chunk_0000_0", [rec.id]) is True
