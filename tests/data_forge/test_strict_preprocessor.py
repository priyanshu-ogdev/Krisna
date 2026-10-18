from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from data_forge.config import ModelSpec
from data_forge.inference.preprocessor import LocalVLMPreprocessor, NoLiveVLMEngineError
from data_forge.inference.structured_output import (
    CaptionOutput,
    OCROutput,
    QualityOutput,
    SafetyOutput,
    StructureOutput,
)


def _live_engine() -> MagicMock:
    engine = MagicMock()
    engine.vllm_client = object()
    engine.current_model = "tier1"
    return engine


def _tier1() -> MagicMock:
    tier1 = MagicMock()
    tier1.generate_caption = AsyncMock(
        return_value=CaptionOutput(
            caption="A detailed login screen with a title, email field, password field, and submit button.",
            ui_elements_mentioned=["text_input", "button"],
            confidence=0.95,
        )
    )
    tier1.extract_structure = AsyncMock(
        return_value=StructureOutput(
            elements=[], layout_type="form", hierarchy_depth=0, background_style="solid_light"
        )
    )
    tier1.score_quality = AsyncMock(
        return_value=QualityOutput(
            aesthetic_score=0.8,
            resolution_adequate=True,
            is_complete_ui=True,
            design_era="modern",
            confidence=0.9,
        )
    )
    tier1.classify_safety = AsyncMock(
        return_value=SafetyOutput(
            tier="safe", confidence=0.99, rationale="A benign interface with no unsafe visual content."
        )
    )
    return tier1


def _ocr() -> MagicMock:
    ocr = MagicMock()
    ocr.extract_text = AsyncMock(return_value=OCROutput(confidence=0.9))
    return ocr


@pytest.mark.asyncio
async def test_strict_preprocessor_rejects_missing_engine(config, manifest, sample_image) -> None:
    config.models = {"tier1": ModelSpec(model_id="test-tier1")}
    record = manifest.create_record(source_dataset="test", image_path=str(sample_image.relative_to(config.data_root)))

    with pytest.raises(NoLiveVLMEngineError, match="live ModelEngine"):
        await LocalVLMPreprocessor(config, manifest).process([record])

    assert manifest.get_record(record.id).status == "fetched"


@pytest.mark.asyncio
async def test_strict_preprocessor_commits_valid_bundle_and_atomic_receipt(
    config, manifest, sample_image, tmp_path: Path
) -> None:
    config.models = {"tier1": ModelSpec(model_id="test-tier1")}
    record = manifest.create_record(source_dataset="test", image_path=str(sample_image.relative_to(config.data_root)))
    receipt = tmp_path / "audit" / "receipt.json"

    result = await LocalVLMPreprocessor(
        config,
        manifest,
        _live_engine(),
        tier1_engine=_tier1(),
        ocr_engine=_ocr(),
        receipt_path=receipt,
    ).process([record])

    updated = manifest.get_record(record.id)
    assert result.processed == 1
    assert result.excluded == 0
    assert updated.status == "structured"
    assert updated.caption
    assert updated.structure_output is not None
    assert updated.ocr_output is not None
    assert updated.quality_output is not None
    assert updated.safety_tier == "safe"
    assert receipt.is_file()
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_data["engine"] == "live_vllm"
    assert receipt_data["fallbacks_allowed"] is False
    assert receipt_data["validated_outputs"] == ["caption", "structure", "ocr", "quality", "safety"]
    assert receipt_data["records"][0]["record_id"] == record.id


@pytest.mark.asyncio
async def test_strict_preprocessor_excludes_inference_failure_with_reason(
    config, manifest, sample_image, tmp_path: Path
) -> None:
    config.models = {"tier1": ModelSpec(model_id="test-tier1")}
    record = manifest.create_record(source_dataset="test", image_path=str(sample_image.relative_to(config.data_root)))
    tier1 = _tier1()
    tier1.extract_structure = AsyncMock(side_effect=RuntimeError("mock VLM unavailable"))

    result = await LocalVLMPreprocessor(
        config,
        manifest,
        _live_engine(),
        tier1_engine=tier1,
        ocr_engine=_ocr(),
        receipt_path=tmp_path / "receipt.json",
    ).process([record])

    updated = manifest.get_record(record.id)
    assert result.processed == 0
    assert result.excluded == 1
    assert updated.status == "excluded_failed"
    assert updated.exclusion_reason is not None
    assert "inference_or_schema_failure" in updated.exclusion_reason
    assert "RuntimeError" in updated.exclusion_reason
    assert updated.caption is None
