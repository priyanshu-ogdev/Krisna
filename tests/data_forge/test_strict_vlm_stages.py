"""Strict VLM stages must infer, or fail before mutating records."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from data_forge.config import (
    DatasetSpec,
    ModelSpec,
    PipelineConfig,
    StageConfig,
    load_config,
)
from data_forge.inference.structured_output import (
    CaptionOutput,
    OCROutput,
    StructureOutput,
)
from data_forge.orchestrator import Orchestrator
from data_forge.stages.base import MissingVLMEngineError
from data_forge.stages.s00_manifest_planning import ManifestPlanningStage
from data_forge.stages.s01_6_preference_pairs import PreferencePairsStage
from data_forge.stages.s01_7_preference_pair_pii import PreferencePairPIIStage
from data_forge.stages.s02_dedup import DedupStage
from data_forge.stages.s03_5_pii_scrub import PIIScrubStage
from data_forge.stages.s03_quality import QualityStage
from data_forge.stages.s04_5_escalation import EscalationStage
from data_forge.stages.s04_safety import SafetyStage
from data_forge.stages.s05_ocr_enrichment import OCREnrichmentStage
from data_forge.stages.s05_recaption import RecaptionStage
from data_forge.stages.s06_structure import StructureStage
from data_forge.stages.s10_audit import AuditStage


def _structure(label="VLM result"):
    return StructureOutput(
        elements=[{"type": "button", "bbox": [0.1, 0.1, 0.5, 0.3], "label": label}],
        layout_type="form", hierarchy_depth=1, background_style="solid_light",
    )


def _record(manifest, config, sample_image, status, **kwargs):
    record = manifest.create_record(
        source_dataset=kwargs.pop("source_dataset", "test"),
        image_path=str(sample_image.relative_to(config.data_root)),
    )
    manifest.update_record(record.id, "setup", new_status=status, **kwargs)
    return manifest.get_record(record.id)


@pytest.mark.parametrize("stage,status,fields", [
    (QualityStage, "deduped", {}),
    (SafetyStage, "pii_scrubbed", {}),
    (RecaptionStage, "safety_classified", {"safety_tier": "safe"}),
    (StructureStage, "recaptioned", {}),
    (OCREnrichmentStage, "structured", {}),
    (OCREnrichmentStage, "structured", {"domain": "general_design"}),
    (EscalationStage, "safety_classified", {"safety_tier": "borderline"}),
    (AuditStage, "training_pool", {}),
])
async def test_strict_required_engine_fails_before_record_mutation(
    stage, status, fields, config, manifest, sample_image,
):
    config.strict_vlm_no_fallback = True
    record = _record(manifest, config, sample_image, status, **fields)

    with pytest.raises(MissingVLMEngineError, match=f"{stage.name} requires an active VLM engine"):
        await stage().run(manifest, config, [record.id], engine=None)

    assert manifest.get_record(record.id) == record
    assert not (config.resolved_paths["audit_reports"] / "latest_audit.json").exists()


@pytest.mark.parametrize("stage", [PreferencePairsStage, PreferencePairPIIStage])
async def test_strict_preference_pair_vlm_guard(stage, config, manifest):
    config.strict_vlm_no_fallback = True
    config.datasets["pairs"] = DatasetSpec(
        display_name="Pairs", source_type="huggingface", category="dpo_preference_general",
        fetch_config={"download_mode": "preference_pair"},
    )
    pair_dir = config.resolved_paths["preference_pairs"] / "pairs"
    pair_dir.mkdir()
    metadata = pair_dir / "pair.json"
    metadata.write_text('{"pair_id":"untouched"}', encoding="utf-8")

    with pytest.raises(MissingVLMEngineError, match=stage.name):
        await stage().run(manifest, config, [], engine=None)

    assert metadata.read_text(encoding="utf-8") == '{"pair_id":"untouched"}'


async def test_missing_engine_legacy_quality_skip_is_preserved(config, manifest, sample_image):
    assert config.strict_vlm_no_fallback is False
    record = _record(manifest, config, sample_image, "deduped")
    result = await QualityStage().run(manifest, config, [record.id], engine=None)
    assert result.records_processed == 0
    assert result.success
    assert manifest.get_record(record.id) == record


@pytest.mark.parametrize("strict", [False, True])
async def test_source_structure_cannot_substitute_for_missing_engine_in_strict_mode(
    strict, config, manifest, sample_image,
):
    config.strict_vlm_no_fallback = strict
    record = _record(
        manifest, config, sample_image, "recaptioned",
        source_structure=_structure("Source annotation").model_dump(),
    )
    if strict:
        with pytest.raises(MissingVLMEngineError, match="source/empty/freeform fallback"):
            await StructureStage().run(manifest, config, [record.id], engine=None)
        assert manifest.get_record(record.id) == record
    else:
        result = await StructureStage().run(manifest, config, [record.id], engine=None)
        assert result.records_processed == 1
        assert manifest.get_record(record.id).structure_output == record.source_structure


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("fallback_fields", [
    {"source_structure": _structure("Source annotation").model_dump()},
    {"domain": "general_design"},
    {"source_dataset": "pd12m"},
])
async def test_structure_strict_mode_infers_instead_of_source_or_general_fallback(
    strict, fallback_fields, config, manifest, sample_image,
):
    config.strict_vlm_no_fallback = strict
    record = _record(manifest, config, sample_image, "recaptioned", **fallback_fields)
    output = _structure()
    with patch("data_forge.stages.s06_structure.Tier1Engine") as tier1_class:
        tier1_class.return_value.extract_structure = AsyncMock(return_value=output)
        result = await StructureStage().run(manifest, config, [record.id], engine=object())
        if strict:
            tier1_class.return_value.extract_structure.assert_awaited_once_with(sample_image)
            assert manifest.get_record(record.id).structure_output == output.model_dump()
        else:
            tier1_class.return_value.extract_structure.assert_not_awaited()
            actual = manifest.get_record(record.id).structure_output
            if record.source_structure:
                assert actual == record.source_structure
            else:
                assert actual["layout_type"] == "freeform"
                assert actual["elements"] == []
    assert result.records_processed == 1


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("fields", [{"domain": "general_design"}, {"source_dataset": "pd12m"}])
async def test_ocr_strict_mode_infers_instead_of_empty_general_design_output(
    strict, fields, config, manifest, sample_image,
):
    config.strict_vlm_no_fallback = strict
    record = _record(manifest, config, sample_image, "structured", **fields)
    output = OCROutput(
        text_regions=[{"text": "Submit", "role": "button", "bbox": [0.1, 0.1, 0.5, 0.3], "confidence": 0.95}],
        total_text_regions=1, confidence=0.95,
    )
    with patch("data_forge.stages.s05_ocr_enrichment.OCREngine") as ocr_class:
        ocr_class.return_value.extract_text = AsyncMock(return_value=output)
        result = await OCREnrichmentStage().run(manifest, config, [record.id], engine=object())
        if strict:
            ocr_class.return_value.extract_text.assert_awaited_once_with(sample_image)
            assert manifest.get_record(record.id).ocr_output == output.model_dump()
        else:
            ocr_class.assert_not_called()
            assert manifest.get_record(record.id).ocr_output["text_regions"] == []
    assert result.records_processed == 1


@pytest.mark.parametrize("stage,module,method,status,output_field", [
    (StructureStage, "s06_structure", "extract_structure", "recaptioned", "structure_output"),
    (OCREnrichmentStage, "s05_ocr_enrichment", "extract_text", "structured", "ocr_output"),
])
async def test_strict_failed_inference_does_not_fabricate_general_output(
    stage, module, method, status, output_field, config, manifest, sample_image,
):
    config.strict_vlm_no_fallback = True
    record = _record(
        manifest, config, sample_image, status, domain="general_design",
        source_structure=_structure("Source annotation").model_dump(),
    )
    wrapper = "Tier1Engine" if stage is StructureStage else "OCREngine"
    with patch(f"data_forge.stages.{module}.{wrapper}") as engine_class:
        setattr(engine_class.return_value, method, AsyncMock(return_value=None))
        result = await stage().run(manifest, config, [record.id], engine=object())

    assert result.records_failed == 1
    updated = manifest.get_record(record.id)
    assert updated.status == "excluded_failed"
    assert getattr(updated, output_field) is None


async def test_strict_policy_keeps_deterministic_dedup_engine_free(config, manifest, sample_image):
    config.strict_vlm_no_fallback = True
    records = [
        _record(manifest, config, sample_image, "fetched", content_hash_sha256="same", perceptual_hash="same")
        for _ in range(2)
    ]
    result = await DedupStage().run(manifest, config, [r.id for r in records], engine=None)
    assert result.records_excluded == 1
    assert {manifest.get_record(r.id).status for r in records} == {"deduped", "excluded_duplicate"}


async def test_strict_policy_keeps_deterministic_face_scrub_engine_free(config, manifest, sample_image):
    config.strict_vlm_no_fallback = True
    record = _record(manifest, config, sample_image, "quality_scored")
    with (
        patch("data_forge.stages.s03_5_pii_scrub.load_face_detector", return_value=MagicMock()),
        patch("data_forge.stages.s03_5_pii_scrub.blur_faces", side_effect=lambda image, *args: (image, False, [])),
    ):
        result = await PIIScrubStage().run(manifest, config, [record.id], engine=None)
    assert result.records_processed == 1
    updated = manifest.get_record(record.id)
    assert updated.pii_scrubbed is True
    assert updated.status == "pii_scrubbed"
    assert (config.data_root / updated.scrubbed_image_path).is_file()


async def test_strict_orchestrator_cannot_enter_unified_fallback_path(
    config, manifest, sample_image, monkeypatch,
):
    config.strict_vlm_no_fallback = True
    names = ["s05_recaption", "s06_structure", "s05_ocr_enrichment"]
    config.stages = {name: StageConfig() for name in names}
    config.models = {"tier1": ModelSpec("test-model"), "ocr": ModelSpec("test-model")}
    record = _record(
        manifest, config, sample_image, "safety_classified", safety_tier="safe",
        domain="general_design", source_structure=_structure("Source annotation").model_dump(),
    )

    @asynccontextmanager
    async def session(*args, **kwargs):
        yield object()

    from data_forge.inference.engine import ModelEngine

    monkeypatch.setattr(ModelEngine, "vllm_session", staticmethod(session))
    with (
        patch("data_forge.inference.vlm_batch.VLMUnifiedPassCoordinator") as coordinator,
        patch("data_forge.stages.s05_recaption.Tier1Engine") as recaption_class,
        patch("data_forge.stages.s06_structure.Tier1Engine") as structure_class,
        patch("data_forge.stages.s05_ocr_enrichment.OCREngine") as ocr_class,
    ):
        recaption_class.return_value.generate_caption = AsyncMock(return_value=CaptionOutput(
            caption="A sufficiently detailed caption of the screen and its submit button.",
            ui_elements_mentioned=["button"], confidence=0.95,
        ))
        structure_class.return_value.extract_structure = AsyncMock(return_value=_structure())
        ocr_class.return_value.extract_text = AsyncMock(return_value=OCROutput(confidence=0.95))
        await Orchestrator(config, manifest).execute_pipeline(stages_filter=names, resume=False)
        coordinator.assert_not_called()
        recaption_class.return_value.generate_caption.assert_awaited_once()
        structure_class.return_value.extract_structure.assert_awaited_once()
        ocr_class.return_value.extract_text.assert_awaited_once()
    updated = manifest.get_record(record.id)
    assert updated.structure_output == _structure().model_dump()
    assert updated.ocr_output["confidence"] == 0.95


@pytest.mark.parametrize("value", [True, False])
def test_load_strict_policy_flag(tmp_path, monkeypatch, value):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(f"pipeline:\n  strict_vlm_no_fallback: {str(value).lower()}\n", encoding="utf-8")
    config = load_config(pipeline, tmp_path / "no_models.yaml", tmp_path / "no_datasets.yaml")
    assert config.strict_vlm_no_fallback is value


@pytest.mark.parametrize("value", ['"false"', "null", "1"])
def test_load_rejects_ambiguous_strict_policy(tmp_path, value):
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(f"pipeline:\n  strict_vlm_no_fallback: {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict_vlm_no_fallback must be a boolean"):
        load_config(pipeline, tmp_path / "no_models.yaml", tmp_path / "no_datasets.yaml")


def test_direct_config_rejects_non_boolean_strict_policy():
    with pytest.raises(ValueError, match="strict_vlm_no_fallback must be a boolean"):
        PipelineConfig(strict_vlm_no_fallback="false")


@pytest.mark.asyncio
async def test_strict_preflight_fails_before_pipeline_mutation(config, manifest):
    config.strict_vlm_no_fallback = True
    config.dry_run = False
    with patch(
        "data_forge.stages.s00_manifest_planning._production_preflight",
        return_value=["CUDA is unavailable"],
    ), pytest.raises(RuntimeError, match="Strict local-VLM preflight failed"):
        await ManifestPlanningStage().run(manifest, config, [])
