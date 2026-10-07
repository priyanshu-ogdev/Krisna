"""Adversarial bug tests covering edge cases across pipeline stages s00-s12."""

import asyncio
import json
from pathlib import Path
import pytest
from PIL import Image

from data_forge.config import PipelineConfig, StageConfig
from data_forge.manifest import Manifest
from data_forge.stages.s03_quality import QualityStage
from data_forge.stages.s04_safety import SafetyStage
from data_forge.stages.s04_5_escalation import EscalationStage
from data_forge.stages.s05_recaption import RecaptionStage
from data_forge.stages.s05_ocr_enrichment import OCREnrichmentStage
from data_forge.stages.s05_5_pii_text_redact import PIITextRedactStage
from data_forge.stages.s06_structure import StructureStage
from data_forge.stages.s07_routing import RoutingStage
from data_forge.stages.s09_heldout import HeldoutStage
from data_forge.stages.s10_audit import AuditStage


@pytest.mark.asyncio
async def test_escalation_handles_exceptions_and_missing_images_without_dropping(manifest, config, sample_image, data_root):
    """Borderline records where Tier-2 fails, throws an exception, or encounters a missing image
    must be marked as excluded_pending_review, NEVER silently dropped."""
    rec1 = manifest.create_record(source_dataset="test", image_path="missing_file.png")
    manifest.update_record(rec1.id, "test", safety_tier="borderline", new_status="safety_classified")

    rel_sample = str(sample_image.relative_to(data_root))
    rec2 = manifest.create_record(source_dataset="test", image_path=rel_sample)
    manifest.update_record(rec2.id, "test", safety_tier="borderline", new_status="safety_classified")

    class MockFailingTier2Engine:
        def __init__(self, *args, **kwargs):
            pass
        async def reclassify_safety(self, img_path, tier1_output):
            raise RuntimeError("Tier-2 GPU inference crash simulation")

    import data_forge.stages.s04_5_escalation as s04_5_mod
    orig_engine = s04_5_mod.Tier2Engine
    s04_5_mod.Tier2Engine = MockFailingTier2Engine
    try:
        stage = EscalationStage()
        result = await stage.run(manifest, config, [rec1.id, rec2.id], engine=object())
        assert result.records_excluded == 2
        
        updated1 = manifest.get_record(rec1.id)
        assert updated1.status == "excluded_pending_review"
        assert updated1.exclusion_reason in ("image_missing", "escalation_exception")

        updated2 = manifest.get_record(rec2.id)
        assert updated2.status == "excluded_pending_review"
        assert updated2.exclusion_reason == "escalation_exception"
    finally:
        s04_5_mod.Tier2Engine = orig_engine


@pytest.mark.asyncio
async def test_stages_handle_none_image_paths_without_directory_open_errors(manifest, config):
    """Stages must cleanly exclude records with None image_path instead of resolving to data_root."""
    from unittest.mock import MagicMock
    from data_forge.config import ModelSpec
    config.models["tier1"] = ModelSpec(model_id="mock-tier1")
    mock_engine = MagicMock()
    mock_engine.vllm_client = MagicMock()
    rec = manifest.create_record(source_dataset="test", image_path=None)
    manifest.update_record(rec.id, "test", new_status="deduped")

    # Stage 3
    q_stage = QualityStage()
    q_res = await q_stage.run(manifest, config, [rec.id], engine=mock_engine)
    rec_after = manifest.get_record(rec.id)
    assert rec_after.status == "excluded_failed"
    assert rec_after.exclusion_reason == "image_missing"

    # Reset and test Stage 4
    manifest.update_record(rec.id, "test", new_status="pii_scrubbed")
    s_stage = SafetyStage()
    s_res = await s_stage.run(manifest, config, [rec.id], engine=mock_engine)
    rec_after = manifest.get_record(rec.id)
    assert rec_after.status == "excluded_failed"
    assert rec_after.exclusion_reason == "image_missing"


@pytest.mark.asyncio
async def test_pii_text_redact_handles_both_normalized_and_pixel_bboxes(manifest, config, sample_image, data_root):
    """Stage 5.5 must redact correctly whether OCR returned normalized [0-1] or pixel coordinates."""
    rel_sample = str(sample_image.relative_to(data_root))
    rec = manifest.create_record(
        source_dataset="test",
        image_path=rel_sample,
    )
    manifest.update_record(
        rec.id, "test",
        new_status="structured",
        scrubbed_image_path=rel_sample,
        ocr_output={
            "text_regions": [
                {"text": "secret@example.com", "bbox": [0.1, 0.1, 0.4, 0.2]},
                {"text": "555-123-4567", "bbox": [50, 60, 200, 100]},  # pixel coords
            ]
        }
    )
    config.stages["s05_5_pii_text_redact"] = StageConfig(
        enabled=True,
        params={
            "regex_patterns": {
                "email": r"[\w\.-]+@[\w\.-]+\.\w+",
                "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            }
        }
    )
    stage = PIITextRedactStage()
    res = await stage.run(manifest, config, [rec.id])
    assert res.records_processed == 1
    assert res.metadata["regions_redacted"] == 2
    
    updated = manifest.get_record(rec.id)
    assert "email_in_ocr_text" in updated.pii_detections
    assert "phone_in_ocr_text" in updated.pii_detections


@pytest.mark.asyncio
async def test_routing_accepts_recaptioned_records_when_structure_skipped(manifest, config):
    """Stage 7 routing must route records in recaptioned status if structure extraction was skipped."""
    config.stages["s07_routing"] = StageConfig(params={"ui_first_ratio": 1.0})
    rec = manifest.create_record(source_dataset="rico_core", image_path="x.png")
    manifest.update_record(rec.id, "test", new_status="recaptioned")

    stage = RoutingStage()
    res = await stage.run(manifest, config, [rec.id])
    assert res.records_processed == 1
    updated = manifest.get_record(rec.id)
    assert updated.status == "routed"
    assert updated.domain == "ui_first"


@pytest.mark.asyncio
async def test_heldout_sampling_is_deterministic(manifest, config):
    """Stage 9 heldout sampling with a configured seed must be completely reproducible."""
    rec_ids = []
    for i in range(20):
        rec = manifest.create_record(source_dataset="rico_core", image_path=f"img_{i}.png")
        manifest.update_record(
            rec.id, "test",
            new_status="encoded",
            domain="ui_first",
            encoding_paths={"z_image_latent": f"z_{i}.safetensors", "control_map": f"c_{i}.json"},
        )
        rec_ids.append(rec.id)

    stage = HeldoutStage()
    config.stages["s09_heldout"] = StageConfig(params={"heldout_fraction": 0.2, "random_seed": 12345})
    await stage.run(manifest, config, rec_ids)
    heldout_run1 = [r.id for r in manifest.get_heldout()]

    # Reset statuses
    for rid in rec_ids:
        manifest.update_record(rid, "test", new_status="encoded")
    await stage.run(manifest, config, rec_ids)
    heldout_run2 = [r.id for r in manifest.get_heldout()]

    assert heldout_run1 == heldout_run2
    assert len(heldout_run1) > 0


@pytest.mark.asyncio
async def test_inference_exceptions_do_not_crash_stages(manifest, config, sample_image, data_root):
    """Stages 03, 04, 05, 06 must handle inference exceptions per record without crashing the batch."""
    from unittest.mock import MagicMock, AsyncMock, patch
    from data_forge.config import ModelSpec
    config.models["tier1"] = ModelSpec(model_id="mock-tier1")
    config.stages["s03_quality"] = StageConfig(params={"min_resolution": [64, 64], "max_resolution": [4096, 4096]})

    rel_sample = str(sample_image.relative_to(data_root))
    rec = manifest.create_record(source_dataset="test", image_path=rel_sample, image_width=512, image_height=512)
    manifest.update_record(rec.id, "test", new_status="deduped")

    mock_engine = MagicMock()
    mock_engine.vllm_client = MagicMock()

    # Stage 3 exception
    with patch("data_forge.stages.s03_quality.Tier1Engine.score_quality", new_callable=AsyncMock, side_effect=RuntimeError("vLLM OOM")):
        q_stage = QualityStage()
        res = await q_stage.run(manifest, config, [rec.id], engine=mock_engine)
        assert res.records_failed == 1
        rec_after = manifest.get_record(rec.id)
        assert rec_after.status == "excluded_failed"
        assert rec_after.exclusion_reason == "inference_failed"

    # Stage 4 exception
    manifest.update_record(rec.id, "test", new_status="pii_scrubbed")
    with patch("data_forge.stages.s04_safety.Tier1Engine.classify_safety", new_callable=AsyncMock, side_effect=ConnectionResetError("Connection lost")):
        s_stage = SafetyStage()
        res = await s_stage.run(manifest, config, [rec.id], engine=mock_engine)
        assert res.records_failed == 1
        rec_after = manifest.get_record(rec.id)
        assert rec_after.status == "excluded_failed"
        assert rec_after.exclusion_reason == "inference_failed"

    # Stage 5 exception
    manifest.update_record(rec.id, "test", new_status="safety_classified", safety_tier="safe")
    with patch("data_forge.stages.s05_recaption.Tier1Engine.generate_caption", new_callable=AsyncMock, side_effect=TimeoutError("Request timed out")):
        r_stage = RecaptionStage()
        res = await r_stage.run(manifest, config, [rec.id], engine=mock_engine)
        assert res.records_failed == 1
        rec_after = manifest.get_record(rec.id)
        assert rec_after.status == "excluded_failed"
        assert rec_after.exclusion_reason == "inference_failed"

    # Stage 6 exception
    manifest.update_record(rec.id, "test", new_status="recaptioned")
    with patch("data_forge.stages.s06_structure.Tier1Engine.extract_structure", new_callable=AsyncMock, side_effect=Exception("Unexpected API error")):
        st_stage = StructureStage()
        res = await st_stage.run(manifest, config, [rec.id], engine=mock_engine)
        assert res.records_failed == 1
        rec_after = manifest.get_record(rec.id)
        assert rec_after.status == "excluded_failed"
        assert rec_after.exclusion_reason == "structure_extraction_failed"


@pytest.mark.asyncio
async def test_audit_stage_failing_records_not_promoted(manifest, config, sample_image, data_root):
    """AuditStage must never mark records with status='audited' if overall_pass is False."""
    from unittest.mock import MagicMock, AsyncMock, patch
    from data_forge.stages.s10_audit import AuditStage
    from data_forge.config import ModelSpec
    config.models["tier1"] = ModelSpec(model_id="mock-tier1")

    rel_sample = str(sample_image.relative_to(data_root))
    rec = manifest.create_record(source_dataset="test", image_path=rel_sample)
    manifest.update_record(
        rec.id, "test",
        new_status="training_pool",
        domain="ui_first",
        quality_output={"aesthetic_score": 0.2},
    )

    mock_engine = MagicMock()
    mock_engine.vllm_client = MagicMock()

    from data_forge.inference.structured_output import AuditOutput
    mock_failing_audit = AuditOutput(
        caption_matches_image=False,
        structure_matches_image=False,
        quality_issues=["low aesthetic score"],
        safety_issues=[],
        accuracy_issues=["caption mismatch"],
        hallucination_issues=[],
        overall_pass=False,
        confidence=0.9,
        rationale="Image does not match caption at all.",
    )

    with patch("data_forge.agents.audit_agent.AuditAgent.audit_record", new_callable=AsyncMock, return_value=mock_failing_audit):
        audit_stage = AuditStage()
        res = await audit_stage.run(manifest, config, [rec.id], engine=mock_engine)
        assert res.metadata["failed"] == 1
        assert res.metadata["passed"] == 0
        rec_after = manifest.get_record(rec.id)
        assert rec_after.status != "audited"


@pytest.mark.asyncio
async def test_registry_watcher_handles_malformed_json(manifest, config, tmp_path):
    """Stage 11 must handle corrupt/malformed registry JSON reports safely without throwing."""
    from data_forge.stages.s11_registry_watcher import RegistryWatcherStage
    reg_dir = tmp_path / "registry_reports"
    reg_dir.mkdir(parents=True, exist_ok=True)
    report_file = reg_dir / "latest.json"
    report_file.write_text("{malformed_json: true", encoding="utf-8")

    config.resolved_paths["registry_reports"] = reg_dir
    stage = RegistryWatcherStage()
    res = await stage.run(manifest, config, [])
    assert res.records_failed == 1


@pytest.mark.asyncio
async def test_registry_watcher_handles_valid_recommendations(manifest, config, tmp_path):
    """Stage 11 must parse valid recommendations without NameError."""
    from data_forge.stages.s11_registry_watcher import RegistryWatcherStage
    reg_dir = tmp_path / "registry_reports"
    reg_dir.mkdir(parents=True, exist_ok=True)
    report_file = reg_dir / "latest.json"
    report_file.write_text(
        json.dumps({
            "timestamp": "2026-10-07T00:00:00Z",
            "recommendations": [
                {"action": "swap", "model": "clip", "new_version": "v2", "reason": "better"},
                {"action": "investigate", "model": "ocr", "reason": "drift"},
            ]
        }),
        encoding="utf-8"
    )

    config.resolved_paths["registry_reports"] = reg_dir
    stage = RegistryWatcherStage()
    res = await stage.run(manifest, config, [])
    assert res.records_processed == 1
    assert res.metadata["recommendations"] == 2


def test_bulk_update_records_grouping(manifest):
    """Manifest bulk_update_records with executemany signature grouping must update all fields correctly."""
    rec1 = manifest.create_record(source_dataset="test_ds", source_file="1.png")
    rec2 = manifest.create_record(source_dataset="test_ds", source_file="2.png")
    rec3 = manifest.create_record(source_dataset="test_ds", source_file="3.png")

    updates = [
        {"id": rec1.id, "new_status": "quality_scored", "aesthetic_score": 0.85, "reason": "high quality"},
        {"id": rec2.id, "new_status": "quality_scored", "aesthetic_score": 0.90, "reason": "very high quality"},
        {"id": rec3.id, "new_status": "excluded_low_quality", "reason": "too low", "exclusion_reason": "low_score"},
    ]

    manifest.bulk_update_records(updates, stage="test_stage")

    r1 = manifest.get_record(rec1.id)
    r2 = manifest.get_record(rec2.id)
    r3 = manifest.get_record(rec3.id)

    assert r1.status == "quality_scored"
    assert r1.aesthetic_score == 0.85
    assert r2.status == "quality_scored"
    assert r2.aesthetic_score == 0.90
    assert r3.status == "excluded_low_quality"
    assert r3.exclusion_reason == "low_score"

    # Verify stage history was batch inserted
    hist1 = manifest._conn.execute(
        "SELECT * FROM stage_history WHERE record_id = ? AND stage = 'test_stage'",
        (rec1.id,),
    ).fetchall()
    assert len(hist1) == 1
    assert hist1[0]["new_status"] == "quality_scored"


