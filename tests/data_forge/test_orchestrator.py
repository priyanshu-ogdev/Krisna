import pytest
from data_forge.config import PipelineConfig, PathsConfig, StageConfig
from data_forge.manifest import Manifest
from data_forge.orchestrator import Orchestrator

@pytest.fixture
def config(tmp_path):
    c = PipelineConfig()
    c.data_root = tmp_path
    c.paths = PathsConfig(checkpoints="checkpoints")
    c.stages = {
        "s05_recaption": StageConfig(enabled=True),
        "s10_audit": StageConfig(enabled=True)
    }
    return c

def test_orchestrator_checkpointing(config, tmp_path):
    manifest = Manifest(tmp_path / "manifest.db")
    orch = Orchestrator(config, manifest)
    
    # Assert missing checkpoint
    assert orch._is_stage_complete("s05_recaption", "chunk_0001") is False
    
    # Mark stage as complete
    orch._mark_stage_complete("s05_recaption", "chunk_0001")
    
    # Verify checkpoint detection
    assert orch._is_stage_complete("s05_recaption", "chunk_0001") is True


class TestEscalationRunsBeforeRecaptionAndStructure:
    """Regression test for a real, previously-undetected bug: escalation
    (s04_5_escalation) used to run AFTER s05_recaption/s06_structure in
    run_pipeline's execution order, despite escalation being the only
    thing that can flip a borderline record's safety_tier to "safe".
    Any record Tier-2 rescued was permanently stuck at status=
    "safety_classified" — never recaptioned, structured, routed, or
    encoded — a silent, no-error-raised loss of every record the
    two-tier escalation system successfully rescues. Fixed by moving
    escalation ahead of recaption/structure in run_pipeline. See
    docs/review/16_preprocessing_ordering_audit.md.

    This test doesn't re-run real VLM inference (no torch needed) — it
    verifies the actual *order* run_pipeline invokes stages in, which is
    exactly what the bug was."""

    @pytest.mark.asyncio
    async def test_stage_call_order_puts_escalation_before_recaption(
        self, tmp_path, monkeypatch
    ):
        import contextlib

        from data_forge.config import PipelineConfig, PathsConfig, StageConfig

        config = PipelineConfig()
        config.data_root = tmp_path
        config.paths = PathsConfig(checkpoints="checkpoints")
        config.chunk_size = 10
        config.checkpoint_enabled = False
        for stage_name in [
            "s02_dedup", "s03_quality", "s03_5_pii_scrub", "s04_safety",
            "s04_5_escalation", "s05_recaption", "s06_structure",
            "s05_5_pii_text_redact", "s07_routing", "s08_encoding",
            "s08_5_dpo_encoding", "s09_heldout", "s01_6_preference_pairs",
        ]:
            config.stages[stage_name] = StageConfig(enabled=True)

        manifest = Manifest(tmp_path / "manifest.db")
        rec = manifest.create_record(source_dataset="test", image_path="x.png")
        # A borderline record is the exact scenario that exposed the bug —
        # escalation only has something to do (and only opens its vLLM
        # session) when at least one borderline/pending-review record
        # exists.
        manifest.update_record(rec.id, "test_setup", safety_tier="borderline")

        orch = Orchestrator(config, manifest)

        call_order: list[str] = []

        async def fake_run_stage(stage_name, record_ids, chunk_id, engine=None):
            call_order.append(stage_name)
            from data_forge.orchestrator import StageResult
            return StageResult(stage_name=stage_name)

        monkeypatch.setattr(orch, "_run_stage", fake_run_stage)
        monkeypatch.setattr(orch, "_get_borderline_ids", lambda record_ids: [rec.id])
        # Never actually excludes anything in this test — keeps every
        # phase's record_ids non-empty so every phase's guard condition
        # (`if record_ids:`) is exercised, not skipped.
        monkeypatch.setattr(orch, "_filter_active", lambda record_ids: record_ids)

        @contextlib.asynccontextmanager
        async def fake_session(*a, **k):
            yield None

        import data_forge.inference.engine as engine_module
        monkeypatch.setattr(engine_module.ModelEngine, "vllm_session", staticmethod(fake_session))
        monkeypatch.setattr(engine_module.ModelEngine, "clip_session", staticmethod(fake_session))
        monkeypatch.setattr(engine_module.ModelEngine, "encoder_session", staticmethod(fake_session))

        await orch.execute_pipeline()

        assert "s04_5_escalation" in call_order
        assert "s05_recaption" in call_order
        assert "s06_structure" in call_order
        escalation_idx = call_order.index("s04_5_escalation")
        recaption_idx = call_order.index("s05_recaption")
        structure_idx = call_order.index("s06_structure")
        assert escalation_idx < recaption_idx, (
            f"s04_5_escalation ran at position {escalation_idx}, "
            f"s05_recaption at {recaption_idx} — escalation must resolve "
            f"borderline records BEFORE recaption's safety_tier=='safe' "
            f"filter runs, or rescued records are silently dropped."
        )
        assert escalation_idx < structure_idx
        # s04_safety must still precede escalation (escalation reads the
        # safety_tier s04_safety sets).
        assert call_order.index("s04_safety") < escalation_idx

