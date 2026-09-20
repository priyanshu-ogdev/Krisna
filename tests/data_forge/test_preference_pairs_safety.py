"""Tests for s01_6_preference_pairs.py, specifically the safety-
classification bug fix.

Regression coverage: an earlier revision of this stage ran with
engine=None and never classified preference-pair images for NSFW/harmful
content at all, despite these being T2I-model outputs the rest of the
pipeline never screens. These tests confirm: (1) the stage refuses to run
without an engine rather than silently skipping safety checks, (2) a pair
with either image classified "unsafe" is dropped and never marked
dedup_status: "unique" (so s08_5_dpo_encoding's skip-if-not-unique guard
also protects it), (3) "borderline" pairs are kept but flagged, (4) safe
pairs pass through cleanly, and (5) cross-source exact-duplicate pairs are
still caught alongside the new safety logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from data_forge.config import DatasetSpec, PipelineConfig
from data_forge.stages.s01_6_preference_pairs import PreferencePairsStage


@dataclass
class _FakeSafety:
    tier: str
    confidence: float = 0.95


def _write_pair(pref_root: Path, source: str, pair_id: str, color_a=(10, 20, 30), color_b=(200, 210, 220)) -> None:
    out_dir = pref_root / source
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color_a).save(out_dir / f"{pair_id}_a.png")
    Image.new("RGB", (64, 64), color=color_b).save(out_dir / f"{pair_id}_b.png")
    (out_dir / f"{pair_id}.json").write_text(json.dumps({
        "pair_id": pair_id,
        "prompt": "a test prompt",
        "image_a": f"{pair_id}_a.png",
        "image_b": f"{pair_id}_b.png",
        "preferred": "a",
        "origin": source,
        "label_source": "human",
    }), encoding="utf-8")


def _configure_preference_source(config: PipelineConfig, key: str) -> None:
    config.datasets[key] = DatasetSpec(
        display_name=key,
        source_type="huggingface",
        category="dpo_preference_general",
        expected_record_count=100,
        fetch_config={"download_mode": "preference_pair"},
    )
    config.stages = {"s01_6_preference_pairs": {}}


@pytest.fixture(autouse=True)
def _stub_face_blur():
    """No mediapipe in the test environment — stub face detection out
    entirely (returns the image unmodified, no faces found), same
    behavior as a real run with mediapipe unavailable."""
    with patch("data_forge.stages.s01_6_preference_pairs.load_face_detector", return_value=None):
        with patch(
            "data_forge.stages.s01_6_preference_pairs.blur_faces",
            side_effect=lambda img, detector, kernel: (img, False, []),
        ):
            yield


class TestPreferencePairsSafetyGate:
    async def test_refuses_to_run_without_engine(self, config, manifest):
        """The core regression guard: no engine means no safety
        classification, which must mean no processing at all — not a
        silent skip that lets unscreened images through."""
        pref_root = config.resolved_paths["preference_pairs"]
        _configure_preference_source(config, "pickapic_v2")
        _write_pair(pref_root, "pickapic_v2", "p0000001")

        stage = PreferencePairsStage()
        result = await stage.run(manifest, config, [], engine=None)

        assert result.records_processed == 0
        meta_path = pref_root / "pickapic_v2" / "p0000001.json"
        meta = json.loads(meta_path.read_text())
        assert "dedup_status" not in meta, "must not be marked processed without a safety pass"

    async def test_unsafe_pair_is_dropped(self, config, manifest):
        pref_root = config.resolved_paths["preference_pairs"]
        _configure_preference_source(config, "pickapic_v2")
        _write_pair(pref_root, "pickapic_v2", "p0000002")

        with patch("data_forge.stages.s01_6_preference_pairs.Tier1Engine") as MockTier1:
            instance = MockTier1.return_value
            instance.batch_classify_safety = AsyncMock(
                return_value=[_FakeSafety("unsafe"), _FakeSafety("safe")]
            )
            stage = PreferencePairsStage()
            result = await stage.run(manifest, config, [], engine=object())

        assert result.records_processed == 0
        assert result.metadata["dropped_unsafe"] == 1
        meta = json.loads((pref_root / "pickapic_v2" / "p0000002.json").read_text())
        assert "dedup_status" not in meta

    async def test_borderline_pair_is_kept_and_flagged(self, config, manifest):
        pref_root = config.resolved_paths["preference_pairs"]
        _configure_preference_source(config, "pickapic_v2")
        _write_pair(pref_root, "pickapic_v2", "p0000003")

        with patch("data_forge.stages.s01_6_preference_pairs.Tier1Engine") as MockTier1:
            instance = MockTier1.return_value
            instance.batch_classify_safety = AsyncMock(
                return_value=[_FakeSafety("borderline"), _FakeSafety("safe")]
            )
            stage = PreferencePairsStage()
            result = await stage.run(manifest, config, [], engine=object())

        assert result.records_processed == 1
        assert result.metadata["flagged_borderline"] == 1
        meta = json.loads((pref_root / "pickapic_v2" / "p0000003.json").read_text())
        assert meta["dedup_status"] == "unique"
        assert meta["safety_tier"] == "borderline"

    async def test_safe_pair_kept_low_confidence_flagged_borderline(self, config, manifest):
        """Confidence below threshold is treated the same as an explicit
        "borderline" tier — a low-confidence "safe" call shouldn't be
        trusted any more than an explicit borderline classification."""
        pref_root = config.resolved_paths["preference_pairs"]
        _configure_preference_source(config, "pickapic_v2")
        _write_pair(pref_root, "pickapic_v2", "p0000004")

        with patch("data_forge.stages.s01_6_preference_pairs.Tier1Engine") as MockTier1:
            instance = MockTier1.return_value
            instance.batch_classify_safety = AsyncMock(
                return_value=[_FakeSafety("safe", confidence=0.5), _FakeSafety("safe", confidence=0.99)]
            )
            stage = PreferencePairsStage()
            result = await stage.run(manifest, config, [], engine=object())

        assert result.records_processed == 1
        meta = json.loads((pref_root / "pickapic_v2" / "p0000004.json").read_text())
        assert meta["safety_tier"] == "borderline"

    async def test_clean_safe_pair_passes_through(self, config, manifest):
        pref_root = config.resolved_paths["preference_pairs"]
        _configure_preference_source(config, "pickapic_v2")
        _write_pair(pref_root, "pickapic_v2", "p0000005")

        with patch("data_forge.stages.s01_6_preference_pairs.Tier1Engine") as MockTier1:
            instance = MockTier1.return_value
            instance.batch_classify_safety = AsyncMock(
                return_value=[_FakeSafety("safe"), _FakeSafety("safe")]
            )
            stage = PreferencePairsStage()
            result = await stage.run(manifest, config, [], engine=object())

        assert result.records_processed == 1
        meta = json.loads((pref_root / "pickapic_v2" / "p0000005.json").read_text())
        assert meta["dedup_status"] == "unique"
        assert meta["safety_tier"] == "safe"

    async def test_cross_source_duplicate_still_caught_alongside_safety(self, config, manifest):
        """The pre-existing dedup behavior must survive the safety-check
        addition unchanged — same pixel content in two different sources
        (pickapic_v2, hpdv2) should only be kept once."""
        pref_root = config.resolved_paths["preference_pairs"]
        _configure_preference_source(config, "pickapic_v2")
        config.datasets["hpdv2"] = DatasetSpec(
            display_name="hpdv2", source_type="huggingface",
            category="dpo_preference_general", expected_record_count=100,
            fetch_config={"download_mode": "preference_pair"},
        )
        _write_pair(pref_root, "pickapic_v2", "p0000006", color_a=(1, 2, 3), color_b=(4, 5, 6))
        _write_pair(pref_root, "hpdv2", "h0000006", color_a=(1, 2, 3), color_b=(4, 5, 6))

        with patch("data_forge.stages.s01_6_preference_pairs.Tier1Engine") as MockTier1:
            instance = MockTier1.return_value
            instance.batch_classify_safety = AsyncMock(
                return_value=[_FakeSafety("safe"), _FakeSafety("safe")]
            )
            stage = PreferencePairsStage()
            result = await stage.run(manifest, config, [], engine=object())

        assert result.records_processed == 1
        assert result.metadata["dropped_duplicate"] == 1

    async def test_eval_only_source_is_skipped_even_if_misplaced(self, config, manifest):
        """Belt-and-suspenders guard: an eval_only-flagged source
        physically present under preference_pairs/ (shouldn't happen —
        see fetcher.py) must still never be processed as training data."""
        pref_root = config.resolved_paths["preference_pairs"]
        config.datasets["taste"] = DatasetSpec(
            display_name="taste", source_type="huggingface",
            category="eval_design_multiaxis", expected_record_count=100,
            eval_only=True,
            fetch_config={"download_mode": "preference_pair"},  # misconfigured on purpose
        )
        config.stages = {"s01_6_preference_pairs": {}}
        _write_pair(pref_root, "taste", "t0000001")

        with patch("data_forge.stages.s01_6_preference_pairs.Tier1Engine") as MockTier1:
            instance = MockTier1.return_value
            instance.batch_classify_safety = AsyncMock(return_value=[_FakeSafety("safe"), _FakeSafety("safe")])
            stage = PreferencePairsStage()
            result = await stage.run(manifest, config, [], engine=object())

        assert result.records_processed == 0
        meta = json.loads((pref_root / "taste" / "t0000001.json").read_text())
        assert "dedup_status" not in meta
