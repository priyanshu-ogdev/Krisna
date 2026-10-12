from __future__ import annotations

import pytest
from PIL import Image

from krisna_inference.backends.polish_backend_qwen import QwenImage21Backend
from krisna_inference.orchestrator.model_registry import ModelSpec, Tier
from krisna_inference.orchestrator.store import DesignStateStore
from krisna_inference.verifiers.critique_adapter import from_planner_review
from krisna_inference.verifiers.layout_iou import LayoutIoUVerifier


def test_locked_regions_bbox_handling_in_polish_backend():
    spec = ModelSpec(tier=Tier.POLISH_DEFAULT, name="test", vram_gb=12.0, quantization="bf16")
    backend = QwenImage21Backend(spec)

    src = Image.new("RGB", (100, 100), color=(255, 0, 0))
    dst = Image.new("RGB", (100, 100), color=(0, 255, 0))

    # Test with standard LockedRegion schema: {"bbox": (0.2, 0.2, 0.6, 0.6), "reason": "preserve"}
    locked_bbox = [{"bbox": (0.2, 0.2, 0.6, 0.6), "reason": "preserve"}]
    blended = backend._apply_locked_regions(src, dst, locked_bbox)
    assert blended.size == (100, 100)
    # Center pixel (50, 50) should be preserved from src (red)
    r, g, b = blended.getpixel((50, 50))
    assert r > 200
    assert g < 50

    # Test with flat x, y, w, h format
    locked_flat = [{"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6}]
    blended_flat = backend._apply_locked_regions(src, dst, locked_flat)
    r2, g2, b2 = blended_flat.getpixel((50, 50))
    assert r2 > 200
    assert g2 < 50


def test_layout_iou_verifier_handles_both_bbox_and_flat_formats():
    verifier = LayoutIoUVerifier(detector=lambda img: [[0.1, 0.1, 0.2, 0.2]])

    # Format 1: {"bbox": [0.1, 0.1, 0.2, 0.2]}
    s1 = verifier.score(image=None, expected_regions=[{"bbox": [0.1, 0.1, 0.2, 0.2]}])
    assert s1 == pytest.approx(1.0)

    # Format 2: {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    s2 = verifier.score(image=None, expected_regions=[{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}])
    assert s2 == pytest.approx(1.0)


def test_from_planner_review_extracts_edits_and_verifier_dimensions():
    planner_output = {
        "reply_text": "Consider improving hero hierarchy and button sizing.",
        "design_state_delta": {
            "reasoning_note": "Hero section lacks contrast against dark backdrop.",
            "constraint_updates": {
                "style": "high_contrast_dark",
            },
        },
        "retrieved_critique_ids": ["crit_123"],
    }
    verifier_scores = {
        "clip_alignment": 0.85,
        "ocr_readability": 0.92,
        "layout_iou": 0.88,
        "aesthetic": 0.90,
        "handoff_consistency": 0.95,
    }

    critique = from_planner_review(planner_output, verifier_scores=verifier_scores)
    assert critique.critique_source == "planner_review"
    assert critique.overall_score > 0.8
    assert "visual_hierarchy" in critique.dimensions
    assert "readability" in critique.dimensions
    assert len(critique.suggested_edits) >= 2
    instructions = [e["instruction"] for e in critique.suggested_edits]
    assert any("Hero section lacks contrast" in inst for inst in instructions)
    assert any("hero hierarchy" in inst for inst in instructions)


def test_store_wal_mode_and_concurrency(tmp_path):
    db_file = tmp_path / "test_sessions.db"
    store = DesignStateStore(db_path=db_file)
    try:
        # Check that PRAGMA journal_mode is wal
        cursor = store._conn.cursor()
        mode = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        store.close()
