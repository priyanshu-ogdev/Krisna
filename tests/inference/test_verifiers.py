from __future__ import annotations

import pytest
from PIL import Image

from krivex_inference.verifiers.layout_iou import LayoutIoUVerifier, _iou
from krivex_inference.verifiers.lock_preservation import LockPreservationVerifier
from krivex_inference.verifiers.verifier_stack import VerifierStack


# ---------------------------------------------------------------------------
# Layout IoU Verifier Tests
# ---------------------------------------------------------------------------

def test_iou_identical_boxes():
    box = [0.1, 0.1, 0.3, 0.2]
    assert _iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    a = [0.0, 0.0, 0.1, 0.1]
    b = [0.5, 0.5, 0.1, 0.1]
    assert _iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = [0.0, 0.0, 0.2, 0.2]
    b = [0.1, 0.1, 0.2, 0.2]
    iou = _iou(a, b)
    assert 0.0 < iou < 1.0


def test_score_no_expected_regions_returns_1():
    verifier = LayoutIoUVerifier(detector=lambda img: [[0.1, 0.1, 0.2, 0.2]])
    assert verifier.score(image=None, expected_regions=[]) == 1.0


def test_score_no_detections_returns_0():
    verifier = LayoutIoUVerifier(detector=lambda img: [])
    score = verifier.score(image=None, expected_regions=[{"bbox": [0.1, 0.1, 0.2, 0.2]}])
    assert score == 0.0


def test_score_uses_best_match_per_expected_region():
    detected = [[0.0, 0.0, 0.05, 0.05], [0.1, 0.1, 0.2, 0.2]]  # second is a perfect match
    verifier = LayoutIoUVerifier(detector=lambda img: detected)
    score = verifier.score(image=None, expected_regions=[{"bbox": [0.1, 0.1, 0.2, 0.2]}])
    assert score == pytest.approx(1.0)


def test_score_averages_across_multiple_expected_regions():
    detected = [[0.1, 0.1, 0.2, 0.2]]  # matches only the first expected region
    verifier = LayoutIoUVerifier(detector=lambda img: detected)
    score = verifier.score(
        image=None,
        expected_regions=[{"bbox": [0.1, 0.1, 0.2, 0.2]}, {"bbox": [0.7, 0.7, 0.1, 0.1]}],
    )
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Lock Preservation Verifier Tests
# ---------------------------------------------------------------------------

def test_lock_preservation_is_exact_for_unchanged_region():
    source = Image.new("RGB", (20, 20), (20, 20, 20))
    rendered = source.copy()
    score = LockPreservationVerifier().score(
        source,
        rendered,
        [{"bbox": [0.0, 0.0, 0.5, 0.5]}],
    )
    assert score == 1.0


def test_lock_preservation_penalizes_changed_pixels():
    source = Image.new("RGB", (20, 20), (0, 0, 0))
    rendered = Image.new("RGB", (20, 20), (255, 255, 255))
    score = LockPreservationVerifier().score(
        source,
        rendered,
        [{"bbox": [0.0, 0.0, 0.5, 0.5]}],
    )
    assert score == 0.0


# ---------------------------------------------------------------------------
# Evaluation Receipt / Verifier Stack Tests
# ---------------------------------------------------------------------------

def test_evaluation_receipt_combines_metrics_and_reports_missing_dimensions():
    stack = VerifierStack.__new__(VerifierStack)
    stack.score_finalize_output = lambda *args, **kwargs: {
        "clip_alignment": 0.82,
        "ocr_readability": 0.91,
        "layout_iou": 0.74,
        "aesthetic": 0.68,
        "handoff_consistency": None,
        "lock_preservation": None,
        "seam_continuity": None,
    }

    receipt = stack.evaluate_finalize_output(object(), prompt="ui", safety_score=0.96)

    assert receipt["policy"] == "krisna.eval.v1"
    assert receipt["status"] == "incomplete"
    assert receipt["missing_metrics"] == ["handoff_consistency", "lock_preservation", "seam_continuity"]
    assert receipt["failed_metrics"] == ["aesthetic"]
    assert receipt["safety_gate_passed"] is True
    assert 0.0 < receipt["aggregate_geometric_mean"] < 1.0


def test_evaluation_receipt_passes_only_when_all_targets_are_available():
    stack = VerifierStack.__new__(VerifierStack)
    stack.score_finalize_output = lambda *args, **kwargs: {
        "clip_alignment": 0.80,
        "ocr_readability": 0.90,
        "layout_iou": 0.75,
        "aesthetic": 0.72,
        "handoff_consistency": 0.70,
        "lock_preservation": 0.98,
        "seam_continuity": 0.90,
    }

    receipt = stack.evaluate_finalize_output(object(), prompt="ui", safety_score=0.95)

    assert receipt["status"] == "pass"
    assert receipt["missing_metrics"] == []
    assert receipt["failed_metrics"] == []
