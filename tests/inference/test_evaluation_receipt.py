from __future__ import annotations

from krisna_inference.verifiers.verifier_stack import VerifierStack


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
