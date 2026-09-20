from __future__ import annotations

import pytest

from krisna_inference.verifiers.layout_iou import LayoutIoUVerifier, _iou


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
