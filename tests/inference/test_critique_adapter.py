from __future__ import annotations

import pytest

from krisna_inference.orchestrator.design_state import CritiqueResult
from krisna_inference.verifiers.critique_adapter import from_gemma_output, from_verifier_scores


def test_from_gemma_output_normalizes_full_payload():
    raw = {
        "critique_source": "gemma4_31b_frozen",
        "overall_score": 0.82,
        "dimensions": {"visual_hierarchy": {"score": 0.8, "note": "Clear."}},
        "suggested_edits": [{"region": [0, 0, 0.5, 0.5], "instruction": "Enlarge CTA."}],
        "raw_model_output_ref": None,
    }
    result = from_gemma_output(raw)
    assert isinstance(result, CritiqueResult)
    assert result.critique_source == "gemma4_31b_frozen"
    assert result.overall_score == 0.82
    assert result.dimensions["visual_hierarchy"].score == 0.8
    assert result.suggested_edits[0]["instruction"] == "Enlarge CTA."


def test_from_gemma_output_fills_defaults_on_partial_payload():
    result = from_gemma_output({})
    assert result.critique_source == "gemma4_31b_frozen"
    assert result.overall_score == 0.0
    assert result.dimensions == {}
    assert result.suggested_edits == []


def test_from_verifier_scores_maps_known_dimensions():
    scores = {
        "clip_alignment": 0.7,
        "ocr_readability": 0.9,
        "layout_iou": 0.6,
        "aesthetic": 0.5,
        "handoff_consistency": 0.8,
    }
    result = from_verifier_scores(scores)
    assert result.critique_source == "verifier_stack"
    assert result.dimensions["readability"].score == 0.9
    assert result.dimensions["layout_consistency"].score == 0.6
    assert result.dimensions["visual_hierarchy"].score == 0.5
    assert result.dimensions["brand_alignment"].score == 0.7
    assert result.overall_score == pytest.approx(
        (scores["ocr_readability"] + scores["layout_iou"] + scores["aesthetic"] + scores["clip_alignment"]) / 4
    )
    # Note: handoff_consistency has no corresponding critique dimension (only
    # 4 dimensions exist per §5.2, vs. 5 verifier scores) — it's intentionally
    # excluded from this average, not a bug in the test above.
    assert result.suggested_edits == []


def test_from_verifier_scores_skips_none_values():
    scores = {
        "clip_alignment": 0.7,
        "ocr_readability": None,  # this verifier failed
        "layout_iou": 0.6,
        "aesthetic": None,
        "handoff_consistency": None,
    }
    result = from_verifier_scores(scores)
    assert "readability" not in result.dimensions
    assert "brand_alignment" in result.dimensions
    assert result.overall_score == (0.7 + 0.6) / 2


def test_from_verifier_scores_all_none_gives_zero_overall():
    scores = {k: None for k in ["clip_alignment", "ocr_readability", "layout_iou", "aesthetic", "handoff_consistency"]}
    result = from_verifier_scores(scores)
    assert result.dimensions == {}
    assert result.overall_score == 0.0
