"""Critique Adapter — §5.2's actual job, formalized: normalize whichever
source produced a critique (the verifier stack's own aesthetic/alignment
scoring, or Gemma 4's judgment) into the exact same
design_state.CritiqueResult shape, so anything downstream (DPO ranking,
the preference-pair store) never needs to know or care which one it was —
`critique_source` is the only distinguishing field, per §5.2's own text.

critic_worker.py (which runs in the isolated critic venv, see
critic_backend.py) does its own best-effort JSON shaping because it can't
import this module — that venv only has transformers/unsloth, not the rest
of krisna_inference (formerly krisna_orchestrator). `from_gemma_output()` here re-validates/normalizes
whatever comes back from that subprocess through the same pydantic model
every other source goes through, so a malformed or partial worker response
never reaches the DesignState store un-normalized.
"""

from __future__ import annotations

from krisna_inference.orchestrator.design_state import CritiqueDimensionScore, CritiqueResult


def from_gemma_output(raw: dict) -> CritiqueResult:
    """raw: the `critique_result` dict CriticBackend.run() returns (itself
    already close to this shape — see critic_worker.py's _run()). This
    re-validates it through the pydantic model rather than trusting the
    subprocess blindly."""
    return CritiqueResult.model_validate(
        {
            "critique_source": raw.get("critique_source", "gemma4_31b_frozen"),
            "overall_score": raw.get("overall_score", 0.0),
            "dimensions": raw.get("dimensions", {}),
            "suggested_edits": raw.get("suggested_edits", []),
            "raw_model_output_ref": raw.get("raw_model_output_ref"),
        }
    )


# Best-effort proxy mapping from verifier-stack dimensions onto the same
# four critique dimensions Gemma reports. These are NOT semantically
# identical (e.g. "aesthetic" is a rough stand-in for "visual_hierarchy",
# not a real hierarchy judgment) — documented here rather than presented
# as equivalent, but still genuinely useful as a cheap critique source that
# doesn't require swapping in the Critic tier at all.
_VERIFIER_TO_DIMENSION = {
    "readability": ("ocr_readability", "OCR-detected text legibility."),
    "layout_consistency": ("layout_iou", "Layout region overlap with expected constraints."),
    "visual_hierarchy": ("aesthetic", "Aesthetic-predictor score used as a rough proxy — not a real hierarchy judgment."),
    "brand_alignment": ("clip_alignment", "CLIP text-image alignment to the style/constraints prompt."),
}


def from_verifier_scores(scores: dict) -> CritiqueResult:
    """scores: the dict verifier_stack.VerifierStack.score_finalize_output()
    returns (clip_alignment, ocr_readability, layout_iou, aesthetic,
    handoff_consistency — any of which may be None if that verifier
    failed)."""
    dimensions: dict[str, CritiqueDimensionScore] = {}
    available = []
    for dim_name, (score_key, note) in _VERIFIER_TO_DIMENSION.items():
        value = scores.get(score_key)
        if value is not None:
            dimensions[dim_name] = CritiqueDimensionScore(score=value, note=note)
            available.append(value)

    overall = sum(available) / len(available) if available else 0.0

    return CritiqueResult(
        critique_source="verifier_stack",
        overall_score=overall,
        dimensions=dimensions,
        suggested_edits=[],  # verifier stack scores, it doesn't propose edits
        raw_model_output_ref=None,
    )
