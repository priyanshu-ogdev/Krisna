"""Builds (chosen, rejected) preference pairs from scored candidates.

A "candidate" here is any dict shaped {"image_ref": str, "score": float}.
"""

from __future__ import annotations

from krisna_training.preference.preference_store import PreferencePair, PreferenceStore

DEFAULT_MIN_SCORE_GAP = 0.1  # don't pair near-ties — that's noise, not signal


def aggregate_verifier_score(verifier_scores: dict) -> float | None:
    """Mean of whatever verifier scores are non-None. Returns None if the
    whole stack failed (nothing usable to rank on)."""
    values = [v for v in verifier_scores.values() if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def rank_candidates(candidates: list[dict]) -> list[dict]:
    """Highest score first. Candidates with score=None sort last."""
    return sorted(candidates, key=lambda c: (c["score"] is None, -(c["score"] or 0.0)))


def build_pair_from_candidates(
    store: PreferenceStore,
    prompt: str,
    candidates: list[dict],
    source: str,
    session_id: str | None = None,
    min_score_gap: float = DEFAULT_MIN_SCORE_GAP,
) -> PreferencePair | None:
    """candidates: [{"image_ref": str, "score": float | None}, ...] — at
    least 2 needed. Picks the highest- and lowest-scored as chosen/rejected.
    Returns None (no pair written) if fewer than 2 scored candidates exist,
    or if the score gap between best and worst is too small to be a
    meaningful preference signal.
    """
    scored = [c for c in candidates if c["score"] is not None]
    if len(scored) < 2:
        return None

    ranked = rank_candidates(scored)
    best, worst = ranked[0], ranked[-1]
    if best["image_ref"] == worst["image_ref"]:
        return None
    if (best["score"] - worst["score"]) < min_score_gap:
        return None

    pair = PreferencePair(
        prompt=prompt,
        chosen_ref=best["image_ref"],
        rejected_ref=worst["image_ref"],
        chosen_score=best["score"],
        rejected_score=worst["score"],
        source=source,
        session_id=session_id,
    )
    return store.add(pair)
