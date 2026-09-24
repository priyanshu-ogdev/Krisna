"""Builds (chosen, rejected) preference pairs from scored candidates.

A "candidate" here is any dict shaped {"image_ref": str, "score": float},
optionally with a "prompt_used" key (see build_pair_from_candidates).
"""

from __future__ import annotations

from krisna_training.preference.preference_store import PreferencePair, PreferenceStore

DEFAULT_MIN_SCORE_GAP = 0.1  # don't pair near-ties — that's noise, not signal


class MismatchedPromptError(ValueError):
    """UPGRADE (training-data-integrity review): raised when two
    candidates being paired carry different `prompt_used` values. See
    build_pair_from_candidates's docstring for why this must be a raised
    error, not a warning."""


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
    """candidates: [{"image_ref": str, "score": float | None,
    "prompt_used": str | None}, ...] — at least 2 needed. Picks the
    highest- and lowest-scored as chosen/rejected. Returns None (no pair
    written) if fewer than 2 scored candidates exist, or if the score gap
    between best and worst is too small to be a meaningful preference
    signal.

    UPGRADE (training-data-integrity review — closes the caller-contract
    gap flows.py's critique_pass() documents but couldn't enforce on its
    own): Diffusion-DPO's derivation assumes both candidates in a pair
    are conditioned on the SAME prompt/design intent. This project's own
    (prompt, image) mixup bug (see design_state.py's `prompt_used`
    field, flows.py's finalize()/critique_pass()) was exactly a case
    where that assumption silently broke. A caller can now pass an
    optional per-candidate `"prompt_used"` key — the prompt that
    candidate's OWN image was actually generated from, when known. If
    two or more candidates supply one and they disagree with each other
    (or with the `prompt` argument itself), this raises
    MismatchedPromptError rather than silently writing a pair whose
    stored prompt is wrong for one of its two images — the same failure
    mode the prompt_used fix closes on the finalize()/critique_pass()
    side, now also enforced here for any candidate that opts in to
    reporting it. A candidate that omits "prompt_used" entirely (older
    callers, or a caller that genuinely doesn't track it) is not
    checked — this is additive validation, not a new required field.
    """
    scored = [c for c in candidates if c["score"] is not None]
    if len(scored) < 2:
        return None

    known_prompts = {prompt} | {
        c["prompt_used"] for c in candidates if c.get("prompt_used")
    }
    if len(known_prompts) > 1:
        raise MismatchedPromptError(
            "build_pair_from_candidates: candidates report different "
            f"prompt_used values ({sorted(known_prompts)!r}) than each other "
            "and/or the `prompt` argument. Refusing to write a preference "
            "pair with an ambiguous/wrong prompt for one of its two images — "
            "this is exactly the (prompt, image) mixup class of bug this "
            "check exists to catch. Verify the caller is comparing two "
            "candidates from the SAME design intent."
        )

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
