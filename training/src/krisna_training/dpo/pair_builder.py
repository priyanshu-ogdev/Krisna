"""Backward-compatibility wrapper.

Redirects to `krisna_training.preference.pair_builder`.
"""

from __future__ import annotations

from krisna_training.preference.pair_builder import (
    DEFAULT_MIN_SCORE_GAP,
    aggregate_verifier_score,
    build_pair_from_candidates,
    rank_candidates,
)

__all__ = [
    "DEFAULT_MIN_SCORE_GAP",
    "aggregate_verifier_score",
    "build_pair_from_candidates",
    "rank_candidates",
]
