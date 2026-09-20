"""DPO preference-pair pipeline and database layer.

Re-exports core classes and utilities from `krisna_training.preference`
for backwards compatibility.
"""

from __future__ import annotations

from krisna_training.preference import (
    DEFAULT_MIN_SCORE_GAP,
    VALID_SOURCES,
    PreferencePair,
    PreferenceStore,
    aggregate_verifier_score,
    build_pair_from_candidates,
    export_jsonl,
    import_records,
    rank_candidates,
)

__all__ = [
    "PreferencePair",
    "PreferenceStore",
    "VALID_SOURCES",
    "rank_candidates",
    "aggregate_verifier_score",
    "build_pair_from_candidates",
    "DEFAULT_MIN_SCORE_GAP",
    "import_records",
    "export_jsonl",
]
