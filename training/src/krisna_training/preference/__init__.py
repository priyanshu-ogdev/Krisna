"""Preference-pair store and dataset utilities for Diffusion-DPO alignment.

Provides SQLite storage, candidate ranking and pair generation,
export to standard JSONL format, and UICrit seed data parsing.
"""

from __future__ import annotations

from krisna_training.preference.export import export_jsonl
from krisna_training.preference.pair_builder import (
    DEFAULT_MIN_SCORE_GAP,
    aggregate_verifier_score,
    build_pair_from_candidates,
    rank_candidates,
)
from krisna_training.preference.preference_store import (
    VALID_SOURCES,
    PreferencePair,
    PreferenceStore,
)
from krisna_training.preference.uicrit_importer import import_records

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
