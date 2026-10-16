"""Backward-compatibility wrapper.

Redirects to `krisna_training.preference.preference_store`.
Enforces single canonical PreferenceStore implementation with idempotent
INSERT OR IGNORE and PRD §6 no-AI-judge constraints.
"""

from __future__ import annotations

from krisna_training.preference.preference_store import (
    SCHEMA,
    VALID_SOURCES,
    PreferencePair,
    PreferenceStore,
)

__all__ = [
    "SCHEMA",
    "VALID_SOURCES",
    "PreferencePair",
    "PreferenceStore",
]
