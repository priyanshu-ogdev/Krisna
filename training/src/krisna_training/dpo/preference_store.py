"""Backward-compatibility wrapper.

Redirects to `krisna_training.preference.preference_store`.
"""

from __future__ import annotations

from krisna_training.preference.preference_store import (
    SCHEMA,
    VALID_SOURCES,
    PreferencePair,
    PreferenceStore,
)

__all__ = ["SCHEMA", "VALID_SOURCES", "PreferencePair", "PreferenceStore"]
