"""Backward-compatibility wrapper.

Redirects to `krivex_training.preference.uicrit_importer`.
"""

from __future__ import annotations

from krivex_training.preference.uicrit_importer import import_records

__all__ = ["import_records"]
