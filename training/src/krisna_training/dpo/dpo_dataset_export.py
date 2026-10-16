"""Backward-compatibility wrapper.

Redirects to `krisna_training.preference.export`.
"""

from __future__ import annotations

from krisna_training.preference.export import export_jsonl

__all__ = ["export_jsonl"]
