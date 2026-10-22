"""Backward-compatibility wrapper.

Redirects to `krivex_training.preference.export`.
"""

from __future__ import annotations

from krivex_training.preference.export import export_jsonl

__all__ = ["export_jsonl"]
