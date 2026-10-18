"""Stable source-identity keys shared by split carving and export checks."""

from __future__ import annotations

from data_forge.manifest import ManifestRecord


def source_split_identity(record: ManifestRecord) -> str:
    """Return a conservative group key for train/held-out separation.

    Dataset-level separation remains the fallback. Adapter-backed sources add
    an upstream app/file group, which prevents screens from the same app or
    Figma file crossing the split while still allowing unrelated sources to be
    combined in one manifest and one export.
    """
    dataset = str(record.source_dataset or "unknown")
    group = str(record.source_group_id or "").strip()
    return f"{dataset}:{group}" if group else f"{dataset}:__dataset__"
