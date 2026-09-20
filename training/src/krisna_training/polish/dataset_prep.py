"""Backward-compatibility wrapper.

Redirects to `krisna_training.polish.prepare_dataset`.
"""

from __future__ import annotations

from krisna_training.polish.prepare_dataset import (
    IMAGE_EXTENSIONS,
    count_prepared,
    load_captions,
    prepare,
    prepare_dataset,
)

__all__ = ["prepare", "prepare_dataset", "count_prepared", "load_captions", "IMAGE_EXTENSIONS"]
