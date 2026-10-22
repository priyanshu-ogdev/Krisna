from __future__ import annotations

import os

from krivex_inference.backends.common import BlobStore

_instance: BlobStore | None = None


def get_blob_store() -> BlobStore:
    global _instance
    if _instance is None:
        root = os.environ.get("KRISNA_BLOB_ROOT", "krisna_blobs")
        _instance = BlobStore(root=root)
    return _instance


def set_blob_store(store: BlobStore | None) -> None:
    global _instance
    _instance = store
