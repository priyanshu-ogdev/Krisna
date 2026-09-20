from __future__ import annotations

import os

from krisna_inference.backends.common import BlobStore

_instance: BlobStore | None = None


def get_blob_store() -> BlobStore:
    global _instance
    if _instance is None:
        root = os.environ.get("KRISNA_BLOB_ROOT", "krisna_blobs")
        _instance = BlobStore(root=root)
    return _instance
