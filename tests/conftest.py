"""Root-level pytest configuration for the Krivex monorepo.
Ensures tests run in hermetic in-memory databases and isolated storage environments,
preventing stray SQLite databases, data folders, or blob folders from polluting the workspace root.
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("KRIVEX_SESSIONS_DB", ":memory:")
os.environ.setdefault("KRIVEX_PREFERENCE_DB", ":memory:")
_test_blob_dir = os.path.join(tempfile.gettempdir(), "krivex_test_blobs")
os.environ.setdefault("KRIVEX_BLOB_ROOT", _test_blob_dir)
_test_data_dir = os.path.join(tempfile.gettempdir(), "krivex_test_data")
os.environ.setdefault("DATA_ROOT", _test_data_dir)
os.environ.setdefault("KRIVEX_POLISH_SYNTHETIC", "1")
