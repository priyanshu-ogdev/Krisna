"""SQLite-backed persistence for DesignState objects.

One row per session, full JSON blob + a bare `revision` column for cheap
optimistic-concurrency checks without deserializing on every write.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from krisna_inference.orchestrator.design_state import DesignState
from krisna_inference.orchestrator.exceptions import SessionNotFoundError, StaleDesignStateError

SCHEMA = """
CREATE TABLE IF NOT EXISTS design_state (
    session_id TEXT PRIMARY KEY,
    revision   INTEGER NOT NULL,
    stage      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_design_state_stage ON design_state(stage);
"""


class DesignStateStore:
    """Thread-safe (single-process) SQLite store. One connection, guarded
    by a lock — this is a local single-GPU service, not a distributed
    system, so this is deliberately the simplest thing that's actually
    correct rather than a connection pool."""

    def __init__(self, db_path: str | Path = "krisna_sessions.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def create(self, state: DesignState | None = None) -> DesignState:
        state = state or DesignState()
        with self._lock:
            self._conn.execute(
                "INSERT INTO design_state (session_id, revision, stage, updated_at, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    state.session_id,
                    state.revision,
                    state.stage.value,
                    state.updated_at,
                    state.model_dump_json(),
                ),
            )
            self._conn.commit()
        return state

    def get(self, session_id: str) -> DesignState:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM design_state WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        return DesignState.model_validate_json(row["payload_json"])

    def save(self, state: DesignState, expected_revision: int) -> DesignState:
        """Optimistic-concurrency write. `state.revision` should already
        have been bumped (via state.touch()) past `expected_revision`
        before calling this — expected_revision is what the caller
        originally read, used to detect a concurrent writer."""
        with self._lock:
            row = self._conn.execute(
                "SELECT revision FROM design_state WHERE session_id = ?",
                (state.session_id,),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(state.session_id)
            if row["revision"] != expected_revision:
                raise StaleDesignStateError(state.session_id, expected_revision, row["revision"])

            self._conn.execute(
                "UPDATE design_state SET revision = ?, stage = ?, updated_at = ?, payload_json = ? "
                "WHERE session_id = ?",
                (
                    state.revision,
                    state.stage.value,
                    state.updated_at,
                    state.model_dump_json(),
                    state.session_id,
                ),
            )
            self._conn.commit()
        return state

    def list_sessions(self, stage: str | None = None) -> list[str]:
        query = "SELECT session_id FROM design_state"
        params: tuple = ()
        if stage:
            query += " WHERE stage = ?"
            params = (stage,)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [r["session_id"] for r in rows]

    def delete(self, session_id: str, purge_blobs: bool = False) -> bool:
        """Delete a session from the store.

        If purge_blobs is True, also deletes any on-disk blobs associated with
        this session (tokens, confmap, finalize image) to prevent orphan accumulation.
        Returns True if session was found and deleted, False otherwise.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM design_state WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return False

            if purge_blobs:
                try:
                    from krisna_inference.backends.blob_store_singleton import get_blob_store
                    bs = get_blob_store()
                    st = DesignState.model_validate_json(row["payload_json"])
                    for ref in (
                        st.sketch_tokens.vq_tokens,
                        st.sketch_tokens.confidence_map,
                        getattr(st.finalize_output, "image_ref", None),
                    ):
                        if ref and isinstance(ref, str) and ref.startswith("blob://"):
                            try:
                                bs.delete(ref)
                            except Exception:
                                pass
                except Exception:
                    pass

            self._conn.execute("DELETE FROM design_state WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return True

    def close(self) -> None:
        with self._lock:
            self._conn.close()
