"""SQLite store for DPO preference pairs. Same threading/locking pattern as
store.DesignStateStore — this is a local single-GPU service, not a
distributed system.

UPGRADE (training-loop review, bug found in data_forge_bridge/
sync_dpo_pairs.py): `add()` uses `INSERT OR IGNORE` rather than a plain
`INSERT`. `PreferencePair.id` defaults to a fresh random uuid4 per
Python-side construction, so re-running a data-forge sync against
already-imported source data previously created a full set of brand-new
rows for the exact same underlying image pairs every time — silently
duplicating a source's contribution every resync, skewing dataset size
and per-source sampling weight. `INSERT OR IGNORE` makes `add()` safe to
call repeatedly with the SAME id (a no-op on the second+ call) — but
this only actually fixes the bug once the id itself is deterministic,
which is sync_dpo_pairs.py's responsibility (derived from
source+pair_id/image filenames there, not left as the random default).
Runtime-labeled pairs (verifier_stack/gemma_critique/uicrit_seed) are
unaffected — those still get a fresh random id per real labeling event,
which is correct; they aren't being "resynced" from a stable source.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS preference_pairs (
    id              TEXT PRIMARY KEY,
    session_id      TEXT,
    prompt          TEXT NOT NULL,
    chosen_ref      TEXT NOT NULL,
    rejected_ref    TEXT NOT NULL,
    chosen_score    REAL,
    rejected_score  REAL,
    source          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pref_pairs_source ON preference_pairs(source);
CREATE INDEX IF NOT EXISTS idx_pref_pairs_session ON preference_pairs(session_id);
"""

# Valid sources include real human-labeled external preference datasets (Pick-a-Pic v2, HPDv2,
# DesignSense-10k, DesignPref) and runtime session sources.
VALID_SOURCES = frozenset({
    "verifier_stack", "gemma_critique", "uicrit_seed",
    "pickapic_v2", "hpdv2", "designsense_10k", "designpref",
})


@dataclass
class PreferencePair:
    prompt: str
    chosen_ref: str
    rejected_ref: str
    source: str
    session_id: str | None = None
    chosen_score: float | None = None
    rejected_score: float | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(f"Unknown source {self.source!r}; must be one of {sorted(VALID_SOURCES)}")
        if self.chosen_ref == self.rejected_ref:
            raise ValueError("chosen_ref and rejected_ref cannot be identical")


class PreferenceStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def add(self, pair: PreferencePair) -> PreferencePair:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO preference_pairs
                    (id, session_id, prompt, chosen_ref, rejected_ref,
                     chosen_score, rejected_score, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pair.id,
                    pair.session_id,
                    pair.prompt,
                    pair.chosen_ref,
                    pair.rejected_ref,
                    pair.chosen_score,
                    pair.rejected_score,
                    pair.source,
                    pair.created_at,
                ),
            )
            self._conn.commit()
        return pair

    def list(
        self,
        source: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[PreferencePair]:
        query = "SELECT * FROM preference_pairs WHERE 1=1"
        params: list = []
        if source:
            query += " AND source = ?"
            params.append(source)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        return [
            PreferencePair(
                id=r["id"],
                session_id=r["session_id"],
                prompt=r["prompt"],
                chosen_ref=r["chosen_ref"],
                rejected_ref=r["rejected_ref"],
                chosen_score=r["chosen_score"],
                rejected_score=r["rejected_score"],
                source=r["source"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def count(self, source: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM preference_pairs"
        params: list = []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        with self._lock:
            return self._conn.execute(query, params).fetchone()[0]

    def exists(self, pair_id: str) -> bool:
        """UPGRADE: lets a sync path skip re-doing image I/O (blob-store
        writes, decode) for a pair it already imported, rather than only
        deduping at the final INSERT step (see module docstring) — that
        alone still leaked a fresh blob copy per resync even once the DB
        row itself became a no-op."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM preference_pairs WHERE id = ? LIMIT 1", (pair_id,)
            ).fetchone()
        return row is not None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
