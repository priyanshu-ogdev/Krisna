"""SQLite-backed manifest — single source of truth for every pipeline record.

The manifest tracks every image through its full lifecycle: fetched → deduped →
quality_scored → pii_scrubbed → safety_classified → recaptioned → structured →
routed → encoded → audited → training_pool | excluded_*.

All mutations are logged with timestamp, stage, and reason for full audit trail.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_forge.logging_setup import get_logger

log = get_logger("manifest")

# Valid status transitions
VALID_STATUSES = frozenset([
    "fetched",
    "deduped",
    "quality_scored",
    "pii_scrubbed",
    "safety_classified",
    "recaptioned",
    "structured",
    "routed",
    "encoded",
    "audited",
    "training_pool",
    "heldout",
    "excluded_duplicate",
    "excluded_low_quality",
    "excluded_unsafe",
    "excluded_pending_review",
    "excluded_failed",
    "overflow_excluded",
])

# Statuses that stop further processing
TERMINAL_STATUSES = frozenset([
    "training_pool",
    "heldout",
    "excluded_duplicate",
    "excluded_low_quality",
    "excluded_unsafe",
    "excluded_pending_review",
    "excluded_failed",
    "overflow_excluded",
])

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records (
    id                    TEXT PRIMARY KEY,
    source_dataset        TEXT NOT NULL,
    source_file           TEXT,
    status                TEXT NOT NULL DEFAULT 'fetched',
    image_path            TEXT,
    scrubbed_image_path   TEXT,
    content_hash_sha256   TEXT,
    perceptual_hash       TEXT,
    image_width           INTEGER,
    image_height          INTEGER,
    file_size_bytes       INTEGER,
    license_verified      INTEGER,  -- 0/1 boolean
    license_output_json   TEXT,     -- JSON blob
    aesthetic_score       REAL,
    quality_output_json   TEXT,
    pii_scrubbed          INTEGER,  -- 0/1 boolean
    pii_detections_json   TEXT,     -- JSON array
    safety_tier           TEXT,
    safety_output_json    TEXT,
    caption               TEXT,
    caption_output_json   TEXT,
    ocr_output_json       TEXT,
    structure_output_json TEXT,
    domain                TEXT,
    shard_id              TEXT,
    encoding_paths_json   TEXT,     -- JSON dict
    audit_output_json     TEXT,
    exclusion_reason      TEXT,
    duplicate_of          TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
CREATE INDEX IF NOT EXISTS idx_records_source ON records(source_dataset);
CREATE INDEX IF NOT EXISTS idx_records_domain ON records(domain);
CREATE INDEX IF NOT EXISTS idx_records_shard ON records(shard_id);
CREATE INDEX IF NOT EXISTS idx_records_hash ON records(content_hash_sha256);

CREATE TABLE IF NOT EXISTS stage_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   TEXT NOT NULL,
    stage       TEXT NOT NULL,
    old_status  TEXT,
    new_status  TEXT NOT NULL,
    reason      TEXT,
    timestamp   TEXT NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(id)
);

CREATE INDEX IF NOT EXISTS idx_history_record ON stage_history(record_id);
CREATE INDEX IF NOT EXISTS idx_history_stage ON stage_history(stage);

CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    record_count INTEGER,
    notes        TEXT
);
"""


@dataclass
class ManifestRecord:
    """In-memory representation of a single manifest record."""

    id: str
    source_dataset: str
    source_file: str | None = None
    status: str = "fetched"
    image_path: str | None = None
    scrubbed_image_path: str | None = None
    content_hash_sha256: str | None = None
    perceptual_hash: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    file_size_bytes: int | None = None
    license_verified: bool | None = None
    license_output: dict[str, Any] | None = None
    aesthetic_score: float | None = None
    quality_output: dict[str, Any] | None = None
    pii_scrubbed: bool | None = None
    pii_detections: list[str] | None = None
    safety_tier: str | None = None
    safety_output: dict[str, Any] | None = None
    caption: str | None = None
    caption_output: dict[str, Any] | None = None
    ocr_output: dict[str, Any] | None = None
    structure_output: dict[str, Any] | None = None
    domain: str | None = None
    shard_id: str | None = None
    encoding_paths: dict[str, str] | None = None
    audit_output: dict[str, Any] | None = None
    exclusion_reason: str | None = None
    duplicate_of: str | None = None
    created_at: str = ""
    updated_at: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _json_loads(s: str | None) -> Any:
    if s is None:
        return None
    return json.loads(s)


def _row_to_record(row: sqlite3.Row) -> ManifestRecord:
    return ManifestRecord(
        id=row["id"],
        source_dataset=row["source_dataset"],
        source_file=row["source_file"],
        status=row["status"],
        image_path=row["image_path"],
        scrubbed_image_path=row["scrubbed_image_path"],
        content_hash_sha256=row["content_hash_sha256"],
        perceptual_hash=row["perceptual_hash"],
        image_width=row["image_width"],
        image_height=row["image_height"],
        file_size_bytes=row["file_size_bytes"],
        license_verified=bool(row["license_verified"]) if row["license_verified"] is not None else None,
        license_output=_json_loads(row["license_output_json"]),
        aesthetic_score=row["aesthetic_score"],
        quality_output=_json_loads(row["quality_output_json"]),
        pii_scrubbed=bool(row["pii_scrubbed"]) if row["pii_scrubbed"] is not None else None,
        pii_detections=_json_loads(row["pii_detections_json"]),
        safety_tier=row["safety_tier"],
        safety_output=_json_loads(row["safety_output_json"]),
        caption=row["caption"],
        caption_output=_json_loads(row["caption_output_json"]),
        ocr_output=_json_loads(row["ocr_output_json"]),
        structure_output=_json_loads(row["structure_output_json"]),
        domain=row["domain"],
        shard_id=row["shard_id"],
        encoding_paths=_json_loads(row["encoding_paths_json"]),
        audit_output=_json_loads(row["audit_output_json"]),
        exclusion_reason=row["exclusion_reason"],
        duplicate_of=row["duplicate_of"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class Manifest:
    """SQLite-backed manifest for pipeline record tracking.

    Thread-safety: each Manifest instance owns one connection.
    For concurrent access (e.g., multi-worker), use separate instances
    with WAL mode (enabled by default).
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        log.info("manifest_opened", db_path=str(db_path))

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── Record CRUD ──────────────────────────────────────────────────────

    def create_record(
        self,
        source_dataset: str,
        source_file: str | None = None,
        image_path: str | None = None,
        **kwargs: Any,
    ) -> ManifestRecord:
        """Insert a new record with status='fetched'."""
        record_id = str(uuid.uuid4())
        now = _now_iso()
        with self._transaction() as cur:
            cur.execute(
                """INSERT INTO records
                   (id, source_dataset, source_file, status, image_path,
                    created_at, updated_at)
                   VALUES (?, ?, ?, 'fetched', ?, ?, ?)""",
                (record_id, source_dataset, source_file, image_path, now, now),
            )
            # Log creation in stage history
            cur.execute(
                """INSERT INTO stage_history
                   (record_id, stage, old_status, new_status, reason, timestamp)
                   VALUES (?, 'fetch', NULL, 'fetched', 'record created', ?)""",
                (record_id, now),
            )

        record = ManifestRecord(
            id=record_id,
            source_dataset=source_dataset,
            source_file=source_file,
            status="fetched",
            image_path=image_path,
            created_at=now,
            updated_at=now,
        )
        return record

    def get_record(self, record_id: str) -> ManifestRecord | None:
        cur = self._conn.execute("SELECT * FROM records WHERE id = ?", (record_id,))
        row = cur.fetchone()
        return _row_to_record(row) if row else None

    def update_record(
        self,
        record_id: str,
        stage: str,
        new_status: str | None = None,
        reason: str | None = None,
        **fields: Any,
    ) -> None:
        """Update a record's fields and optionally transition its status.

        All JSON-typed fields (license_output, safety_output, etc.) are
        automatically serialized from dicts/lists.
        """
        if new_status and new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {new_status}")

        now = _now_iso()
        set_clauses = ["updated_at = ?"]
        params: list[Any] = [now]

        if new_status:
            set_clauses.append("status = ?")
            params.append(new_status)

        # Map Python field names to DB column names
        json_fields = {
            "license_output": "license_output_json",
            "quality_output": "quality_output_json",
            "pii_detections": "pii_detections_json",
            "safety_output": "safety_output_json",
            "caption_output": "caption_output_json",
            "ocr_output": "ocr_output_json",
            "structure_output": "structure_output_json",
            "encoding_paths": "encoding_paths_json",
            "audit_output": "audit_output_json",
        }
        bool_fields = {"license_verified", "pii_scrubbed"}

        for key, value in fields.items():
            col = json_fields.get(key, key)
            if key in json_fields:
                set_clauses.append(f"{col} = ?")
                params.append(_json_dumps(value))
            elif key in bool_fields:
                set_clauses.append(f"{col} = ?")
                params.append(1 if value else 0)
            else:
                set_clauses.append(f"{col} = ?")
                params.append(value)

        params.append(record_id)

        with self._transaction() as cur:
            cur.execute(
                f"UPDATE records SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )
            if new_status:
                # Get old status for history
                row = cur.execute(
                    "SELECT status FROM records WHERE id = ?", (record_id,)
                ).fetchone()
                old_status = row["status"] if row else None
                cur.execute(
                    """INSERT INTO stage_history
                       (record_id, stage, old_status, new_status, reason, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (record_id, stage, old_status, new_status, reason, now),
                )

    def bulk_create_records(
        self, records: list[dict[str, Any]], source_dataset: str
    ) -> int:
        """Bulk-insert records for a dataset fetch. Returns count inserted."""
        now = _now_iso()
        inserted = 0
        with self._transaction() as cur:
            for rec in records:
                record_id = str(uuid.uuid4())
                cur.execute(
                    """INSERT INTO records
                       (id, source_dataset, source_file, status, image_path,
                        content_hash_sha256, image_width, image_height,
                        file_size_bytes, created_at, updated_at)
                       VALUES (?, ?, ?, 'fetched', ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record_id,
                        source_dataset,
                        rec.get("source_file"),
                        rec.get("image_path"),
                        rec.get("content_hash_sha256"),
                        rec.get("image_width"),
                        rec.get("image_height"),
                        rec.get("file_size_bytes"),
                        now,
                        now,
                    ),
                )
                inserted += 1
        log.info(
            "bulk_created",
            source_dataset=source_dataset,
            count=inserted,
        )
        return inserted

    # ── Queries ───────────────────────────────────────────────────────────

    def query_by_status(
        self, status: str, limit: int | None = None
    ) -> list[ManifestRecord]:
        sql = "SELECT * FROM records WHERE status = ?"
        params: list[Any] = [status]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_record(r) for r in rows]

    def query_by_statuses(self, statuses: list[str]) -> list[ManifestRecord]:
        placeholders = ",".join("?" for _ in statuses)
        sql = f"SELECT * FROM records WHERE status IN ({placeholders})"
        rows = self._conn.execute(sql, statuses).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_active_records(self) -> list[ManifestRecord]:
        """Records not in a terminal/excluded status — still being processed."""
        terminal = ",".join(f"'{s}'" for s in TERMINAL_STATUSES)
        sql = f"SELECT * FROM records WHERE status NOT IN ({terminal})"
        rows = self._conn.execute(sql).fetchall()
        return [_row_to_record(r) for r in rows]

    def get_training_pool(self) -> list[ManifestRecord]:
        return self.query_by_status("training_pool")

    def get_heldout(self) -> list[ManifestRecord]:
        return self.query_by_status("heldout")

    def get_excluded_pending_review(self) -> list[ManifestRecord]:
        return self.query_by_status("excluded_pending_review")

    def iter_records_by_status(
        self, status: str, batch_size: int = 1000
    ) -> Iterator[list[ManifestRecord]]:
        """Yield batches of records with a given status for memory-efficient iteration."""
        offset = 0
        while True:
            rows = self._conn.execute(
                "SELECT * FROM records WHERE status = ? LIMIT ? OFFSET ?",
                (status, batch_size, offset),
            ).fetchall()
            if not rows:
                break
            yield [_row_to_record(r) for r in rows]
            offset += batch_size

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM records GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def total_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM records").fetchone()
        return row["cnt"] if row else 0

    def check_hash_exists(self, sha256: str) -> str | None:
        """Check if a content hash already exists. Returns record ID if found."""
        row = self._conn.execute(
            "SELECT id FROM records WHERE content_hash_sha256 = ? LIMIT 1",
            (sha256,),
        ).fetchone()
        return row["id"] if row else None

    def split_into_chunks(self, chunk_size: int) -> list[list[str]]:
        """Split all active (non-terminal) record IDs into chunks for processing."""
        terminal = ",".join(f"'{s}'" for s in TERMINAL_STATUSES)
        rows = self._conn.execute(
            f"SELECT id FROM records WHERE status NOT IN ({terminal}) ORDER BY rowid"
        ).fetchall()
        ids = [r["id"] for r in rows]
        return [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]

    def get_records_by_ids(self, record_ids: list[str]) -> list[ManifestRecord]:
        if not record_ids:
            return []
        placeholders = ",".join("?" for _ in record_ids)
        rows = self._conn.execute(
            f"SELECT * FROM records WHERE id IN ({placeholders})", record_ids
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    # ── Dataset Versions ──────────────────────────────────────────────────

    def create_dataset_version(
        self, version_id: str, notes: str | None = None
    ) -> None:
        now = _now_iso()
        count = self.total_count()
        with self._transaction() as cur:
            cur.execute(
                """INSERT INTO dataset_versions (version_id, created_at, record_count, notes)
                   VALUES (?, ?, ?, ?)""",
                (version_id, now, count, notes),
            )
        log.info("dataset_version_created", version=version_id, records=count)

    def get_latest_version(self) -> str | None:
        row = self._conn.execute(
            "SELECT version_id FROM dataset_versions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return row["version_id"] if row else None

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return comprehensive pipeline statistics."""
        status_counts = self.count_by_status()
        total = self.total_count()
        latest_version = self.get_latest_version()

        # Storage stats
        row = self._conn.execute(
            "SELECT SUM(file_size_bytes) as total_bytes FROM records"
        ).fetchone()
        total_bytes = row["total_bytes"] or 0

        return {
            "total_records": total,
            "status_counts": status_counts,
            "total_raw_bytes": total_bytes,
            "total_raw_gb": round(total_bytes / 1e9, 2),
            "latest_version": latest_version,
            "training_pool_count": status_counts.get("training_pool", 0),
            "heldout_count": status_counts.get("heldout", 0),
            "excluded_count": sum(
                v for k, v in status_counts.items() if k.startswith("excluded_")
            ),
        }
