"""Focused coverage for provenance fields and explicit local source adapters."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pandas as pd
from data_forge.config import DatasetSpec
from data_forge.data.fetcher import DatasetFetcher
from data_forge.data.source_adapters import Figma2CodeAdapter, MobileViewsAdapter
from data_forge.manifest import Manifest
from PIL import Image


def _png_bytes(color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (12, 8), color=color).save(stream, format="PNG")
    return stream.getvalue()


def test_existing_manifest_gets_additive_provenance_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE records (
            id TEXT PRIMARY KEY, source_dataset TEXT NOT NULL, source_file TEXT,
            status TEXT NOT NULL, image_path TEXT, scrubbed_image_path TEXT,
            content_hash_sha256 TEXT, perceptual_hash TEXT, image_width INTEGER,
            image_height INTEGER, file_size_bytes INTEGER, license_verified INTEGER,
            license_output_json TEXT, aesthetic_score REAL, quality_output_json TEXT,
            pii_scrubbed INTEGER, pii_detections_json TEXT, safety_tier TEXT,
            safety_output_json TEXT, caption TEXT, caption_output_json TEXT,
            source_caption TEXT, ocr_output_json TEXT, structure_output_json TEXT,
            domain TEXT, shard_id TEXT, encoding_paths_json TEXT, audit_output_json TEXT,
            critique_output_json TEXT, exclusion_reason TEXT, duplicate_of TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE stage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL,
            stage TEXT NOT NULL, old_status TEXT, new_status TEXT NOT NULL,
            reason TEXT, timestamp TEXT NOT NULL
        );
        CREATE TABLE dataset_versions (
            version_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            record_count INTEGER, notes TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    manifest = Manifest(db_path)
    rec = manifest.create_record(
        "figma2code",
        source_record_id="node-7",
        source_group_id="file-3",
        canonical_id="canonical-7",
        source_metadata={"node": 7},
        source_structure={"elements": [{"type": "button"}]},
    )
    loaded = manifest.get_record(rec.id)
    assert loaded is not None
    assert loaded.source_record_id == "node-7"
    assert loaded.source_group_id == "file-3"
    assert loaded.canonical_id == "canonical-7"
    assert loaded.source_provenance is None
    assert loaded.source_metadata == {"node": 7}
    assert loaded.source_structure == {"elements": [{"type": "button"}]}
    manifest.close()


def test_bulk_create_and_update_support_canonical_fields(manifest: Manifest) -> None:
    assert manifest.bulk_create_records(
        [{
            "source_file": "screen.png",
            "canonical_id": "canon-1",
            "source_record_id": "screen-1",
            "source_group_id": "app-1",
            "source_metadata": {"app": "demo"},
            "source_structure": {"nodes": []},
        }],
        "mobileviews",
    ) == 1
    rec = manifest.query_by_dataset("mobileviews")[0]
    assert rec.canonical_id == "canon-1"
    manifest.update_record(rec.id, "adapter", provenance_url="https://example.test/source", group_id="app-1")
    updated = manifest.get_record(rec.id)
    assert updated is not None
    assert updated.provenance_url == "https://example.test/source"
    assert updated.group_id == "app-1"


def test_explicit_local_adapters_preserve_metadata_and_hash_saved_bytes(tmp_path: Path) -> None:
    source = tmp_path / "figma"
    source.mkdir()
    image_path = source / "screen.png"
    image_path.write_bytes(_png_bytes())
    (source / "screen.json").write_text(
        '{"node_id":"node-1","file_id":"file-9","structure":{"elements":[{"type":"button"}]}}',
        encoding="utf-8",
    )
    spec = DatasetSpec(
        display_name="Figma2Code",
        source_type="local",
        category="web_ui",
        fetch_config={"download_mode": "figma2code_local", "local_path": str(source)},
        provenance_url="https://example.test/figma",
    )
    adapter = Figma2CodeAdapter(
        key="figma2code",
        spec=spec,
        source_root=source,
        output_root=tmp_path / "out",
        data_root=tmp_path,
    )
    records = adapter.fetch()
    assert len(records) == 1
    record = records[0]
    saved = tmp_path / record["image_path"]
    assert record["content_hash_sha256"] == __import__("hashlib").sha256(saved.read_bytes()).hexdigest()
    assert record["source_record_id"] == "node-1"
    assert record["source_group_id"] == "file-9"
    assert record["source_provenance"]["adapter"] == "figma2code"
    assert record["source_structure"] == {"elements": [{"type": "button"}]}


async def test_fetcher_dispatches_explicit_mobileviews_adapter(config, tmp_path: Path) -> None:
    source = tmp_path / "mobileviews"
    source.mkdir()
    pd.DataFrame([{
        "image": {"bytes": _png_bytes(), "path": "screen.png"},
        "app_id": "app-2",
        "screen_id": "screen-4",
        "layout": {"elements": [{"type": "text"}]},
    }]).to_parquet(source / "part-0.parquet")
    spec = DatasetSpec(
        display_name="MobileViews",
        source_type="local",
        category="mobile_ui",
        fetch_config={
            "download_mode": "mobileviews_local",
            "local_path": str(source),
            "image_column": "image",
            "parquet_batch_size": 1,
        },
    )
    records = await DatasetFetcher(config).fetch_dataset("mobileviews", spec)
    assert len(records) == 1
    assert records[0]["source_record_id"] == "screen-4"
    assert records[0]["source_group_id"] == "app-2"
    assert records[0]["source_structure"] == {"elements": [{"type": "text"}]}
    assert records[0]["content_hash_sha256"] == __import__("hashlib").sha256(
        (config.data_root / records[0]["image_path"]).read_bytes()
    ).hexdigest()
    assert MobileViewsAdapter.adapter_name == "mobileviews"
