"""Tests for the manifest module — record state machine, versioning, queries."""

from __future__ import annotations

import pytest

from data_forge.manifest import Manifest


class TestManifestCreation:
    def test_create_record(self, manifest: Manifest):
        rec = manifest.create_record("rico_core", source_file="img001.png")
        assert rec.id is not None
        assert rec.status == "fetched"
        assert rec.source_dataset == "rico_core"
        assert rec.source_file == "img001.png"
        assert rec.created_at != ""

    def test_bulk_create(self, manifest: Manifest):
        records = [
            {"source_file": f"img{i:03d}.png", "image_path": f"raw/test/img{i:03d}.png"}
            for i in range(100)
        ]
        count = manifest.bulk_create_records(records, "test_ds")
        assert count == 100
        assert manifest.total_count() == 100

    def test_get_record(self, manifest: Manifest):
        rec = manifest.create_record("test_ds")
        fetched = manifest.get_record(rec.id)
        assert fetched is not None
        assert fetched.id == rec.id

    def test_get_nonexistent(self, manifest: Manifest):
        assert manifest.get_record("nonexistent-id") is None


class TestManifestStateTransitions:
    def test_update_status(self, manifest: Manifest):
        rec = manifest.create_record("test_ds")
        manifest.update_record(rec.id, "dedup", new_status="deduped")
        updated = manifest.get_record(rec.id)
        assert updated is not None
        assert updated.status == "deduped"

    def test_invalid_status_raises(self, manifest: Manifest):
        rec = manifest.create_record("test_ds")
        with pytest.raises(ValueError, match="Invalid status"):
            manifest.update_record(rec.id, "test", new_status="invalid_status")

    def test_update_json_fields(self, manifest: Manifest):
        rec = manifest.create_record("test_ds")
        manifest.update_record(
            rec.id, "safety",
            safety_tier="safe",
            safety_output={"tier": "safe", "confidence": 0.95, "rationale": "ok", "flags": []},
        )
        updated = manifest.get_record(rec.id)
        assert updated is not None
        assert updated.safety_tier == "safe"
        assert updated.safety_output is not None
        assert updated.safety_output["confidence"] == 0.95

    def test_update_boolean_fields(self, manifest: Manifest):
        rec = manifest.create_record("test_ds")
        manifest.update_record(rec.id, "pii", pii_scrubbed=True)
        updated = manifest.get_record(rec.id)
        assert updated is not None
        assert updated.pii_scrubbed is True


class TestManifestQueries:
    def test_query_by_status(self, populated_manifest: Manifest):
        fetched = populated_manifest.query_by_status("fetched")
        assert len(fetched) == 10

    def test_count_by_status(self, populated_manifest: Manifest):
        counts = populated_manifest.count_by_status()
        assert counts.get("fetched") == 10

    def test_split_into_chunks(self, populated_manifest: Manifest):
        chunks = populated_manifest.split_into_chunks(3)
        assert len(chunks) == 4  # 10 records / 3 per chunk = 4 chunks
        assert len(chunks[0]) == 3
        assert len(chunks[-1]) == 1

    def test_check_hash_exists(self, manifest: Manifest):
        rec = manifest.create_record("ds", source_file="a.png")
        manifest.update_record(rec.id, "fetch", content_hash_sha256="abc123")
        found = manifest.check_hash_exists("abc123")
        assert found == rec.id
        assert manifest.check_hash_exists("nonexistent") is None

    def test_stats(self, populated_manifest: Manifest):
        stats = populated_manifest.stats()
        assert stats["total_records"] == 10
        assert stats["status_counts"]["fetched"] == 10


class TestDatasetVersioning:
    def test_create_version(self, manifest: Manifest):
        manifest.create_dataset_version("krisna_v001", notes="Test")
        latest = manifest.get_latest_version()
        assert latest == "krisna_v001"

    def test_version_ordering(self, manifest: Manifest):
        manifest.create_dataset_version("krisna_v001")
        import time; time.sleep(0.01)
        manifest.create_dataset_version("krisna_v002")
        assert manifest.get_latest_version() == "krisna_v002"
