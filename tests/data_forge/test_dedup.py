"""Tests for Stage 2 Dedup and DedupEngine — exact hash + FAISS semantic deduplication."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from data_forge.config import PipelineConfig, PathsConfig, StageConfig
from data_forge.data.dedup import DedupEngine
from data_forge.manifest import Manifest
from data_forge.stages.s02_dedup import DedupStage
from data_forge.utils.hashing import sha256_file


class TestDedupEngine:
    def test_dedup_batch_intra_chunk(self):
        engine = DedupEngine(similarity_threshold=0.90)

        # 4 vectors: v0 and v1 are almost identical, v2 is orthogonal, v3 is identical to v0
        v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.99, 0.05, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        embeddings = np.stack([v0, v1, v2, v3])
        record_ids = ["rec_0", "rec_1", "rec_2", "rec_3"]

        duplicates, survivors = engine.dedup_batch(embeddings, record_ids)

        # rec_1 and rec_3 are duplicates of rec_0
        dup_ids = [d[0] for d in duplicates]
        canonical_ids = [d[1] for d in duplicates]

        assert "rec_1" in dup_ids
        assert "rec_3" in dup_ids
        assert canonical_ids == ["rec_0", "rec_0"]

        # Survivors must be rec_0 and rec_2
        survivor_ids = [record_ids[i] for i in survivors]
        assert survivor_ids == ["rec_0", "rec_2"]

    def test_search_existing_and_incremental_add(self, tmp_path: Path):
        engine = DedupEngine(similarity_threshold=0.90, index_type="Flat")

        v0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        engine.build_index(np.stack([v0, v1]), ["base_0", "base_1"])

        # Query chunk: q0 similar to base_0, q1 is completely new
        q0 = np.array([0.98, 0.05, 0.0], dtype=np.float32)
        q1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        duplicates, survivors = engine.search_existing(np.stack([q0, q1]), ["new_0", "new_1"])

        assert len(duplicates) == 1
        assert duplicates[0][0] == "new_0"
        assert duplicates[0][1] == "base_0"
        assert survivors == [1]  # new_1 survives

        # Add survivor to index
        engine.add_records(np.stack([q1]), ["new_1"])
        assert engine._index.ntotal == 3
        assert engine._id_map == ["base_0", "base_1", "new_1"]

    def test_hnsw_index_support(self):
        engine = DedupEngine(similarity_threshold=0.90, index_type="HNSW32")
        vectors = np.random.randn(20, 64).astype(np.float32)
        ids = [f"id_{i}" for i in range(20)]

        engine.build_index(vectors, ids)
        assert engine._index.ntotal == 20

        # Query
        duplicates, survivors = engine.search_existing(vectors[:5], ids[:5], threshold=0.99)
        assert len(duplicates) == 5
        assert len(survivors) == 0

    def test_atomic_save_and_load(self, tmp_path: Path):
        engine = DedupEngine(similarity_threshold=0.90)
        vectors = np.eye(4, dtype=np.float32)
        ids = ["a", "b", "c", "d"]

        engine.build_index(vectors, ids)
        index_file = tmp_path / "sub" / "faiss_index.bin"
        engine.save_index(index_file)

        assert index_file.exists()
        assert index_file.with_suffix(".ids.json").exists()

        # Load back
        engine2 = DedupEngine()
        engine2.load_index(index_file)
        assert engine2._index.ntotal == 4
        assert engine2._id_map == ids


class MockClipModel:
    def __init__(self, dim=32):
        self.device = "cpu"
        self.dtype = torch.float32
        self.config = MagicMock()
        self.config.projection_dim = dim
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param

    def get_image_features(self, **kwargs):
        pixel_values = kwargs["pixel_values"]
        batch_size = pixel_values.shape[0]
        features = torch.zeros(batch_size, 32)
        for b in range(batch_size):
            mean_val = float(pixel_values[b].mean())
            bin_idx = min(31, int(mean_val * 32))
            features[b, bin_idx] = 1.0
        return features


class MockClipProcessor:
    def __call__(self, images, return_tensors="pt", padding=True):
        tensors = []
        for img in images:
            arr = np.array(img, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append(t)
        return {"pixel_values": torch.stack(tensors)}


class MockEngine:
    def __init__(self):
        self.clip_model = MockClipModel()
        self.clip_processor = MockClipProcessor()


@pytest.fixture
def dedup_env(tmp_path: Path):
    manifest = Manifest(tmp_path / "manifest.db")
    config = PipelineConfig()
    config.data_root = tmp_path
    config.paths = PathsConfig(
        manifests="manifests",
        checkpoints="checkpoints",
    )
    config.stages = {
        "s02_dedup": StageConfig(
            enabled=True,
            params={
                "exact_hash_dedup": True,
                "semantic_dedup": True,
                "similarity_threshold": 0.95,
            },
        )
    }
    config.resolved_paths = config.paths.resolve(config.data_root)
    return manifest, config


class TestDedupStage:
    @pytest.mark.asyncio
    async def test_exact_hash_deduplication(self, dedup_env, tmp_path: Path):
        manifest, config = dedup_env
        stage = DedupStage()

        # Create two identical image files and one distinct image file
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        img_a = Image.new("RGB", (64, 64), color="red")
        path_a1 = raw_dir / "img_a1.png"
        path_a2 = raw_dir / "img_a2.png"
        img_a.save(path_a1)
        img_a.save(path_a2)

        img_b = Image.new("RGB", (64, 64), color="blue")
        path_b = raw_dir / "img_b.png"
        img_b.save(path_b)

        rec1 = manifest.create_record(
            "test_ds",
            image_path="raw/img_a1.png",
            content_hash_sha256=sha256_file(path_a1),
        )
        rec2 = manifest.create_record(
            "test_ds",
            image_path="raw/img_a2.png",
            content_hash_sha256=sha256_file(path_a2),
        )
        rec3 = manifest.create_record(
            "test_ds",
            image_path="raw/img_b.png",
            content_hash_sha256=sha256_file(path_b),
        )

        result = await stage.run(manifest, config, [rec1.id, rec2.id, rec3.id], engine=None)

        assert result.records_processed == 3
        assert result.records_excluded == 1
        assert result.metadata["exact_duplicates"] == 1

        r1 = manifest.get_record(rec1.id)
        r2 = manifest.get_record(rec2.id)
        r3 = manifest.get_record(rec3.id)

        assert r1.status == "deduped"
        assert r2.status == "excluded_duplicate"
        assert r2.duplicate_of == rec1.id
        assert r3.status == "deduped"

    @pytest.mark.asyncio
    async def test_missing_image_files_marked_failed(self, dedup_env):
        manifest, config = dedup_env
        stage = DedupStage()

        # Record points to non-existent image
        rec = manifest.create_record("test_ds", image_path="raw/non_existent.png")

        result = await stage.run(manifest, config, [rec.id], engine=None)

        assert result.records_excluded == 1
        r = manifest.get_record(rec.id)
        assert r.status == "excluded_failed"
        assert "not found" in r.exclusion_reason

    @pytest.mark.asyncio
    async def test_semantic_deduplication_multi_chunk(self, dedup_env, tmp_path: Path):
        manifest, config = dedup_env
        stage = DedupStage()
        engine = MockEngine()

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Chunk 1: Two images with distinct content
        img1 = Image.new("RGB", (64, 64), color=(100, 100, 100))
        p1 = raw_dir / "c1_img1.png"
        img1.save(p1)

        img2 = Image.new("RGB", (64, 64), color=(200, 200, 200))
        p2 = raw_dir / "c1_img2.png"
        img2.save(p2)

        r1 = manifest.create_record("test_ds", image_path="raw/c1_img1.png")
        r2 = manifest.create_record("test_ds", image_path="raw/c1_img2.png")

        res1 = await stage.run(manifest, config, [r1.id, r2.id], engine=engine)
        assert res1.records_excluded == 0
        assert manifest.get_record(r1.id).status == "deduped"
        assert manifest.get_record(r2.id).status == "deduped"

        # Check FAISS index was saved and has 2 records
        index_file = config.resolved_paths["manifests"] / "faiss_index.bin"
        assert index_file.exists()

        # Chunk 2:
        # r3 is semantically near-identical to r1 (different bytes, same semantic bin)
        # r4 is a new unique image
        img3 = Image.new("RGB", (64, 64), color=(101, 101, 101))
        p3 = raw_dir / "c2_img3.png"
        img3.save(p3)

        img4 = Image.new("RGB", (64, 64), color=(50, 50, 50))
        p4 = raw_dir / "c2_img4.png"
        img4.save(p4)

        r3 = manifest.create_record("test_ds", image_path="raw/c2_img3.png")
        r4 = manifest.create_record("test_ds", image_path="raw/c2_img4.png")

        res2 = await stage.run(manifest, config, [r3.id, r4.id], engine=engine)
        assert res2.records_excluded == 1
        assert res2.metadata["semantic_duplicates"] == 1
        assert res2.metadata["historical_duplicates"] == 1

        rec3_after = manifest.get_record(r3.id)
        assert rec3_after.status == "excluded_duplicate"
        assert rec3_after.duplicate_of == r1.id

        rec4_after = manifest.get_record(r4.id)
        assert rec4_after.status == "deduped"

        # Verify FAISS index now has 3 records (r1, r2, r4) and NEVER added r3
        dedup_check = DedupEngine()
        dedup_check.load_index(index_file)
        assert dedup_check._index.ntotal == 3
        assert set(dedup_check._id_map) == {r1.id, r2.id, r4.id}
        assert r3.id not in dedup_check._id_map

    def test_dense_cluster_deduplication(self):
        engine = DedupEngine(similarity_threshold=0.90)
        # Cluster of 60 identical vectors
        vectors = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (60, 1))
        ids = [f"clust_{i}" for i in range(60)]

        duplicates, survivors = engine.dedup_batch(vectors, ids)

        assert len(duplicates) == 59
        assert len(survivors) == 1
        assert survivors == [0]
        # All 59 duplicates must point directly to clust_0
        for dup_id, canonical_id, sim in duplicates:
            assert canonical_id == "clust_0"

    def test_ivf_nprobe_restored_on_load(self, tmp_path: Path):
        engine = DedupEngine(similarity_threshold=0.90, index_type="IVF", nprobe=42)
        vectors = np.random.randn(300, 32).astype(np.float32)
        ids = [f"id_{i}" for i in range(300)]
        engine.build_index(vectors, ids)

        index_file = tmp_path / "ivf_test.bin"
        engine.save_index(index_file)

        engine2 = DedupEngine(similarity_threshold=0.90, nprobe=99)
        engine2.load_index(index_file)
        assert hasattr(engine2._index, "nprobe")
        assert engine2._index.nprobe == 99

    @pytest.mark.asyncio
    async def test_missing_hashes_bulk_computed(self, dedup_env, tmp_path: Path):
        manifest, config = dedup_env
        stage = DedupStage()

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        img = Image.new("RGB", (32, 32), color="purple")
        path1 = raw_dir / "hash_test_1.png"
        path2 = raw_dir / "hash_test_2.png"
        img.save(path1)
        img.save(path2)

        # Neither record has content_hash_sha256 initially
        rec1 = manifest.create_record("test_ds", image_path="raw/hash_test_1.png")
        rec2 = manifest.create_record("test_ds", image_path="raw/hash_test_2.png")
        assert rec1.content_hash_sha256 is None
        assert rec2.content_hash_sha256 is None

        res = await stage.run(manifest, config, [rec1.id, rec2.id], engine=None)

        assert res.metadata["exact_duplicates"] == 1
        # Check that hash was computed and stored in manifest
        r1_after = manifest.get_record(rec1.id)
        assert r1_after.content_hash_sha256 is not None
        assert r1_after.status == "deduped"

        r2_after = manifest.get_record(rec2.id)
        assert r2_after.status == "excluded_duplicate"
        assert r2_after.duplicate_of == rec1.id

    def test_transparent_png_compositing(self, tmp_path: Path):
        # Create an RGBA transparent image
        img = Image.new("RGBA", (64, 64), color=(0, 0, 0, 0))  # Fully transparent
        path = tmp_path / "transparent.png"
        img.save(path)

        mock_clip = MockClipModel(dim=32)
        mock_proc = MockClipProcessor()

        embeddings, valid_idx, failed_idx = DedupEngine.generate_embeddings(
            image_paths=[path],
            clip_model=mock_clip,
            clip_processor=mock_proc,
            batch_size=10,
        )

        assert len(valid_idx) == 1
        assert len(failed_idx) == 0
        assert embeddings.shape == (1, 32)

