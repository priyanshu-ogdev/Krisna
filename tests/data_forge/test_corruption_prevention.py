import math
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from data_forge.manifest import Manifest, _json_dumps, _json_loads
from data_forge.utils.image_utils import load_image, normalize_for_vae


def test_manifest_integrity_check_healthy(tmp_path: Path):
    db_file = tmp_path / "test_manifest.db"
    manifest = Manifest(db_file)
    is_healthy, msgs = manifest.check_integrity(full=False)
    assert is_healthy is True
    assert msgs == ["ok"]

    is_healthy_full, msgs_full = manifest.check_integrity(full=True)
    assert is_healthy_full is True
    assert msgs_full == ["ok"]
    manifest.close()


def test_manifest_integrity_check_corrupted(tmp_path: Path):
    db_file = tmp_path / "corrupt_manifest.db"
    manifest = Manifest(db_file)
    # Add record to ensure pages exist
    manifest.create_record(source_dataset="test_ds", image_path="img.png")
    manifest.close()

    # Intentionally corrupt the SQLite file by overwriting page bytes past the header
    with open(db_file, "r+b") as f:
        f.seek(100)  # past 100-byte SQLite header
        f.write(b"\xff\xff\xff\xff" * 200)

    # Opening corrupted DB should fail integrity check on startup
    with pytest.raises(sqlite3.DatabaseError) as exc_info:
        Manifest(db_file)
    err_str = str(exc_info.value).lower()
    assert "corrupt" in err_str or "malformed" in err_str


def test_manifest_json_corruption_resilience():
    # Sanitizing NaN/Inf floats in dumps
    payload = {
        "score": float("nan"),
        "ratio": float("inf"),
        "valid": 0.95,
        "nested": {"nested_nan": float("-inf"), "count": 42},
    }
    dumped = _json_dumps(payload)
    assert dumped is not None
    # Must be standard RFC 8259 compliant JSON (no raw NaN or Infinity)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped

    loaded = _json_loads(dumped)
    assert loaded["score"] is None
    assert loaded["ratio"] is None
    assert loaded["valid"] == 0.95
    assert loaded["nested"]["nested_nan"] is None
    assert loaded["nested"]["count"] == 42

    # Robust handling of malformed or empty strings
    assert _json_loads(None) is None
    assert _json_loads("") is None
    assert _json_loads("   ") is None
    assert _json_loads("{malformed json truncated...") is None


def test_image_utils_corruption_resilience(tmp_path: Path):
    # 0-byte image file
    zero_file = tmp_path / "zero.png"
    zero_file.write_bytes(b"")
    with pytest.raises(ValueError) as exc:
        load_image(zero_file)
    assert "0 bytes" in str(exc.value)

    # Missing file
    missing_file = tmp_path / "missing.jpg"
    with pytest.raises(ValueError):
        load_image(missing_file)

    # Corrupted / truncated image bytes
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtruncated_junk")
    with pytest.raises(Exception):
        load_image(corrupt_file)


def test_normalize_for_vae_clamps_nans():
    # A 3-channel tensor containing NaN or Inf should be sanitized by nan_to_num
    tensor_with_nan = torch.randn(3, 4, 4)
    tensor_with_nan[0, 1, 1] = float("nan")
    tensor_with_nan[1, 2, 2] = float("inf")
    tensor_with_nan[2, 0, 0] = float("-inf")
    normalized = normalize_for_vae(tensor_with_nan)
    assert torch.isfinite(normalized).all()
    assert not torch.isnan(normalized).any()


def test_dedup_clip_nan_filtering():
    from data_forge.data.dedup import DedupEngine

    # Mock inputs
    mock_model = MagicMock()
    mock_model.dtype = torch.float32
    mock_model.device = "cpu"
    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.zeros((2, 3, 224, 224))}

    # Two valid images
    img1 = Image.new("RGB", (64, 64), color="red")
    img2 = Image.new("RGB", (64, 64), color="blue")
    paths = [Path("fake1.png"), Path("fake2.png")]

    with patch.object(DedupEngine, "_preprocess_single_image") as mock_prep:
        mock_prep.side_effect = [
            (0, img1, None),
            (1, img2, None),
        ]
        # Return features where row 1 has a NaN
        feat1 = torch.tensor([[0.5, 0.5]])
        feat2 = torch.tensor([[float("nan"), 0.5]])
        mock_model.get_image_features.return_value = torch.cat([feat1, feat2], dim=0)

        result, valid_indices, failed_indices = DedupEngine.generate_embeddings(
            image_paths=paths,
            clip_model=mock_model,
            clip_processor=mock_processor,
            batch_size=2,
            device="cpu",
        )

        # Corrupt row 1 must be filtered out into failed_indices
        assert 1 in failed_indices
        assert 0 in valid_indices
        assert result.shape[0] == 1
        assert np.isfinite(result).all()


@pytest.mark.asyncio
async def test_s03_quality_nan_aesthetic_score_rejected(tmp_path: Path):
    from data_forge.config import PipelineConfig
    from data_forge.stages.s03_quality import QualityStage

    db_file = tmp_path / "test_s03.db"
    manifest = Manifest(db_file)

    img_path = tmp_path / "test.png"
    Image.new("RGB", (512, 512), "white").save(img_path)

    rec = manifest.create_record(
        source_dataset="test",
        image_path=str(img_path),
        image_width=512,
        image_height=512,
    )
    manifest.update_record(rec.id, stage="dedup", new_status="deduped")

    mock_engine = MagicMock()
    mock_tier1 = MagicMock()
    # Mock quality output returning NaN
    mock_quality_out = MagicMock()
    mock_quality_out.aesthetic_score = float("nan")
    mock_tier1.score_quality = AsyncMock(return_value=mock_quality_out)

    config = MagicMock(spec=PipelineConfig)
    config.data_root = tmp_path
    config.get_stage.return_value = {
        "aesthetic_score_threshold": 0.4,
        "min_resolution": [256, 256],
        "max_resolution": [4096, 4096],
        "batch_size": 16,
    }

    stage = QualityStage()
    with patch("data_forge.stages.s03_quality.Tier1Engine", return_value=mock_tier1):
        res = await stage.run(manifest, config, [rec.id], engine=mock_engine)

    updated_rec = manifest.get_record(rec.id)
    assert updated_rec is not None
    assert updated_rec.status == "excluded_failed"
    assert "inference_failed" in updated_rec.exclusion_reason.lower()


@pytest.mark.asyncio
async def test_s08_encoding_nan_latent_fails_cleanly(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace
    from data_forge.config import EncoderSpec, PipelineConfig, StageConfig
    from data_forge.data.storage import StorageManager
    from data_forge.stages.s08_encoding import EncodingStage

    monkeypatch.setattr(StorageManager, "mid_flight_check", lambda self: {"safe": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = PipelineConfig()
    config.data_root = tmp_path
    config.encoders = {
        "z_image_vae": EncoderSpec(model_id="test-vae", revision="test-revision")
    }
    config.stages["s08_encoding"] = StageConfig(
        enabled=True,
        params={"vae_batch_size": 2, "max_encoding_resolution": 256},
    )

    img_path = tmp_path / "input.png"
    Image.new("RGB", (64, 64), (100, 150, 200)).save(img_path)

    record = SimpleNamespace(
        id="rec-nan-test",
        status="routed",
        scrubbed_image_path=img_path.name,
        structure_output={"layout_type": "test"},
    )

    class _ManifestMock:
        def __init__(self, recs):
            self.recs = recs
            self.updates = []

        def get_records_by_ids(self, ids):
            return [r for r in self.recs if r.id in ids]

        def bulk_update_records(self, updates, stage):
            self.updates.extend(updates)

    # VAE mock that returns NaN latents
    class _NanVae:
        def encode(self, batch):
            nan_latent = torch.full((batch.shape[0], 4, 8, 8), float("nan"), dtype=batch.dtype)
            return SimpleNamespace(
                latent_dist=SimpleNamespace(sample=lambda: nan_latent)
            )

    class _EngineMock:
        def get_encoder(self, name):
            return _NanVae()

    manifest = _ManifestMock([record])
    stage = EncodingStage()

    result = await stage.run(manifest, config, [record.id], _EngineMock())
    assert result.records_failed == 1
    assert result.records_processed == 0

    assert len(manifest.updates) == 1
    failed_update = manifest.updates[0]
    assert failed_update["new_status"] == "excluded_failed"
    assert failed_update["exclusion_reason"] == "encoding_incomplete"
    assert "Corrupted tensor 'latent'" in failed_update["reason"]

