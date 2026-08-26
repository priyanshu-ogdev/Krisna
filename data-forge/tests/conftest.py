"""Shared test fixtures for data-forge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_forge.config import PipelineConfig
from data_forge.manifest import Manifest


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test data."""
    return tmp_path


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Temporary DATA_ROOT."""
    root = tmp_path / "data_krisna"
    root.mkdir()
    return root


@pytest.fixture
def config(tmp_path: Path, data_root: Path) -> PipelineConfig:
    """Test pipeline config with temporary paths."""
    config = PipelineConfig()
    config.data_root = data_root
    config.resolved_paths = config.paths.resolve(data_root)
    config.chunk_size = 100
    config.prompts_dir = Path(__file__).parent.parent / "configs" / "prompts"
    config.schemas_dir = Path(__file__).parent.parent / "configs" / "schemas"
    return config


@pytest.fixture
def manifest(data_root: Path) -> Manifest:
    """Fresh manifest database."""
    db_path = data_root / "manifests" / "test_manifest.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    m = Manifest(db_path)
    yield m
    m.close()


@pytest.fixture
def sample_image(data_root: Path) -> Path:
    """Create a minimal test PNG image."""
    from PIL import Image

    img_dir = data_root / "raw" / "test_dataset"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "test_image.png"

    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    img.save(img_path)
    return img_path


@pytest.fixture
def populated_manifest(manifest: Manifest, sample_image: Path, data_root: Path) -> Manifest:
    """Manifest with 10 sample records."""
    for i in range(10):
        rel_path = str(sample_image.relative_to(data_root))
        manifest.create_record(
            source_dataset="test_dataset",
            source_file=f"test_{i}.png",
            image_path=rel_path,
        )
    return manifest


@pytest.fixture
def mock_vllm_response() -> dict:
    """Canned vLLM response."""
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "caption": "A mobile login screen with username and password fields, a blue submit button, and a 'Forgot Password' link below.",
                    "ui_elements_mentioned": ["text_input", "button", "text_label"],
                    "confidence": 0.92,
                })
            }
        }]
    }


@pytest.fixture
def mock_safety_response() -> dict:
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "tier": "safe",
                    "confidence": 0.95,
                    "rationale": "Standard mobile app login screen with no harmful content.",
                    "flags": [],
                })
            }
        }]
    }


@pytest.fixture
def mock_quality_response() -> dict:
    return {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "aesthetic_score": 0.72,
                    "resolution_adequate": True,
                    "is_complete_ui": True,
                    "design_era": "modern",
                    "issues": [],
                    "confidence": 0.88,
                })
            }
        }]
    }


@pytest.fixture
def sample_vae_config(tmp_path: Path) -> Path:
    """Create a sample VAE config file."""
    config_dir = tmp_path / "vae_model"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps({
        "latent_channels": 64,
        "spatial_downsample_ratio": 16,
        "in_channels": 3,
        "out_channels": 3,
    }))
    return config_dir
