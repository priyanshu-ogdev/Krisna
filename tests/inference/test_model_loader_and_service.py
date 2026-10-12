from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from krisna_inference.backends.model_loader import (
    LlamaCppConfig,
    discover_planner_model,
    discover_mmproj_model,
    probe_cuda_hardware,
    sha256_file,
    verify_manifest,
)
from krisna_inference.orchestrator.service import app


@pytest.fixture
def client():
    return TestClient(app)


def test_llama_cpp_config_generation():
    config = LlamaCppConfig(
        model_path="models/planner/test.gguf",
        mmproj_path="models/planner/mmproj.gguf",
        n_ctx=8192,
        n_gpu_layers=33,
        use_mmap=True,
        use_mlock=False,
        flash_attn=True,
        n_batch=512,
        n_threads=6,
    )
    kwargs = config.to_llama_kwargs()
    assert kwargs["model_path"] == "models/planner/test.gguf"
    assert kwargs["clip_model_path"] == "models/planner/mmproj.gguf"
    assert kwargs["n_ctx"] == 8192
    assert kwargs["n_gpu_layers"] == 33
    assert kwargs["use_mmap"] is True
    assert kwargs["use_mlock"] is False
    assert kwargs["flash_attn"] is True
    assert kwargs["n_batch"] == 512
    assert kwargs["n_threads"] == 6


def test_sha256_and_manifest_verification(tmp_path: Path):
    dummy_model = tmp_path / "model.gguf"
    dummy_model.write_bytes(b"GGUF_TEST_PAYLOAD")

    expected_hash = sha256_file(dummy_model)
    assert len(expected_hash) == 64

    manifest_data = {
        "repo_id": "test/model-repo",
        "artifacts": [{"filename": "model.gguf", "sha256": expected_hash}],
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data))

    res = verify_manifest(manifest_file)
    assert res["valid"] is True
    assert res["repo_id"] == "test/model-repo"


def test_discover_planner_model_explicit(tmp_path: Path):
    model_file = tmp_path / "custom.gguf"
    model_file.write_bytes(b"dummy")
    discovered = discover_planner_model(explicit_path=str(model_file))
    assert discovered == model_file.resolve()


def test_probe_cuda_hardware_safe_return():
    hw = probe_cuda_hardware()
    assert "cuda_available" in hw
    assert "vram_total_gb" in hw
    assert "recommended_gpu_layers" in hw
    assert "supports_flash_attn" in hw


def test_service_ready_endpoint(client: TestClient):
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert "ready" in data
    assert "residency_state" in data
    assert data["store_accessible"] is True


def test_service_models_status_endpoint(client: TestClient):
    resp = client.get("/models/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "residency_state" in data
    assert "planner" in data
    assert data["planner"]["runtime"] == "llama.cpp"
    assert "polish" in data
    assert "vram_ledger" in data
    assert "ram_ledger" in data
    assert "cuda_hardware" in data


def test_service_models_warm_and_unload(client: TestClient):
    # Warm models
    resp_warm = client.post("/models/warm")
    assert resp_warm.status_code == 200
    assert resp_warm.json()["status"] == "warmed"

    # Unload models
    resp_unload = client.post("/models/unload")
    assert resp_unload.status_code == 200
    assert resp_unload.json()["status"] == "unloaded"
