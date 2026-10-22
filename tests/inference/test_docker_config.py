"""Tests validating Docker containerization configuration and isolation setup."""

from __future__ import annotations

from pathlib import Path
import yaml

from krivex_inference.common.hardware import find_repo_root


def test_dockerfile_inference_structure():
    root = find_repo_root()
    dockerfile = root / "inference" / "Dockerfile"
    assert dockerfile.is_file(), "inference/Dockerfile must exist"

    content = dockerfile.read_text(encoding="utf-8")

    # Base image with CUDA 12.4.1
    assert "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04" in content

    # Unified venv architecture (all Qwen-based models in single runtime)
    assert "/opt/venv-inference" in content
    assert "/opt/venv-critic" not in content

    # Environment variables
    assert "KRIVEX_USE_REAL_BACKENDS=1" in content
    assert "KRIVEX_PORT=8420" in content

    # Requirements installations
    assert "requirements.txt" in content

    # Port and Entrypoint
    assert "EXPOSE 8420" in content
    assert 'ENTRYPOINT ["/app/inference/entrypoint.sh"]' in content


def test_entrypoint_script_validation():
    root = find_repo_root()
    entrypoint = root / "inference" / "entrypoint.sh"
    assert entrypoint.is_file(), "inference/entrypoint.sh must exist"

    content = entrypoint.read_text(encoding="utf-8")
    assert "check_hardware.py --require-gpu" in content
    assert "KRIVEX_USE_REAL_BACKENDS" in content
    assert 'exec "$@"' in content


def test_docker_compose_valid_yaml_and_services():
    root = find_repo_root()
    compose_path = root / "inference" / "docker-compose.yml"
    assert compose_path.is_file(), "inference/docker-compose.yml must exist"

    with open(compose_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "services" in config
    services = config["services"]
    assert "inference" in services
    # Single unified container serves both API and Studio UI; no separate frontend container
    assert "frontend" not in services

    # Inference service verification
    inf = services["inference"]
    assert "8420:8420" in inf["ports"]
    assert "KRIVEX_USE_REAL_BACKENDS=1" in inf["environment"]
    assert "deploy" in inf
    reservations = inf["deploy"]["resources"]["reservations"]["devices"][0]
    assert reservations["driver"] == "nvidia"
    assert "gpu" in reservations["capabilities"]

    # Healthcheck
    assert "healthcheck" in inf
    assert any("/healthz" in str(arg) for arg in inf["healthcheck"]["test"])

    # Volumes
    vols = inf.get("volumes", [])
    assert any("../models:" in v for v in vols)
    assert any("../checkpoints:" in v for v in vols)
    assert any("../data:" in v for v in vols)


def test_root_cleanliness_no_docker_in_root():
    root = find_repo_root()
    assert not (root / "docker").exists(), "root / docker directory must be cleaned up"
    assert not (root / "docker-compose.yml").exists(), "root / docker-compose.yml must be cleaned up"


def test_dockerignore_exclusions():
    root = find_repo_root()
    dockerignore = root / ".dockerignore"
    assert dockerignore.is_file(), ".dockerignore must exist"

    content = dockerignore.read_text(encoding="utf-8")
    for pattern in [".git", "__pycache__", ".venv", "*.zip", "krivex_sessions.db"]:
        assert pattern in content


def test_docker_run_scripts_exist():
    root = find_repo_root()
    sh_script = root / "scripts" / "docker" / "run_docker.sh"
    ps_script = root / "scripts" / "docker" / "run_docker.ps1"

    assert sh_script.is_file(), "scripts/docker/run_docker.sh must exist"
    assert ps_script.is_file(), "scripts/docker/run_docker.ps1 must exist"

    sh_content = sh_script.read_text(encoding="utf-8")
    assert "docker info" in sh_content
    assert "nvidia" in sh_content

    ps_content = ps_script.read_text(encoding="utf-8")
    assert "docker info" in ps_content
