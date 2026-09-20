"""Tests validating Docker containerization configuration and isolation setup."""

from __future__ import annotations

from pathlib import Path
import yaml

from krisna_inference.common.hardware import find_repo_root


def test_dockerfile_inference_structure():
    root = find_repo_root()
    dockerfile = root / "docker" / "Dockerfile.inference"
    assert dockerfile.is_file(), "docker/Dockerfile.inference must exist"

    content = dockerfile.read_text(encoding="utf-8")

    # Base image with CUDA 12.4.1
    assert "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04" in content

    # Multi-venv isolation
    assert "/opt/venv-inference" in content
    assert "/opt/venv-critic" in content

    # Environment variables
    assert "KRISNA_CRITIC_VENV_PYTHON=/opt/venv-critic/bin/python" in content
    assert "KRISNA_USE_REAL_BACKENDS=1" in content
    assert "KRISNA_PORT=8420" in content

    # Requirements installations
    assert "requirements-inference.txt" in content
    assert "requirements-critic.txt" in content

    # Port and Entrypoint
    assert "EXPOSE 8420" in content
    assert 'ENTRYPOINT ["/app/docker/entrypoint.sh"]' in content


def test_dockerfile_frontend_structure():
    root = find_repo_root()
    dockerfile = root / "docker" / "Dockerfile.frontend"
    assert dockerfile.is_file(), "docker/Dockerfile.frontend must exist"

    content = dockerfile.read_text(encoding="utf-8")
    assert "FROM node:20-alpine" in content
    assert "EXPOSE 3000" in content
    assert 'CMD ["node", "server.js"]' in content


def test_entrypoint_script_validation():
    root = find_repo_root()
    entrypoint = root / "docker" / "entrypoint.sh"
    assert entrypoint.is_file(), "docker/entrypoint.sh must exist"

    content = entrypoint.read_text(encoding="utf-8")
    assert "check_hardware.py --require-gpu" in content
    assert "KRISNA_USE_REAL_BACKENDS" in content
    assert 'exec "$@"' in content


def test_docker_compose_valid_yaml_and_services():
    root = find_repo_root()
    compose_path = root / "docker-compose.yml"
    assert compose_path.is_file(), "docker-compose.yml must exist at repo root"

    with open(compose_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "services" in config
    services = config["services"]
    assert "inference" in services
    assert "frontend" in services

    # Inference service verification
    inf = services["inference"]
    assert "8420:8420" in inf["ports"]
    assert "KRISNA_USE_REAL_BACKENDS=1" in inf["environment"]
    assert "KRISNA_CRITIC_VENV_PYTHON=/opt/venv-critic/bin/python" in inf["environment"]
    assert "deploy" in inf
    reservations = inf["deploy"]["resources"]["reservations"]["devices"][0]
    assert reservations["driver"] == "nvidia"
    assert "gpu" in reservations["capabilities"]

    # Healthcheck
    assert "healthcheck" in inf
    assert any("/healthz" in str(arg) for arg in inf["healthcheck"]["test"])

    # Volumes
    vols = inf.get("volumes", [])
    assert any("./models:" in v for v in vols)
    assert any("./checkpoints:" in v for v in vols)
    assert any("./krisna_blobs:" in v for v in vols)

    # Frontend service verification
    fe = services["frontend"]
    assert "3000:3000" in fe["ports"]
    assert fe["depends_on"]["inference"]["condition"] == "service_healthy"

    # Networks
    assert "networks" in config
    assert "krisna-network" in config["networks"]


def test_dockerignore_exclusions():
    root = find_repo_root()
    dockerignore = root / ".dockerignore"
    assert dockerignore.is_file(), ".dockerignore must exist"

    content = dockerignore.read_text(encoding="utf-8")
    for pattern in [".git", "__pycache__", ".venv", "*.zip", "krisna_sessions.db"]:
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
