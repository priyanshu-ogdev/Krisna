# scripts/ — Cross-Platform Automation & Execution

Unified runner scripts supporting both **Linux/macOS (`bash`)** and **Windows (`PowerShell`)**.
All scripts are designed to be executed from the monorepo root.

---

## Root Orchestrators

Convenience wrappers at the repository root delegating to the appropriate package runners:

| Root Command (Bash) | Windows (PowerShell) | Description |
|---|---|---|
| `./setup.sh [frontend]` | `.\setup.sh` (Git Bash) | Sets up Python venvs, dependencies, and optional Node frontend. |
| `./run_data_forge.sh <command>` | `.\run_data_forge.ps1 <command>` | Executes the `data-forge` pipeline (planning, fetch, dedup, OCR, etc.). |
| `./train.sh <tier>` | `.\scripts\training\train_all.ps1 -Tier <tier>` | End-to-end training launcher (`sketch-stage1`, `sketch-stage2`, `polish-lora`, `polish-dpo`). |
| `./run_inference.sh` | `.\run_inference.ps1 [-Real] [-LowVram] [-Port 8420]` | Launches the FastAPI serving layer with MockBackend or Real backends. |

---

## Subsystem Scripts

### 1. `scripts/data-forge/`
- **`run_data_forge.sh` / `.ps1`**: Orchestrates pipeline stages with configurable `--data-root` and stage flags.
- **`setup_env.sh` / `.ps1`**: Installs data-forge dependencies (FAISS, MediaPipe, DeepSeek-OCR, PyArrow).
- **`sync_to_training.py`**: Exports processed manifests and pairs directly into training directory trees.
- **`pin_revisions.py`**: Resolves HuggingFace dataset and model tags to immutable git commit SHAs.
- **`verify_schemas.py`**: Validates JSON Schemas in `configs/schemas/` against Pydantic models.
- **`inspect_hf_dataset.py`**: Queries HuggingFace datasets server API for schemas, features, and split counts.

### 2. `scripts/training/`
- **Environment Setup**:
  - `setup_env_training.sh` / `.ps1`: Core training venv (PyTorch, torchvision, wandb, accelerate).
  - `setup_env_diffusers_training.sh` / `.ps1`: Diffusers/PEFT training environment for Z-Image-Turbo.
  - `setup_env_critic.sh`: Isolated venv for Gemma 4 Critic evaluation.
- **Data Preparation & Synchronization**:
  - `download_vqgan.sh` / `.ps1`: Fetches official VQGAN checkpoint/config needed for Sketch training & Finalize decoding.
  - `prepare_sketch_dataset.sh` / `.ps1` & `sync_from_data_forge_sketch.sh`: Bridges data-forge records into Sketch tokens.
  - `prepare_polish_dataset.sh` / `.ps1` & `sync_from_data_forge_polish.sh`: Bridges images for Z-Image-Turbo fine-tuning.
  - `sync_from_data_forge_dpo.sh`: Syncs human preference pairs for Diffusion-DPO alignment.
- **Training Launchers**:
  - `train_all.ps1`: Master multi-tier Windows PowerShell training pipeline with automatic environment isolation.
  - `train_sketch_stage1.sh`: Trains MaskGIT 256px from scratch (configs/sketch_stage1_256.yaml).
  - `train_sketch_stage2.sh`: Trains MaskGIT 512px with bicubic positional embedding interpolation.
  - `train_polish_default_lora.sh`: Fine-tunes Z-Image-Turbo via DreamBooth LoRA.
  - `train_polish_dpo.sh`: Direct Preference Optimization (DPO) velocity-loss refinement.
  - `train_planner_lora.sh` & `train_critic_qlora.sh`: Deprecated reference scripts (models ship frozen per PRD §6).

### 3. `scripts/inference/`
- **`check_hardware.py`**: Standalone preflight diagnostic CLI probing GPU, CUDA, compute capability, VRAM, and deployment configs (`--require-gpu`, `--low-vram`, `--deploy-check`, `--json`).
- **`download_weights.py`**: Discovers trained checkpoints, downloads frozen HF models, and generates `.env.inference`.
- **`run_service.sh` / `.ps1`**: Starts Uvicorn/FastAPI inference service (`--real`, `--low-vram`, host, port).
- **`setup_env.sh` / `setup_env_inference.sh` / `.ps1`**: Sets up inference virtual environment with hardware preflight check.
- **`setup_env_verifiers.sh`**: Installs verifier models (CLIP, OCR, aesthetic scorers).
- **`run_tests.sh`**: Executes inference test suite.

### 4. `scripts/docker/`
- **`run_docker.sh`**: Linux/macOS host preflight checker (Docker daemon & NVIDIA Container Toolkit) and Compose launcher (`--build`, `--detach`, `--low-vram`, `--down`).
- **`run_docker.ps1`**: Windows PowerShell launcher with Docker Desktop WSL2 GPU acceleration detection.

