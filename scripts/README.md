# Operational Scripts Directory (`scripts/`)

Central repository of execution scripts, automation harnesses, and hardware dispatchers for Krivex.
All commands assume execution from the monorepo root (e.g., `./scripts/inference/run_service.sh`, not `cd scripts/inference && ./run_service.sh`).

---

## Subsystem Index

### 1. `scripts/eval/` — Post-Training Multi-Hardware Evaluation & Paper Synchronization
Dedicated harness for empirical benchmark evaluation and automated research monograph synchronization:
- **`run_evaluation.py`**:
  - Evaluates multi-metric visual quality, layout fidelity, typographic legibility, and serving latency.
  - Profiles supported: `--profile 48gb` (Enterprise Max A6000), `--profile 24gb` (Mid-Tier RTX 3090/4090), `--profile 12gb` (Consumer Desktop RTX 4070).
  - Fast offline dry-run mode: `--synthetic` emulates generation passes without 14GB GPU allocations to verify metrics and exporters in seconds.
  - Generates `eval_comparison.json` (machine data), `eval_comparison.md` (Markdown summary), and `eval_comparison_latex.tex` (publication-grade LaTeX booktabs table).
- **`ingest_eval_results.py`**:
  - Ingests quantitative evaluation results and validates them against the 7 pre-registered OSF hypotheses (H1–H7).
  - Emits cryptographic provenance receipt `eval_results/evaluation_receipt.json` with SHA-256 digests, Git commit hashes, and local timestamps.
  - Updates empirical evaluation tables (Tables 11.1–11.6) in `docs/RESEARCH_PAPER.md` via `--apply`.
  - Detailed runbook: [scripts/eval/README.md](eval/README.md).

### 2. `scripts/training/` — Training Launchers & Hardware Dispatchers
1:1 cross-platform parity between Windows PowerShell (`train_all.ps1`) and Linux/WSL2 Bash (`train.sh`):
- **`setup_env_training.sh` / `setup_env_diffusers_training.sh`**: Installs CUDA 12.4+, PyTorch 2.6+, and pinned Diffusers (`efe46a5`).
- **`audit_training_pipeline.py`**: Non-destructive, read-only preflight audit of model IDs, realized dataset pair counts, and hardware readiness.
- **`train_polish_default_lora.sh`**: Production launcher for Qwen-Image-2.1 DoRA fine-tuning.
  - Supports Enterprise Max (`training/configs/polish_qwen_image21_lora_a6000_max.yaml`, batch=16, rank=128, FP32 AdamW, ~1.85h).
  - Supports Enterprise Standard (`training/configs/polish_qwen_image21_lora.yaml`, batch=8, rank=64, ~15.2h).
  - Supports Consumer Desktop (`training/configs/ablations/polish_qwen_qdora_12gb.yaml`, batch=1, rank=64, NF4).
- **`train_polish_dpo.sh`**: Deferred preference alignment launcher.
- **`train_polish_fast_preview.sh`**: Consistency Flow Matching distillation launcher.
- Detailed runbook: [scripts/training/README.md](training/README.md).

### 3. `scripts/inference/` — Serving & Runtime Harnesses
- **`setup_env.sh` / `setup_env_inference.sh` / `setup_env_verifiers.sh`**: Environment bootstrapping for the unified Qwen model plane.
- **`download_weights.py` / `download_planner_models.py`**: Model artifact discovery, HuggingFace download, and SHA-256 manifest verification.
- **`check_hardware.py`**: Probes local CUDA capabilities, VRAM capacity, and driver versions.
- **`run_service.sh` / `run_service.ps1`**: Starts the FastAPI inference orchestrator on port `8420`.
- **`run_tests.sh`**: Runs package-level tests with optional coverage reporting.

### 4. `scripts/data-forge/` — Data Pipeline & Dataset Bridges
1:1 parity between Windows PowerShell (`run_data_forge.ps1`) and Linux/WSL2 Bash (`run_data_forge.sh`):
- **`audit_registry.py`**: Preflight verification of license compliance, record estimates, and candidate status.
- **`pin_revisions.py`**: Resolves unpinned `main` branches to immutable 40-character Git commit SHAs.
- **`verify_schemas.py`**: Validates JSON Schemas against Pydantic models.
- **`sync_to_training.py`**: Bridges curated Figma2Code / UI pairs into HuggingFace dataset format with group-disjoint splits and 7-bucket aspect ratio manifold binning.
- Detailed runbook: [scripts/data-forge/README.md](data-forge/README.md).
