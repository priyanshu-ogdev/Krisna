# Data-Forge — Zero-Touch Data Pipeline (v13)

> Automated data pipeline for the **Krisna** project. Implements the v13 specification:
> every verification, judgment, and spot-check step that was previously manual is now
> handled by deterministic logic or a resident reasoning/VLM model via vLLM.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Orchestrator (Chunk-Based)             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ ┌──────┐ │
│  │Stage 0 │→│Stage 1 │→│Stage 2 │→│Stage 3  │→│ 3.5  │ │
│  │Manifest│ │Fetch+  │ │Dedup   │ │Quality  │ │ PII  │ │
│  │Planning│ │License │ │(FAISS) │ │Scoring  │ │Scrub │ │
│  └────────┘ └────────┘ └────────┘ └─────────┘ └──────┘ │
│       ↓                                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │Stage 4 │→│  4.5   │→│Stage 5 │→│Stage 6 │→│Stage 7 │ │
│  │Safety  │ │Escalate│ │Recap+  │ │Struct  │ │Routing │ │
│  │Tier-1  │ │Tier-2  │ │OCR     │ │Extract │ │& Shard │ │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ │
│       ↓                                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            │
│  │Stage 8 │→│Stage 9 │→│Stage 10│ │Stage 11│            │
│  │Tri-Path│ │Heldout │ │Audit   │ │Registry│            │
│  │Encode  │ │Carve   │ │Pass    │ │Watcher │            │
│  └────────┘ └────────┘ └────────┘ └────────┘            │
└──────────────────────────────────────────────────────────┘
```

## Requirements

- **OS**: Ubuntu 22.04 / 24.04 LTS (production), Windows (development)
- **CUDA**: 12.4 (strictly pinned)
- **Python**: 3.10 or 3.11
- **GPU**: 48GB VRAM (e.g., RTX 6000 Ada, A6000, L40S)
- **RAM**: 128GB system RAM recommended
- **Disk**: ≥3TB at `DATA_ROOT`

## Setup & Verification

To properly run the data-forge on Windows, you must use the provided bootstrap script. The script resolves pip compatibility issues by using Conda to install `faiss-gpu` and the correct CUDA 12.4 PyTorch wheels.

### 1. Environment Bootstrap
```powershell
# Navigate to the project root
cd data-forge

# Run the Windows environment setup script (requires Conda installed)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_env.ps1

# The script creates the environment. You must manually activate it before proceeding:
conda activate krisna-forge
```
*Linkage*: The script handles Python 3.10 initialization, `torch+cu124`, `faiss-gpu`, and finally executes `pip install -e .[dev]` to link the `data-forge` CLI commands globally to this environment.

### 2. Pre-Flight Schema Verification
Before loading any heavy GPU models, validate that your YAML configurations meet the strict Pydantic requirements of the v13 spec.
```powershell
python scripts/verify_schemas.py
```

### 3. Pipeline Dry-Run Validation
Verify the SQLite manifest connection (WAL-mode locking) and storage quota limits without spinning up the inference engine.
```powershell
$env:DATA_ROOT="D:\kf_data"
data-forge run --dry-run
```

## Usage

Once setup and verification are complete, you can execute the pipeline natively.

```powershell
# Set required credentials
$env:DATA_ROOT="D:\kf_data"
$env:HF_TOKEN="your_huggingface_token"

# Execute the Smoke Test (100-record micro-batch)
data-forge run --chunk-size 100 --limit 100

# Full pipeline run
data-forge run

# Resume from checkpoint
data-forge run --resume

# Run specific stages (e.g., fetch, dedup, quality)
data-forge run --stages 0,1,2

# Registry watcher (schedule this via Windows Task Scheduler)
data-forge registry check

# Inspect manifest
data-forge manifest stats
data-forge manifest query --status excluded_pending_review
```

## Configuration

All configuration lives in `configs/`:
- `pipeline.yaml` — stage toggles, thresholds, paths, chunk sizes
- `models.yaml` — pinned model versions, quant settings, VRAM budgets
- `datasets.yaml` — source dataset URLs, license status, categories

Paths in `pipeline.yaml` are **relative to `DATA_ROOT`**.

## Stage Reference

| Stage | Name | GPU Model | Purpose |
|-------|------|-----------|---------|
| 00 | Manifest Planning | — | Init DB, storage check, read watcher report |
| 01 | Fetch & License | Tier-1 | Download + inline license verification |
| 02 | Dedup | CLIP | FAISS near-duplicate removal |
| 03 | Quality | Tier-1 | Aesthetic/resolution scoring |
| 03.5 | PII Scrub | MediaPipe | Face blur, text redaction |
| 04 | Safety | Tier-1 | NSFW/harmful classification |
| 04.5 | Escalation | Tier-2 | Borderline second opinion |
| 05 | Recaption + OCR | Tier-1 → OCR | Dense captioning + text extraction |
| 06 | Structure | Tier-1 | UI layout JSON extraction |
| 07 | Routing | — | Domain tagging + shard assignment |
| 08 | Tri-Path Encode | VAEs/VQ | Z-Image, Qwen-Image, MaskGIT tokens |
| 09 | Heldout Carve | — | Stratified eval set split |
| 10 | Audit Pass | Tier-1 + Tier-2 | VLM-as-judge on 2-5% sample |
| 11 | Registry Watch | — | Model/dataset release polling |

## Testing

```bash
# Unit tests (no GPU required)
pytest tests/ -v -m "not integration"

# Integration tests (requires GPU + models downloaded)
pytest tests/ -v -m integration

# Coverage
pytest tests/ --cov=data_forge --cov-report=html
```
