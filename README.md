# Krisna: Agentic Design System & Data-Forge

**Krisna** is a single-GPU (48GB VRAM), conversational AI design agent capable of generating, iterating, and polishing UI/graphic designs through human-in-the-loop interaction. It utilizes a novel **Two-Tier Generation Architecture** (Sparse Masked Sketching -> Continuous Flow Polishing) and is powered by a fully automated, zero-touch data ingestion pipeline (**Data-Forge**).

## 🏗️ High-Level System Architecture

Krisna separates the *interactive exploration* phase from the *high-fidelity rendering* phase to maintain sub-second conversational latency without sacrificing final output quality.

```text
User Prompt / Feedback
       │
       ▼
┌──────────────────────────────────────┐
│  Qwen3.5-9B Planner/Critic (BF16)    │ ◄── Agentic Orchestrator
│  • Intent extraction & JSON state    │     (Design State Manager)
│  • Continuous conditioning embeds    │
└──────────────┬───────────────────────┘
               │
   ┌───────────▼────────────┐
   │ Stage 1: Sketch Tier   │ ◄── MaskGIT / MAR Lineage (Discrete VQ Tokens)
   │ • 8-Step Masked Gen.   │     • Halton Scheduler (Spatial dispersion)
   │ • Interactive/Iterative│     • Token-Critic (Confidence gating)
   └───────────┬────────────┘
               │ User approves layout/structure
               ▼
   ┌──────────────────────────────────────┐
   │ Stage 2: Polish Tier (Handoff)       │ ◄── Z-Image-Turbo / Qwen-Image-2.0
   │ • Continuous Flow-Matching           │     • In-context token conditioning
   │ • Region-locked differential diffusion│
   └──────────────────────────────────────┘
               │
               ▼
      Verifier Stack (CLIP, OCR, Layout-IoU) ──► Final Export
```

### 🧠 The Model Stack
| Component | Model Family | Role | Precision / Serving |
| :--- | :--- | :--- | :--- |
| **Planner / Critic** | Qwen3.5-9B (or 4B) | Conversational state management, design critique, tool routing. | BF16 (Interactive) / FP8 (Serving) |
| **Sketch Tier** | MaskGIT / MaskGIL (Custom) | Sparse, iterative layout and structural exploration. | Discrete VQ Tokens (INT16) |
| **Polish Tier (Primary)** | Z-Image-Turbo (6B) | Fast, high-fidelity final rendering and texture synthesis. | NF4 / AWQ (vLLM) |
| **Polish Tier (Quality)** | Qwen-Image-2.0 (15B) | Complex semantic rendering and strict prompt adherence. | NF4 / AWQ (vLLM) |

---

## ⚙️ The Data-Forge: Zero-Touch Ingestion Pipeline

The Data-Forge is a custom, 12-stage Python orchestrator designed to process millions of images into training-ready latents on a single workstation. It replaces manual curation with **VLM-as-Judge** auditing and deterministic logic.

### Pipeline Stages
1. **`s00_manifest_planning`**: Pre-flight storage checks, registry watcher sync.
2. **`s01_fetch`**: HuggingFace/GitHub ingestion + Inline License Verification Agent.
3. **`s02_dedup`**: GPU-accelerated FAISS semantic near-duplicate removal.
4. **`s03_quality`**: Aesthetic and resolution scoring via Tier-1 VLM.
5. **`s03_5_pii_scrub`**: **(Critical)** Regex text redaction + MediaPipe face blurring for UI screenshots.
6. **`s04_safety`**: NSFW/Harmful content classification.
7. **`s04_5_escalation`**: Tier-2 VLM second-opinion on borderline records.
8. **`s05_recaption`**: Dense structural captioning + DeepSeek-OCR text extraction.
9. **`s06_structure`**: UI component tree and bounding box JSON extraction.
10. **`s07_routing`**: Domain tagging and stratified shard routing.
11. **`s08_encoding`**: **Tri-Path Latent Encoding** (Z-Image VAE, Qwen VAE, VQ Tokens, Control Maps).
12. **`s09_heldout`**: Stratified evaluation set carve-out.
13. **`s10_audit`**: Automated VLM-as-Judge rubric evaluation (replaces human spot-checks).
14. **`s11_registry_watcher`**: Automated polling for new SOTA model releases.

### 💾 Tri-Path Latent Storage Strategy
To support both the discrete Sketch Tier and continuous Polish Tier, the Data-Forge encodes every approved image into four distinct representations:
1. `latents_zimage/` (fp16 `.safetensors`) - Continuous latents for Z-Image-Turbo.
2. `latents_qwenimage/` (fp16 `.safetensors`) - Continuous latents for Qwen-Image-2.0.
3. `vq_tokens_sketch/` (int16 `.pt`) - Discrete codebook indices for MaskGIT/MAR.
4. `control_tokens/` (`.json`) - Structural bounding boxes and Canny edges for handoff conditioning.

---

## 💻 Hardware & Infrastructure Target

Krisna is explicitly designed to run on a **Single 48GB Workstation GPU** (e.g., NVIDIA RTX A6000 / Ada 6000). 

* **Inference Engine**: vLLM with dynamic model swapping (Tier-1 <-> Tier-2 <-> OCR) and strict `psutil` zombie-process teardown.
* **Orchestration**: Custom Python DAG with SQLite WAL-mode manifest for ACID-compliant state tracking across millions of records.
* **OS**: Windows 11 / Ubuntu 22.04 (Fully cross-platform via `pathlib` and `spawn` multiprocessing guards).

## 🚀 Setup, Verification & Execution Pipeline (Windows)

The data-forge relies on a carefully linked sequence of scripts to securely build the environment, validate schemas, and execute the orchestrator. Follow these steps in order:

### 1. Environment Bootstrap (`scripts/setup_env.ps1`)
Because `faiss-gpu` lacks pre-compiled pip wheels for Windows, we use a Conda bootstrap script to bridge the dependency gap before installing the core Python toolchain.

```powershell
# Open a PowerShell terminal
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_env.ps1

# The script creates the environment. You must manually activate it before proceeding:
conda activate krisna-forge
```
*Linkage*: This script sets up Python 3.10, installs `torch` with CUDA 12.4, uses conda for `faiss-gpu`, and finally runs `pip install -e .[dev]` to link the `data-forge` CLI.

### 2. Pre-Flight Schema Verification (`scripts/verify_schemas.py`)
Before processing terabytes of data, validate that your YAML configurations (`configs/pipeline.yaml`, `configs/models.yaml`) meet the strict Pydantic requirements of v13.

```powershell
python scripts/verify_schemas.py
```
*Linkage*: This utility ensures your model pins, chunk sizes, and VRAM budgets are correctly typed before the orchestrator boots, preventing mid-run crashes.

### 3. Pipeline Dry-Run Validation (`--dry-run`)
Validate the SQLite manifest connection, the storage budget, and the stage DAG without spinning up the GPU models.

```powershell
$env:DATA_ROOT="D:\kf_data"
python -m data_forge.cli run --dry-run
```
*Linkage*: The `run` command invokes `data_forge/cli.py`, which instantiates the `Orchestrator`. The `--dry-run` flag bypasses `engine.py` (vLLM subprocesses), verifying the local DAG logic and SQLite `IMMEDIATE` lock resiliency.

### 4. The Smoke Test Micro-Run (End-to-End)
Execute a 100-record micro-batch to prove the VRAM swapping, Tri-Path Encoding, and model teardowns function flawlessly on your hardware.

```powershell
$env:HF_TOKEN="your_huggingface_token"
python -m data_forge.cli run --chunk-size 100 --limit 100
```
*Linkage*: This activates the full infrastructure. The orchestrator processes the chunk, dynamically spinning up models via `subprocess.Popen` in `engine.py`, tearing them down recursively using `psutil` to prevent zombie VRAM leaks.

### 5. Registry Watcher Sync
Schedule this to poll for new open-source models and datasets automatically.
```powershell
python -m data_forge.cli registry check
```

### 6. Launch the Agentic Server
*(Coming in Phase A Implementation)*
```bash
python -m krisna.serve --planner qwen3.5-9b --renderer z-image-turbo
```
