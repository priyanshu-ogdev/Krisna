# Krisna: Agentic Design System & Data-Forge

**Krisna** is a single-GPU (48GB VRAM), conversational AI design agent capable of generating, iterating, and polishing UI/graphic designs through human-in-the-loop interaction. It utilizes a novel **Two-Tier Generation Architecture** (Sparse Masked Sketching -> Continuous Flow Polishing), an on-demand **Critic Tier** for VL-judged design review, and is powered by a fully automated, zero-touch data ingestion pipeline (**Data-Forge**).

## 🏗️ High-Level System Architecture

Krisna separates the *interactive exploration* phase from the *high-fidelity rendering* phase to maintain sub-second conversational latency without sacrificing final output quality. A third, on-demand Critic Tier is swapped in only when explicitly requested — never resident during conversation.

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
   │ Stage 2: Polish Tier (Handoff)       │ ◄── Z-Image-Turbo / Qwen-Image-Edit-2511
   │ • Continuous Flow-Matching           │     • In-context token conditioning
   │ • Region-locked differential diffusion│
   └──────────────────────────────────────┘
               │
               ▼
      Verifier Stack (CLIP, OCR, Layout-IoU) ──► Final Export
               │
               │ optional, explicit "critique this"
               ▼
   ┌──────────────────────────────────────┐
   │ Tier C: Critic (on-demand)            │ ◄── Gemma 4 31B Dense (NF4)
   │ • Judges the finished render only     │     Apache 2.0, QLoRA via Unsloth
   │ • Never resident during conversation  │
   └──────────────────────────────────────┘
```

### 🧠 The Model Stack
| Component | Model Family | Role | Precision / Serving |
| :--- | :--- | :--- | :--- |
| **Planner / Critic** | Qwen3.5-9B (or 4B) | Conversational state management, design critique, tool routing. | BF16 (Interactive) / FP8 (Serving) |
| **Sketch Tier** | MaskGIT / MaskGIL (Custom) | Sparse, iterative layout and structural exploration. | Discrete VQ Tokens (INT16) |
| **Polish Tier (Primary)** | Z-Image-Turbo (6B) | Fast, high-fidelity final rendering and texture synthesis. | NF4 / AWQ (vLLM) |
| **Polish Tier (Quality)** | Qwen-Image-Edit-2511 (20B) | Complex semantic rendering, treats Stage-1 output as an edit target. | NF4 / AWQ (vLLM) |
| **Critic Tier (on-demand)** | Gemma 4 31B Dense | UI/UX critique scoring, seeds `/ui_critique/` for DPO. | NF4 (Apache 2.0, QLoRA via Unsloth) |

> **Note on Qwen-Image-2.0**: earlier revisions of this document specified Qwen-Image-2.0 (15B) as the quality-tier polish model. Its weights were never open-sourced — confirmed via the official changelog, an unanswered HF community thread (Feb–Apr 2026), and Alibaba having since moved to Qwen-Image-3.0 without ever releasing 2.0's weights. **Qwen-Image-Edit-2511 replaces it** everywhere in this repo (confirmed downloadable, Apache 2.0, same MMDiT lineage). If you see "Qwen-Image-2.0" referenced anywhere outside this note, treat it as stale and file an issue.

---

## ⚙️ The Data-Forge: Zero-Touch Ingestion Pipeline

The Data-Forge is a custom, 20-stage Python orchestrator designed to process millions of images into training-ready latents on a single workstation. It replaces manual curation with **VLM-as-Judge** auditing and deterministic logic. **v15 adds dedicated data production for every one of the PRD's five trainable components** (previously, three of five had incomplete or entirely missing data paths — see `data-forge/docs/DATA_COMPLETENESS.md` for the full trace).

### Pipeline Stages
1. **`s00_manifest_planning`**: Pre-flight storage checks, registry watcher sync.
2. **`s01_fetch`**: HuggingFace/GitHub ingestion + Inline License Verification Agent. Includes a dedicated URL-list download path (img2dataset-style) for metadata-only sources like PD12M/CC12M, which ship as parquet/TSV + external image URLs rather than bundled files.
3. **`s01_5_uicrit_join`**: *(New)* Joins UICrit's real human critique/ratings against already-ingested RICO records — UICrit isn't a standalone image dataset, its screenshots are RICO's.
4. **`s01_6_planner_synthesis`**: *(New)* Generates the Planner's (Qwen3.5-9B) conversational SFT training data, seeded from real UICrit critique text — previously the Planner had no training data path at all.
5. **`s02_dedup`**: GPU-accelerated FAISS semantic near-duplicate removal.
6. **`s03_quality`**: Aesthetic and resolution scoring via Tier-1 VLM.
7. **`s03_5_pii_scrub`**: **(Critical)** Regex text redaction + MediaPipe face blurring for UI screenshots.
8. **`s04_safety`**: NSFW/Harmful content classification.
9. **`s04_5_escalation`**: Tier-2 VLM second-opinion on borderline records.
10. **`s05_recaption`**: Dense structural captioning — now uses `source_caption` (the source dataset's own label, when one exists) as a prior hint.
11. **`s05_ocr_enrichment`**: DeepSeek-OCR text extraction — a distinct, separately-registered stage from `s05_recaption`, gated by `s05_recaption`'s `ocr_enrichment` config flag but requiring its own `stages.s05_ocr_enrichment` block to actually run.
12. **`s05_5_pii_text_redact`**: Redacts sensitive text found by the OCR pass.
13. **`s06_structure`**: UI component tree and bounding box JSON extraction.
14. **`s07_routing`**: Domain tagging and stratified shard routing.
15. **`s07_5_edit_pairs`**: *(New)* Synthetic (rough sketch, target) paired data for Qwen-Image-Edit-2511's actual edit-conditioned task — plus MagicBrush/InstructPix2Pix ingestion for general-domain task-shape grounding.
16. **`s08_encoding`**: **Tri-Path Latent Encoding** (Z-Image VAE, Qwen-Image-Edit-2511 VAE, VQ Tokens — now `ui_first`-domain-gated, Control Maps).
17. **`s09_heldout`**: Stratified evaluation set carve-out — now rejects encoding-incomplete records via a domain-aware completeness check before admitting them.
18. **`s10_audit`**: Automated VLM-as-Judge rubric evaluation (replaces human spot-checks).
19. **`s10_5_critic_preference`**: Gemma 4 31B critique generation, seeding `/ui_critique/*.parquet` (self-generated signal — see `s01_5_uicrit_join` above for the real human-calibration source).
20. **`s11_registry_watcher`**: Automated polling for new SOTA model releases.
21. **`s12_model_data_export`**: *(New)* Segments the fully-processed manifest into `model_data/`, one clean folder per PRD-trainable component.

### 💾 Tri-Path Latent Storage Strategy
To support both the discrete Sketch Tier and continuous Polish Tier, the Data-Forge encodes every approved image into four distinct representations:
1. `latents_zimage/` (fp16 `.safetensors`) - Continuous latents for Z-Image-Turbo.
2. `latents_qwenimage/` (fp16 `.safetensors`) - Continuous latents for Qwen-Image-Edit-2511.
3. `vq_tokens_sketch/` (int16 `.pt`) - Discrete codebook indices for MaskGIT/MAR.
4. `control_tokens/` (`.json`) - Structural bounding boxes and Canny edges for handoff conditioning.

Two additional Parquet outputs support alignment training:
5. `ui_critique/` (`.parquet`) - Scored critiques (UICrit-rubric + Gemma-4-generated), the DPO training signal input.
6. `preference_pairs/` (`.parquet`) - Reserved for ranked A/B DPO pairs once the product's own candidate-sampling loop exists (out of data-forge's scope — see `s10_5_critic_preference.py`'s docstring).

---

## 💻 Hardware & Infrastructure Target

Krisna is explicitly designed to run on a **Single 48GB Workstation GPU** (e.g., NVIDIA RTX A6000 / Ada 6000) — evaluated against and chosen over a dual-machine or NVIDIA DGX Spark topology; see the project's PRD for the full comparison (Spark's 273 GB/s bandwidth vs. the A6000's 768 GB/s would tax every conversational turn under a single-machine constraint).

* **Inference Engine**: vLLM with dynamic model swapping (Tier-1 <-> Tier-2 <-> OCR <-> Critic) and strict `psutil` zombie-process teardown.
* **Orchestration**: Custom Python DAG with SQLite WAL-mode manifest for ACID-compliant state tracking across millions of records.
* **OS**: Windows 11 / Ubuntu 22.04 (Fully cross-platform via `pathlib` and `spawn` multiprocessing guards).

## 🔧 Known Fixes in This Revision

**v15 — data completeness (see `data-forge/docs/DATA_COMPLETENESS.md` for the full model-to-data trace):** three datasets (UICrit, Screen2Words, and — from v14 — PD12M/CC12M) were silently ingesting zero usable records due to structurally similar fetch-path assumptions; the Planner and Qwen-Image-Edit-2511 had incomplete or entirely missing training data despite the model stack itself being finalized. All fixed — see that doc for the component-by-component before/after.

**v14 fixes:**

A thorough pass found and fixed several bugs, some severe enough to silently break core guarantees of the pipeline. Full detail lives in code comments at each fix site; summarized here for visibility:

- **PII protection was completely dead.** `s05_ocr_enrichment` had no config entry, so it silently defaulted to disabled — OCR never ran, and the "critical" text-PII redaction stage downstream never matched a single record. Fixed.
- **Tri-Path Encoding crashed on first use.** `qwen_image_vae` pointed at `Qwen/Qwen-Image-2.0-VAE`, a repo that was never published. `load_encoders()` had no per-encoder error handling, so this took down the entire pipeline run on the first chunk that reached Stage 8. Fixed — swapped to Qwen-Image-Edit-2511 and made encoder loading resilient per-model.
- **PD12M/CC12M (≈25M records, half the corpus) were never actually ingested.** They're metadata-only (parquet/TSV + external URLs); the fetcher only downloaded files and globbed for image extensions, silently finding zero images for both. Fixed with a real URL-list download path.
- **The `/ui_critique/` and `/preference_pairs/` outputs described throughout the docs and configs were never implemented** — `pandas`/`pyarrow` weren't even dependencies. Fixed, plus the new Critic Tier stage that actually populates `/ui_critique/`.
- **Several silent data-loss and correctness bugs**: a full-table manifest scan repeated per dataset (real perf problem at this pipeline's target scale), a Windows MAX_PATH check that used an arbitrary threshold instead of the actual worst-case path length, dead FAISS config, an unclosed `httpx.AsyncClient` leaking connections on every registry check, and three real UI datasets (`enrico`, `screen2words`, `uicrit`) missing from the domain tagger's source list, silently falling through to a fragile heuristic instead of being correctly tagged.
- **One near-miss, corrected before shipping**: an initial attempt to "fix" `ShardRouter`'s ratio enforcement (backfilling a domain's shortfall from the other's surplus) was reverted after it turned out to defeat the actual purpose of `ui_first_ratio` — it silently skewed the output composition away from the configured target instead of honoring it. The original per-domain-cap behavior, which accepts a smaller total routed set rather than a skewed one, is correct and is what `tests/test_shard_router.py` already locks in.

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
Before processing terabytes of data, validate that your YAML configurations (`configs/pipeline.yaml`, `configs/models.yaml`) are well-formed and internally consistent (dataset keys referenced by stages actually exist, encoder keys referenced by Stage 8 actually exist, etc.) — most of this pipeline's configuration is validated via plain Python dataclasses, not Pydantic; only the VLM structured-output schemas (`inference/structured_output.py`) use Pydantic.

```powershell
python scripts/verify_schemas.py
```

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

