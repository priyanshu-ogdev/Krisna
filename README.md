<div align="center">
<img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg?style=for-the-badge" />
<img src="https://img.shields.io/badge/Stack-Python%20%7C%20vLLM%20%7C%20Stable%20Diffusion%20%7C%20ONNX-blue.svg?style=for-the-badge" />
<img src="https://img.shields.io/badge/AI-Qwen%20%7C%20Z-Image%20%7C%20Gemma%204%20%7C%20YOLOv8-orange.svg?style=for-the-badge" />
</div>

# Krisna: Agentic AI Design System with Zero-Touch Data Pipeline

**Krisna** is a production-ready, single-GPU (48GB VRAM) conversational AI design agent capable of generating, iterating, and polishing UI/graphic designs through human-in-the-loop interaction. It utilizes a novel **Two-Tier Generation Architecture** (Sparse Masked Sketching → Continuous Flow Polishing), an on-demand **Critic Tier** for VL-judged design review, and operates under a fully automated, zero-touch **Data-Forge** ingestion pipeline that processes millions of images into training-ready latents on a single workstation.

Unlike cloud-dependent design agents, Krisna executes entirely on a single 48GB workstation GPU (RTX A6000 / Ada 6000), with dynamic model swapping vLLM engine ensuring only the active tier consumes VRAM at any given time.

---

## ⚡ Architecture Overview

```
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
   │ Tier C: Critic (on-demand)          │ ◄── Gemma 4 31B Dense (NF4)
   │ • Judges the finished render only   │     Apache 2.0, QLoRA via Unsloth
   │ • Never resident during conversation│
   └──────────────────────────────────────┘
```

---

## 🏗️ The Model Stack

| Component | Model Family | Role | Precision / Serving |
| :--- | :--- | :--- | :--- |
| **Planner / Critic** | Qwen3.5-9B (or 4B) | Conversational state management, design critique, tool routing. | BF16 (Interactive) / FP8 (Serving) |
| **Sketch Tier** | MaskGIT / MaskGIL (Custom) | Sparse, iterative layout and structural exploration. | Discrete VQ Tokens (INT16) |
| **Polish Tier (Primary)** | Z-Image-Turbo (6B) | Fast, high-fidelity final rendering and texture synthesis. | NF4 / AWQ (vLLM) |
| **Polish Tier (Quality)** | Qwen-Image-Edit-2511 (20B) | Complex semantic rendering, treats Stage-1 output as an edit target. | NF4 / AWQ (vLLM) |
| **Critic Tier (on-demand)** | Gemma 4 31B Dense | UI/UX critique scoring, seeds `/ui_critique/` for DPO. | NF4 (Apache 2.0, QLoRA via Unsloth) |

---

## 🛠️ The Data-Forge: Zero-Touch Ingestion Pipeline

The Data-Forge is a custom, 21-stage Python orchestrator designed to process millions of images into training-ready latents on a single 48GB workstation GPU. It replaces manual curation with **VLM-as-Judge** auditing and deterministic logic.

### Pipeline Stages

1. **`s00_manifest_planning`**: Pre-flight storage checks, registry watcher sync.
2. **`s01_fetch`**: HuggingFace/GitHub ingestion + Inline License Verification Agent. Includes dedicated URL-list download path for metadata-only sources (PD12M/CC12M).
3. **`s01_5_uicrit_join`**: Joins UICrit's real human critique/ratings against already-ingested RICO records.
4. **`s01_6_planner_synthesis`**: Generates the Planner's (Qwen3.5-9B) conversational SFT training data from real UICrit critique text.
5. **`s02_dedup`**: GPU-accelerated FAISS semantic near-duplicate removal.
6. **`s03_quality`**: Aesthetic and resolution scoring via Tier-1 VLM.
7. **`s03_5_pii_scrub`**: Regex text redaction + MediaPipe face blurring for UI screenshots.
8. **`s04_safety`**: NSFW/Harmful content classification.
9. **`s04_5_escalation`**: Tier-2 VLM second-opinion on borderline records.
10. **`s05_recaption`**: Dense structural captioning with `source_caption` prior hint.
11. **`s05_ocr_enrichment`**: DeepSeek-OCR text extraction (gated by config flag).
12. **`s05_5_pii_text_redact`**: Redacts sensitive text found by the OCR pass.
13. **`s06_structure`**: UI component tree and bounding box JSON extraction.
14. **`s07_routing`**: Domain tagging and stratified shard routing.
15. **`s07_5_edit_pairs`**: Synthetic (rough sketch, target) paired data for Qwen-Image-Edit-2511.
16. **`s08_encoding`**: **Tri-Path Latent Encoding** (Z-Image VAE, Qwen-Image-Edit-2511 VAE, VQ Tokens — domain-gated, Control Maps).
17. **`s09_heldout`**: Stratified evaluation set carve-out with domain-aware completeness check.
18. **`s10_audit`**: Automated VLM-as-Judge rubric evaluation.
19. **`s10_5_critic_preference`**: Gemma 4 31B critique generation, seeding `/ui_critique/*.parquet` (self-generated signal).
20. **`s11_registry_watcher`**: Automated polling for new SOTA model releases.
21. **`s12_model_data_export`**: Segments fully-processed manifest into `model_data/`, one clean folder per PRD-trainable component.

### 💾 Tri-Path Latent Storage Strategy

To support both the discrete Sketch Tier and continuous Polish Tier, the Data-Forge encodes every approved image into four distinct representations:
1. `latents_zimage/` (fp16 `.safetensors`) - Continuous latents for Z-Image-Turbo.
2. `latents_qwenimage/` (fp16 `.safetensors`) - Continuous latents for Qwen-Image-Edit-2511.
3. `vq_tokens_sketch/` (int16 `.pt`) - Discrete codebook indices for MaskGIT/MAR.
4. `control_tokens/` (`.json`) - Structural bounding boxes and Canny edges for handoff conditioning.

Two additional Parquet outputs support alignment training:
5. `ui_critique/` (`.parquet`) - Scored critiques (UICrit-rubric + Gemma-4-generated), the DPO training signal input.
6. `preference_pairs/` (`.parquet`) - Reserved for ranked A/B DPO pairs.

---

## 💻 Hardware & Infrastructure

- **Target GPU**: Single 48GB workstation GPU (RTX A6000 / Ada 6000)
- **Inference Engine**: vLLM with dynamic model swapping (Tier-1 ↔ Tier-2 ↔ OCR ↔ Critic)
- **Orchestration**: Custom Python DAG with SQLite WAL-mode manifest (ACID-compliant)
- **OS**: Windows 11 / Ubuntu 22.04 (cross-platform via `pathlib` + `spawn` multiprocessing guards)

### Key Fixes in This Revision (v15)

- **PII protection was completely dead**: `s05_ocr_enrichment` had no config entry, so OCR never ran and text-PII redaction was silently disabled. Fixed.
- **Tri-Path Encoding crashed on first use**: `qwen_image_vae` pointed to never-published Qwen-Image-2.0-VAE. Swapped to Qwen-Image-Edit-2511 with per-encoder error handling. Fixed.
- **PD12M/CC12M (≈25M records) were never actually ingested**: Metadata-only sources (parquet/TSV + external URLs). Fixed with real URL-list download path.
- **3 datasets silently ingesting zero records**: UICrit, Screen2Words, PD12M/CC12M fixed via URL-list path and corrected fetch assumptions.

---

## 🚀 Setup & Execution

### 1. Environment Bootstrap
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_env.ps1
conda activate krisna-forge
```

### 2. Schema Verification
```powershell
python scripts/verify_schemas.py
```

### 3. Dry-Run Validation
```powershell
$env:DATA_ROOT="D:\kf_data"
python -m data_forge.cli run --dry-run
```

### 4. Smoke Test (100 records)
```powershell
$env:HF_TOKEN="your_huggingface_token"
python -m data_forge.cli run --chunk-size 100 --limit 100
```

### 5. Launch the Agentic Server
```bash
python -m krisna.serve --planner qwen3.5-9b --renderer z-image-turbo
```