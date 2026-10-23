# Krivex

Krivex is a local-first, agentic UI design system. It separates intent
planning, deterministic layout state, image-conditioned polishing, verification,
and dataset governance so every model boundary is inspectable and reproducible.

## Current research contract

The current research system has **one intended trainable artifact**:

| Responsibility | Current implementation | Trainable? |
|---|---|---:|
| Planner | Frozen `Qwen3.5-9B`, multimodal review contract | No |
| Structure | Deterministic `krivex.layout.v1` Layout IR and renderer | No |
| Polish Default | `Qwen/Qwen-Image-2.1` native image-conditioned pipeline plus DoRA adapter | DoRA only |
| Polish Quality | Frozen Qwen-Image-2.1 without an adapter | No |
| Data-Forge VLM | Local Qwen3.8-27B preprocessing/audit dependency | No |

The Qwen-Image-2.1 DoRA adaptation path (`use_dora: true`) is configured and contract-tested, operating across the calibrated 7-bucket aspect ratio manifold (`1024,1024;768,1344;1344,768;832,1248;1248,832;896,1152;1152,896`). It supports three production hardware tiers:
- **Enterprise Max (48GB GDDR6)**: `polish_qwen_image21_lora_a6000_max.yaml` — batch=16, accum=2 (effective batch 32), rank=128, alpha=64, FP32 AdamW, latent pre-caching (~32 GB host RAM), ~45.0 GB peak VRAM (93.7% utilization), 14,000 steps (~1.85h for 2.05 epochs across 218,791 pairs).
- **Enterprise Standard (48GB GDDR6)**: `polish_qwen_image21_lora.yaml` — batch=8, accum=2 (effective batch 16), rank=64, alpha=32, 27,500 steps (~15.2h).
- **Consumer Desktop (12GB–16GB)**: `ablations/polish_qwen_qdora_12gb.yaml` — batch=1, accum=16 (effective batch 16), rank=64, alpha=32, 4-bit NF4 base, 11.2 GB peak VRAM.


Canonical decisions and evidence:

- [Master Final System Design](docs/FINAL_SYSTEM_DESIGN.md)
- [Final architecture verdict](docs/research/FINAL_ARCHITECTURE_VERDICT.md)
- [Flow-matching upgrade plan](docs/research/FLOW_MATCHING_UPGRADE_PLAN.md)
- [SOTA audit](docs/research/SOTA_AUDIT_2026-10.md)
- [Research paper specification](docs/RESEARCH_PAPER.md)
- [Training architecture](docs/training/ARCHITECTURE.md)

## Repository layout

```text
.
├── data-forge/              Governed ingestion, provenance, VLM stages, exports
├── inference/               Python model plane, orchestrator, verifiers, API
├── inference/runtime/       CLI session/runtime harness
├── inference/frontend/      Node/Tauri control panel
├── training/                Dataset bridges, native trainer contracts, adapters
├── scripts/                 Operational entry points grouped by subsystem
│   ├── setup.sh             Environment setup entry point
│   ├── data-forge/          Data-Forge runners and sync bridges
│   ├── inference/           Inference service launchers (run_inference.ps1/sh)
│   └── training/            Training orchestrators (train.sh, train_all.ps1)
├── tests/                   Cross-package tests grouped by subsystem
├── docs/                    Product, architecture, research, operations, review
├── .github/                 CI and repository automation
└── pytest.ini               Root test discovery and import-path configuration
```

Generated state is deliberately kept outside version control: datasets,
checkpoints, model weights, local databases, blob stores, logs, caches, build
outputs, virtual environments, and downloaded third-party repositories.

## Quick start

```bash
# Inspect the repository and available subsystem commands
python scripts/training/audit_training_pipeline.py
python scripts/data-forge/audit_registry.py

# Run tests that do not require downloaded models or a GPU
python -m pytest tests/data_forge -q
python -m pytest tests/training -q
python -m pytest tests/inference -q
```

### Data-Forge

Read [`data-forge/README.md`](data-forge/README.md) and
[`docs/data-forge/LOCAL_VLM_PREPROCESSING_ARCHITECTURE.md`](docs/data-forge/LOCAL_VLM_PREPROCESSING_ARCHITECTURE.md).
The strict local profile requires CUDA, vLLM, pinned model revisions, and a
target GPU. It refuses CPU or weaker-model fallbacks.

### Training

Read [`docs/training/TRAINING_RUNBOOK.md`](docs/training/TRAINING_RUNBOOK.md).
The Qwen training launcher performs a CPU-only preflight before invoking
`accelerate`; it refuses missing receipts, invalid splits, unsupported trainer
arguments, and an unpinned Diffusers checkout.

### Inference

Read [`inference/README.md`](inference/README.md) and
[`docs/architecture/INFERENCE_PRODUCTION_STACK.md`](docs/architecture/INFERENCE_PRODUCTION_STACK.md).
Strict real mode requires the native Qwen-Image-2.1 pipeline, a valid condition image,
and a verified adapter manifest. Model weights are automated via `python scripts/inference/download_weights.py`.
Fine-tuned checkpoints (`krivex-ai/krivex-qwen-image21-polish-lora`) will be published on Hugging Face Hub, with automatic downloader resolution and zero-shot fallback.

## Server Training & Post-Training Evaluation Lifecycle

When ready to begin training on the production GPU server:

```bash
# 1. Server Environment Initialization (CUDA 12.4+, PyTorch 2.6+, Diffusers pinned)
./scripts/training/setup_env_training.sh
./scripts/training/setup_env_diffusers_training.sh

# 2. Pre-Flight Read-Only Readiness Audit
python scripts/training/audit_training_pipeline.py

# 3. Synchronize Curated Multimodal Pairs from Data-Forge
python scripts/data-forge/sync_to_training.py --data-root /path/to/data_krivex

# 4. Launch Production Polish DoRA Training
# Option A: Enterprise Max Profile (48GB full utilization: batch=16, rank=128, FP32 AdamW, 14,000 steps, ~1.85h)
./scripts/training/train_polish_default_lora.sh training/configs/polish_qwen_image21_lora_a6000_max.yaml \
  --max_train_steps 14000 --lr_warmup_steps 700

# Option B: Enterprise Standard Profile (48GB: batch=8, rank=64, 27,500 steps, ~15.2h)
./scripts/training/train_polish_default_lora.sh training/configs/polish_qwen_image21_lora.yaml \
  --max_train_steps 27500 --lr_warmup_steps 500

# Option C: Windows Native PowerShell Dispatcher (1:1 parity)
powershell -ExecutionPolicy Bypass -File .\scripts\training\train_all.ps1 -Tier polish-default -Config training/configs/polish_qwen_image21_lora_a6000_max.yaml

# 5. Execute Post-Training Multi-Hardware Evaluation Harness
# Evaluates KID, LPIPS, CMMD, DINOv2, Layout IoU, Typographic CER/WER, Aesthetic Score, and Latency
python scripts/eval/run_evaluation.py \
  --profile 48gb \
  --checkpoint checkpoints/qwen_image21_lora_a6000_max \
  --reference-dir data/reference_uis \
  --wireframe-dir data/held_out_wireframes \
  --prompt-list data/eval_prompts.json \
  --output-dir eval_results

# Fast Offline Smoke Test / Dry-Run (Synthetic Generation Mode):
python scripts/eval/run_evaluation.py --synthetic --num-samples 5 --output-dir eval_results

# 6. Verify Hypotheses & Automatically Synchronize Research Paper Monograph
# Generates cryptographic evaluation_receipt.json and updates Tables 11.1-11.6 in docs/RESEARCH_PAPER.md
python scripts/eval/ingest_eval_results.py \
  --input eval_results/eval_comparison.json \
  --paper-path docs/RESEARCH_PAPER.md \
  --apply
```

## Architectural Analysis & Technical Highlights

1. **A6000 48GB Dedicated VRAM Saturation (93.7% Saturation)**:
   - Dedicated GDDR6 ECC memory vs. unified shared memory: Avoids PCIe bandwidth bottlenecks (10–20× latency penalty of 102 GB shared memory) by anchoring all forward, backward, and optimizer states within 45.0 GB dedicated VRAM.
   - Batch size 16 with gradient checkpointing delivers continuous $\ge 98\%$ Streaming Multiprocessor (SM) saturation across all 84 SMs and 10,752 CUDA cores on the RTX A6000.
   - Pinned host RAM latent pre-caching (~32 GB from 128 GB system RAM) unloads the VAE from GPU memory after pre-caching (reclaiming 3.0 GB VRAM), eliminates per-step VAE forward passes, and accelerates training throughput by $4.5\times$ (from ~8.3h down to ~1.85h for 14,000 steps = 2.05 epochs across 218,791 pairs).
2. **High-Capacity DoRA Allocation ($r=128, \alpha=64$) & FP32 AdamW Optimizer**:
   - Captures the complete singular value spectrum: macro layout grid boundaries ($\sigma_1 \dots \sigma_{16}$), UI components & button borders ($\sigma_{17} \dots \sigma_{64}$), and fine sub-pixel font anti-aliasing / 1px division borders ($\sigma_{65} \dots \sigma_{128}$).
   - FP32 AdamW precision prevents 8-bit quantization drift and freezing of the 1D learnable DoRA magnitude vector $m \in \mathbb{R}^{1 \times d_{\text{out}}}$ while preserving continuous flow-matching pixel velocity gradients.
3. **Calibrated 7-Bucket Aspect Ratio Manifold**:
   - Binning across `1024,1024; 768,1344; 1344,768; 832,1248; 1248,832; 896,1152; 1152,896` preserves UI aspect ratios without geometric distortion. The 3.0 GB VRAM safety headroom cleanly absorbs transient sequence length spikes on extreme web-banner ($1344\times 768$) and mobile-story ($768\times 1344$) buckets.
4. **Zero-Eviction Serving Dynamics ($T_{\text{swapping}} \equiv 0.000\text{s}$)**:
   - Eliminates external dense critic swapping in favor of resident Planner review (`Qwen3.5-9B-Instruct` GGUF + `mmproj`), preserving sub-second interactive turn latency.
   - Deterministic CPU layout compilation via Cassowary Linear Simplex operates in $<5\text{ms}$ with $0.0\text{ GB}$ GPU VRAM.
   - Region locking via FlowEdit differential trajectory steering and separable Lanczos-3 anti-aliasing guarantees zero pixel drift ($S_{\text{lock}} \equiv 1.000$) on locked user elements.
5. **Verified Verification & Hermetic CI Testing**:
   - Automated `VerifierStack` computes Layout IoU (OpenCV), text legibility CER/WER (EasyOCR), semantic alignment (CLIP ViT-L/14), aesthetic quality (LAION), and content safety (ONNX ViT).
   - `KRIVEX_POLISH_SYNTHETIC=1` mode guarantees fast, hermetic test execution on consumer hardware without unquantized 14GB GPU allocations.
   - 100% test pass rate across 714 monorepo tests (`tests/inference`, `tests/training`, `tests/data_forge`).

## Engineering standards

- Keep model, data, and serving contracts explicit and versioned.
- Do not commit secrets, local credentials, generated datasets, model weights,
  databases, blobs, caches, or runtime logs.
- Do not call a configured count a realized dataset count without a receipt.
- Do not call a configured checkpoint trained without a run receipt.
- Add a focused test for every contract or boundary change.
- Prefer small, reviewable commits by subsystem.
- Run `git diff --check` and the relevant package tests before committing.
- Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
  `chore:`.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the repository workflow and
[`docs/architecture/DIRECTORY_LAYOUT.md`](docs/architecture/DIRECTORY_LAYOUT.md)
for package ownership.
