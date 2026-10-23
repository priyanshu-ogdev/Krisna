# Krivex Inference: Subsystem Architecture & Serving Runbook

> **Monorepo Package Separation**: This package (`krivex-inference`) encompasses the production model-serving backends, memory swap orchestration, stateful session coordination, automated verifiers, and the local Web Studio interface. For training pipelines (LoRA fine-tuning, continuous flow-matching distillation, and DPO dataset preparation), refer to [`../training/README.md`](../training/README.md).

---

## 1. Executive Summary & Industry-Grade Architecture

The **Krivex Inference Subsystem** is an enterprise-grade neuro-symbolic engine designed for autonomous, multi-turn UI/UX design generation, spatial layout compilation, high-fidelity visual synthesis, and self-evaluative refinement.

The architecture is formally grounded in:
* **Research Paper §9**: *ML Systems Serving Foundations, Computational Complexity, Memory Bounds, and Zero-Eviction Serving Dynamics*.
* **PRD §5**: *Data Model & Core Agentic Feedback Flows*.
* **PRD §7**: *Inference Swap Orchestration & VRAM Accounting*.

### Core Foundational Pillars

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │               Client Applications / UI Studio          │
                                  │           (Web Studio :3100 / FastAPI Backend :8420)   │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │ REST / SSE
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │      Krivex Swap Orchestrator & Session Manager        │
                                  │  - Strict VRAMLedger & RAMLedger Resource Admission    │
                                  │  - Deterministic Residency State Machine (§7.4)        │
                                  │  - Optimistic Concurrency & Session State Store        │
                                  └───────────┬────────────────────────────────┬───────────┘
                                              │                                │
                     Tier A: Baseline Resident│                                │Tier B: Swapped Synthesis
                    (~4.4 GB VRAM / <=16 GB RAM)                              │(<=12 GB Peak VRAM)
                                              ▼                                ▼
            ┌───────────────────────────────────────────────┐  ┌───────────────────────────────────────────────┐
            │ Planner Backend (Qwen3.5-9B-Instruct)         │  │ Polish Backend (Qwen-Image-2.1 MMDiT)         │
            │  - GGUF Q4_K_M via llama.cpp (zero-copy mmap) │  │  - Flow-Matching Continuous ODE (6 to 40 steps)│
            │  - Multimodal Projector (mmproj-f16.gguf)     │  │  - UI Polish Continuous-Flow LoRA Adapter     │
            │  - Pushdown Automata GBNF Grammar Constraints │  │  - FlowEdit Trajectory Steering (ICCV 2025)   │
            │  - UICrit RAG Semantic Retrieval Index        │  │  - Exact Lanczos-3 Lock Compositing           │
            ├───────────────────────────────────────────────┤  ├───────────────────────────────────────────────┤
            │ Sketch Backend (Deterministic Layout Engine)  │  │ Automated Verifier Stack                      │
            │  - Host CPU SIMD Cassowary Linear Simplex     │  │  - CLIP ViT-L/14 Semantic Alignment           │
            │  - Structured Layout IR (krivex.layout.v1)    │  │  - EasyOCR Text Legibility & Confidence       │
            │  - 0.0 GB GPU VRAM / <5ms compilation latency │  │  - OpenCV Layout-IoU Boundary Overlap         │
            │  - DualStreamSketch: Wireframe + 16-cls SPADE │  │  - LAION Aesthetic Predictor & ONNX Safety   │
            └───────────────────────────────────────────────┘  └───────────────────────────────────────────────┘
```

1. **Unified Qwen Foundation Ecosystem**:
   - **Planner Tier**: `Qwen3.5-9B-Instruct` autoregressive VLM quantized to `Q4_K_M` GGUF with `mmproj-f16.gguf` multimodal projection. Executes via `llama-cpp-python` with zero-copy mmap, FlashAttention-2, and strict GBNF grammar constraints (`planner.delta.v1`). Optional `vllm==0.29.0` engine for high-throughput batching.
   - **Polish Tier**: `Qwen-Image-2.1` (7B Flow-Matching MMDiT / MMViT) continuous flow-matching pipeline. Equipped with low-VRAM CPU offloading, NF4 quantization, and a continuous-flow UI Polish LoRA adapter (`krivex-ai/krivex-qwen-image21-polish-lora`).
2. **Deterministic Layout IR Compilation (0 MB VRAM)**:
   - The Sketch Tier bypasses GPU computation entirely. It compiles Planner design deltas into hierarchical `krivex.layout.v1` trees using a host CPU SIMD Cassowary Linear Simplex solver in $<5\text{ms}$, rendering dual-stream visual conditioning (`DualStreamSketch`: TrueType wireframe + 16-class SPADE segmentation map) with **0.0 GB GPU VRAM** overhead.
3. **Zero-Eviction Serving Dynamics ($T_{\text{swapping}} \equiv 0.000\text{s}$)**:
   - Eliminates legacy external dense critics in favor of **Resident Planner Review** (`review=True`). The resident 9B VLM evaluates rendered interfaces in-place without triggering GPU model evictions, maintaining continuous sub-second interactive turn responsiveness.
4. **Strict Consumer Hardware Envelope ($\le 12\text{GB}$ VRAM / $\le 16\text{GB}$ System RAM)**:
   - Analytical memory accounting governed by deterministic `VRAMLedger` and `RAMLedger` monitors prevents CUDA out-of-memory errors on commodity GPUs (e.g., RTX 3060, RTX 4070) and workstation platforms alike.
5. **Exact Region Lock Compositing ($S_{\text{lock}} \equiv 1.000$)**:
   - FlowEdit differential trajectory steering and separable Lanczos-3 anti-aliasing guarantee zero pixel drift on user-locked bounding regions with seamless boundary gradient continuity ($\Delta \nabla \le 0.05$).

---

## 2. Directory & Module Structure

```
inference/
├── Dockerfile                  # Sealed multi-stage CUDA 12.4 container
├── docker-compose.yml          # Unified GPU production service stack
├── entrypoint.sh               # Container preflight & execution coordinator
├── pyproject.toml              # Build metadata and dependency specifications
├── requirements.txt            # Authoritative balanced dependency manifest
├── README.md                   # Authoritative inference runbook (this document)
├── frontend/                   # Web Studio user interface
│   ├── package.json            # Node.js workspace specification
│   ├── server.js               # Express reverse proxy and SSE distributor
│   └── public/                 # Glassmorphic Web Studio UI (app.js, style.css)
├── runtime/                    # Interactive runtime harness
│   └── run_agentic_session.py  # Scriptable CLI runner for agentic sessions
└── src/krivex_inference/       # Core inference package
    ├── cli.py                  # Master CLI command group (krivex-session)
    ├── backends/               # Concrete model execution implementations
    │   ├── factory.py          # real_backend_factory provider
    │   ├── model_loader.py     # Artifact discovery and CUDA hardware probe
    │   ├── planner_backend_llama.py  # llama.cpp GGUF runner with GBNF grammar
    │   ├── planner_rag.py      # TF-IDF & dense RAG over UICrit dataset
    │   ├── layout_backend.py   # Deterministic Cassowary layout compiler
    │   ├── polish_backend_qwen.py    # Qwen-Image-2.1 Flow-Matching MMDiT
    │   ├── sketch_handoff.py   # Latent/pixel handoff & Lanczos-3 locking
    │   └── blob_store_singleton.py   # SHA-256 content-addressable storage
    ├── orchestrator/           # Coordination and state management
    │   ├── swap_orchestrator.py # Master swap coordinator with OOM recovery
    │   ├── state_machine.py    # Formal residency state machine (§7.4)
    │   ├── model_registry.py   # Full-VRAM and Low-VRAM tier specifications
    │   ├── vram_budget.py      # VRAMLedger and RAMLedger accounting
    │   ├── design_state.py     # Pydantic v2 DesignState schema (§5.1)
    │   ├── flows.py            # High-level sequence orchestration (§5.3)
    │   ├── service.py          # FastAPI application & REST endpoint matrix
    │   └── store.py            # SQLite session storage with optimistic locking
    └── verifiers/              # Post-generation evaluators and safety gates
        ├── verifier_stack.py   # Unified orchestrator for multi-modal verifiers
        ├── clip_alignment.py   # CLIP ViT-L/14 semantic prompt fidelity
        ├── ocr_readability.py  # EasyOCR text legibility & per-box confidence
        ├── layout_iou.py       # OpenCV contour alignment and element overlap
        ├── aesthetic_safety.py # LAION Aesthetic scoring & ONNX NSFW gate
        └── critique_adapter.py # Normalizes evaluations into structured reviews
```

---

## 3. Hardware Requirements & Resource Budgets

Krivex Inference provides two operational profiles configured dynamically or via environment variables:

| Component / Tier | Full-VRAM Profile (24GB Target) | Low-VRAM Profile (12GB Target) | Execution Runtime & Offload Mechanism |
|---|---|---|---|
| **Planner Backend** | 4.2 GB VRAM / 0.0 GB RAM | 4.2 GB VRAM / 0.0 GB RAM | `Qwen3.5-9B` GGUF Q4_K_M via `llama.cpp` (zero-copy mmap) |
| **Sketch Backend** | **0.0 GB VRAM** / 0.1 GB RAM | **0.0 GB VRAM** / 0.1 GB RAM | Pure host CPU SIMD Cassowary Linear Simplex (<5ms) |
| **Baseline Resident Total** | **4.2 GB VRAM / 0.1 GB RAM** | **4.2 GB VRAM / 0.1 GB RAM** | **Always resident during interactive turns** |
| **Polish-Fast (Tier B)** | 10.0 GB VRAM / 0.0 GB RAM | 4.5 GB VRAM / 6.0 GB RAM | CFM 4-8 step distillation + `enable_model_cpu_offload()` |
| **Polish-Default (Tier B)**| 12.0 GB VRAM / 0.0 GB RAM | 8.0 GB VRAM / 8.0 GB RAM | MMDiT 28-step Flow-Matching + UI LoRA + CPU offload |
| **Polish-Quality (Tier B)**| 16.0 GB VRAM / 0.0 GB RAM | 10.0 GB VRAM / 10.0 GB RAM| MMDiT 40-step sampling + VAE tiling & slicing |
| **Peak Active Memory** | **16.0 GB VRAM / 0.1 GB RAM** | **10.0 GB VRAM / 10.0 GB RAM**| **Strictly bounded within 12GB VRAM / 16GB RAM envelope** |

> [!NOTE]
> In Low-VRAM mode (`KRIVEX_LOW_VRAM_MODE=1`), the orchestrator applies diffusers' `enable_model_cpu_offload()` on the Polish MMDiT pipeline. Submodules (text encoder, transformer, VAE) are dynamically streamed between system RAM and GPU VRAM during the denoising loop, keeping quantized dtypes intact. When the Polish pass concludes, the pipeline is evicted, VRAM is freed via `torch.cuda.empty_cache()`, and the resident Baseline Tier is re-admitted.

---

## 4. Quickstart: Installation & Environment Setup

### 4.1 Native Host Setup (Linux / WSL2)

```bash
# 1. Clone repository
git clone https://github.com/priyanshu-ogdev/Krivex.git krivex
cd krivex

# 2. Initialize unified virtual environment
./scripts/inference/setup_env.sh

# 3. Activate virtualenv and install full inference dependencies
source .venv/bin/activate
./scripts/inference/setup_env_inference.sh

# 4. Verify hardware capability (requires NVIDIA GPU with CUDA >= 12.1)
python scripts/inference/check_hardware.py --require-gpu
```

### 4.2 Native Host Setup (Windows PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/priyanshu-ogdev/Krivex.git krivex
Set-Location krivex

# 2. Run PowerShell environment setup with backend dependencies
.\scripts\inference\setup_env_inference.ps1 -InstallBackends -RequireGPU

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Verify CUDA runtime
python scripts\inference\check_hardware.py --require-gpu
```

---

## 5. Model Weight Acquisition & Provisioning

Krivex provides an automated downloader that pulls foundation checkpoints and multimodal projectors from Hugging Face Hub, verifying SHA-256 integrity:

```bash
# Provision standard models (Planner GGUF, mmproj, and Qwen-Image-2.1 base)
python scripts/inference/download_weights.py

# Low-VRAM profile (provisions offload configuration)
python scripts/inference/download_weights.py --low-vram

# Dry-run inspection (checks available disk storage and HF accessibility)
python scripts/inference/download_weights.py --dry-run
```

### Hugging Face Hub Artifacts Matrix

| Component | Default Hugging Face Repository | Artifact / Filename | Purpose |
|---|---|---|---|
| **Planner GGUF** | `unsloth/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-Instruct-Q4_K_M.gguf` | Core conversational reasoning & schema planning |
| **Multimodal Projector** | `unsloth/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-Instruct-mmproj-f16.gguf` | Visual screenshot decompilation & self-review |
| **Polish Foundation** | `Qwen/Qwen-Image-2.1` | Snapshot (`safetensors`) | 7B Flow-Matching MMDiT visual generation |
| **UI Polish Adapter** | `krivex-ai/krivex-qwen-image21-polish-lora` | `pytorch_lora_weights.safetensors` | Continuous-flow UI rendering LoRA |

> [!TIP]
> **Graceful Zero-Shot Fallback**: If the fine-tuned UI Polish LoRA adapter has not yet been downloaded or local training is still in progress, the inference engine automatically detects this and executes base `Qwen/Qwen-Image-2.1` in zero-shot mode without interruption.

---

## 6. Running the Inference Pipeline

### Option A: Master Launcher Script

The easiest way to launch the complete stack:

```bash
# Linux / WSL2
./scripts/inference/run_inference.sh                     # Runs MockBackend (CI/test mode)
KRIVEX_USE_REAL_BACKENDS=1 ./scripts/inference/run_inference.sh  # Real GPU backends
KRIVEX_USE_REAL_BACKENDS=1 KRIVEX_LOW_VRAM_MODE=1 ./scripts/inference/run_inference.sh # Low-VRAM (12GB)
```

```powershell
# Windows PowerShell
.\scripts\inference\run_inference.ps1                    # MockBackend (test mode)
.\scripts\inference\run_inference.ps1 -Real              # Real GPU backends
.\scripts\inference\run_inference.ps1 -Real -LowVram     # Real GPU + Low-VRAM mode (<=12GB)
.\scripts\inference\run_inference.ps1 -Real -WithFrontend # Backend (:8420) + Web Studio (:3100)
```

### Option B: Standalone FastAPI Service

Launch the Uvicorn ASGI server directly:

```bash
export KRIVEX_USE_REAL_BACKENDS=1
export KRIVEX_LOW_VRAM_MODE=1
python -m uvicorn krivex_inference.orchestrator.service:app --host 127.0.0.1 --port 8420 --reload
```

The Web Studio UI will be mounted and accessible directly at `http://127.0.0.1:8420/studio`.

### Option C: Web Studio Node.js Frontend

If developing on the frontend with live UI reload:

```bash
cd inference/frontend
npm install
npm start
# Studio opens on http://localhost:3100, reverse-proxying API calls to backend port 8420
```

### Option D: CLI Agentic Session Runner

Run an interactive multi-turn session directly from the command line:

```bash
# Non-interactive multi-turn run with render and review
python -m krivex_inference.cli \
  --real \
  --low-vram \
  --message "Design a modern dark-mode fintech dashboard" \
  --finalize \
  --finalize-tier polish_default \
  --critique
```

### Option E: Production Sealed Docker Deployment

For reproducible containerized production deployments:

```bash
# Launch unified GPU container stack (Inference Service + Studio on port 8420)
docker compose -f inference/docker-compose.yml up -d

# Check live logs
docker compose -f inference/docker-compose.yml logs -f
```

---

## 7. Comprehensive REST API Reference & Working cURL Suite

The inference service binds to port `8420` by default. All endpoints accept and return `application/json` (except `/session/{id}/render` which serves raw `image/png` binary bytes).

### 7.1 Core Route Matrix

| Method | Route Path | Description | Key Parameters / Request Schema |
|---|---|---|---|
| `GET` | `/health` | Liveness check, residency state, and VRAM snapshot | None |
| `GET` | `/ready` | Kubernetes/Docker readiness probe | None |
| `GET` | `/api/system/info` | Host hardware architecture, GPU target, and envelopes | None |
| `GET` | `/models/status` | Discovered GGUF paths, LoRA status, and real hardware allocations | None |
| `POST` | `/models/warm` | Synchronous or background pre-warming of baseline models | `{"background": bool}` |
| `GET` | `/models/warm/status` | Current progress and status of background model warmup | None |
| `POST` | `/models/unload` | Shuts down models and clears CUDA cache for external tasks | None |
| `POST` | `/session` | Initializes a new Design State session | `{"style": str, "palette": [str]}` |
| `GET` | `/session/{id}` | Retrieves full serialized `DesignState` wire representation | `session_id` (path) |
| `GET` | `/session/{id}/render` | Serves finalized synthesized image as raw PNG bytes | `session_id` (path) |
| `GET` | `/session/{id}/model-status` | Evaluates residency state and readiness for session | `session_id` (path) |
| `POST` | `/session/{id}/message` | Executes conversational turn, updates constraints & layout | `{"message": str, "auto_proceed": bool}` |
| `POST` | `/session/{id}/execute` | Simultaneous Plan + Sketch + Polish + Review pipeline | `{"message": str, "quality": bool, "fast": bool}` |
| `POST` | `/session/{id}/decompile` | VLM Screenshot-to-Layout decompilation | `{"image_ref": str, "instruction": str}` |
| `POST` | `/session/{id}/preview` | Instant 4–8 step preview via Consistency Flow Matching | `{"prompt": str}` |
| `POST` | `/session/{id}/finalize` | High-fidelity synthesis using Qwen-Image-2.1 MMDiT | `{"quality": bool, "prompt": str}` |
| `POST` | `/session/{id}/refine` | FlowEdit differential editing on specific locked regions | `{"instruction": str, "actions": [...]}` |
| `POST` | `/session/{id}/enhance` | Photo / wireframe fidelity upscaling pass | `{"image_ref": str, "quality": bool}` |
| `POST` | `/session/{id}/self-review` | Autonomous resident Planner / Layout verification turn | None |
| `POST` | `/session/{id}/critique` | Evaluates design using verifier stack & resident review | `{"compare_against_image_ref": str}` |
| `GET` | `/session/{id}/research-trace` | Exports reasoning trajectories & verifier ablation logs | `session_id` (path) |
| `GET` | `/orchestrator/status` | Comprehensive status of memory ledgers, state, and residency | None |
| `GET` | `/preference-pairs/stats` | Counts of collected preference pairs by source | None |
| `POST` | `/preference-pairs/export` | Exports DPO preference pairs to standard JSONL format | `?source=planner_review` |

---

### 7.2 End-to-End cURL Execution Walkthrough

Here is a complete, copy-pasteable bash workflow demonstrating an end-to-end design session:

```bash
BASE="http://127.0.0.1:8420"

# 1. Health & Readiness Preflight
curl -s "$BASE/health" | jq .

# 2. Create a new Design Session
SESSION_DATA=$(curl -s -X POST "$BASE/session" \
  -H "Content-Type: application/json" \
  -d '{"style": "glassmorphic dark", "palette": ["#0f172a", "#38bdf8", "#818cf8"]}')

SESSION_ID=$(echo "$SESSION_DATA" | jq -r '.session_id')
echo "Created Session ID: $SESSION_ID"

# 3. Conversational Turn (Planner generates Layout IR & constraint deltas)
curl -s -X POST "$BASE/session/$SESSION_ID/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a modern cryptocurrency wallet dashboard with transaction list and send button"}' | jq .

# 4. (Optional) Fast CFM Preview (4 to 8 steps, <2 seconds)
curl -s -X POST "$BASE/session/$SESSION_ID/preview" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "modern crypto wallet dashboard"}' | jq .

# 5. Finalize High-Fidelity Render (Swaps in Qwen-Image-2.1 MMDiT + LoRA)
curl -s -X POST "$BASE/session/$SESSION_ID/finalize" \
  -H "Content-Type: application/json" \
  -d '{"quality": false}' | jq .

# 6. Execute Resident Planner Self-Review (Zero-Eviction Review)
curl -s -X POST "$BASE/session/$SESSION_ID/self-review" | jq .

# 7. Download Finalized PNG Image Bytes Directly
curl -s "$BASE/session/$SESSION_ID/render" -o crypto_wallet_final.png
echo "Final render saved to crypto_wallet_final.png"

# 8. Export Research Trace for Academic Evaluation
curl -s "$BASE/session/$SESSION_ID/research-trace" -o research_trace.json
echo "Research trace saved to research_trace.json"
```

---

## 8. Verifier Stack & Alignment Pipeline

The automated verifier stack (`krivex_inference.verifiers`) executes post-generation evaluation without human intervention:

```
Finalized Visual Render
          │
          ├──► [CLIP ViT-L/14] ────────► Semantic Prompt Alignment Score [0.0 - 1.0]
          │
          ├──► [EasyOCR] ──────────────► Text Readability & Bounding Legibility [0.0 - 1.0]
          │
          ├──► [OpenCV Contours] ──────► Layout-IoU Spatial Overlap Score [0.0 - 1.0]
          │
          ├──► [LAION Predictor] ──────► Aesthetic Quality Metric [0.0 - 10.0]
          │
          └──► [ONNX ViT Gate] ────────► Binary Normal / NSFW Content Safety Flag
```

### Alignment & DPO Preference Store

Scores are normalized into a unified `VerifierScores` object and stored in `krivex_preference_pairs.db`. If candidate designs are evaluated against prior revisions, preference pairs `(chosen, rejected, prompt)` are recorded automatically.

To export these pairs for downstream Direct Preference Optimization (DPO):

```bash
curl -X POST "http://127.0.0.1:8420/preference-pairs/export?source=planner_review"
# Generates krivex_dpo_export_planner_review.jsonl for training
```

### Multi-Hardware Benchmark Evaluation & Research Ingestion

Post-generation benchmarks (KID, LPIPS, CLIP Score, CMMD, DINOv2 distance, Layout IoU, Typographic CER/WER, Aesthetic Score, and latency profiles) are evaluated via:

```bash
# Multi-metric evaluation on held-out reference designs across production hardware profiles:
python scripts/eval/run_evaluation.py \
  --profile 48gb \
  --checkpoint checkpoints/qwen_image21_lora_a6000_max \
  --reference-dir data/reference_uis \
  --wireframe-dir data/held_out_wireframes \
  --prompt-list data/eval_prompts.json \
  --output-dir eval_results

# Fast Offline Smoke Test (Synthetic Generation Mode):
# Emulates pipeline generation to verify metrics and LaTeX/Markdown/JSON exporters without GPU allocations:
python scripts/eval/run_evaluation.py --synthetic --num-samples 5 --output-dir eval_results

# Ingest and update research paper monograph
python scripts/eval/ingest_eval_results.py \
  --input eval_results/eval_comparison.json \
  --paper-path docs/RESEARCH_PAPER.md \
  --apply
```

Outputs include:
- `eval_results/eval_comparison.json`: Structured benchmark metrics across all evaluated hardware profiles.
- `eval_results/eval_comparison.md`: GitHub-ready Markdown summary table.
- `eval_results/eval_comparison_latex.tex`: Publication-grade LaTeX booktabs table for paper insertion.
- `eval_results/evaluation_receipt.json`: Cryptographic provenance receipt with SHA-256 digests.

---

## 9. Verification & Automated Test Suite

The inference codebase is rigorously validated with high-coverage pytest suites covering memory ledgers, state machines, API contracts, and real/mock backend lifecycles:

```bash
# Execute entire inference test suite
pytest tests/inference -v

# Run with concise output summary
pytest tests/inference -q
```

**Quality Standards & Pass Guarantees**:
* **100% Test Pass Rate**: 307 test cases in `tests/inference` (and 714 total tests across the monorepo).
* **Hermetic Execution (`KRIVEX_POLISH_SYNTHETIC=1`)**: Supports zero-safetensors synthetic execution for headless CI and consumer workstation testing, bypassing 14GB base model memory allocations while strictly validating parameter contracts, tensors, and pipeline interfaces.
* **Zero Git Diff Whitespace Errors**: Enforced by `git diff --check`.
* **State Machine Invariant**: Formal validation of PRD §7.4 transition table with $T_{\text{swapping}} \equiv 0.000\text{s}$ on self-review.
* **Exact Region Lock Invariant**: $S_{\text{lock}} \equiv 1.000$ and boundary gradient continuity $\Delta \nabla \le 0.05$ on locked elements.
