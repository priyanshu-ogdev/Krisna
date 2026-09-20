# Krisna — Agentic Design System

> [!IMPORTANT]
> **Canonical Master Document**: The formal, authoritative system specification is [docs/PRD.md](file:///d:/Krisna/docs/PRD.md) (Product & Research Requirements Document — Final No-RLHF Revision). All architecture, hardware envelopes, state machines, and citations conform strictly to it.

A conversational, agentic UI-design system: a Planner discusses intent, a
Sketch tier generates a partial draft, a Polish tier finalizes a
high-fidelity render, and an on-demand Critic tier reviews it. This
monorepo covers the full pipeline from raw public datasets to a running
inference service, plus a working control-panel UI over that service.

**Final architecture (no-RLHF-loop PRD revision)**: the Planner
(Qwen3.5-9B) and Qwen-Image-Edit-2511 ship **frozen** — RAG retrieval and
zero-shot ICL respectively, no fine-tuning. Gemma-4 (Critic) ships
**frozen** as an on-demand product feature. Only the **Sketch tier**
(trained from scratch) and **Z-Image-Turbo** (LoRA + Diffusion-DPO) are
actually trained. There is no AI-judge-labeled training data anywhere in
this repository — every training signal is either a real public dataset
or a frozen model needing no training data at all.

## Layout

```
krisna/
├── docs/                # all documentation — start at docs/PRD.md & docs/README.md,
│                           and docs/review/README.md for the full audit trail
├── scripts/              # all shell scripts, split by package
├── tests/                # ALL tests, root-level — pytest tests/ runs everything
├── models/                # trained checkpoint artifacts (empty until you train)
├── data-forge/             # data pipeline: raw public datasets -> model_data/
├── training/                # trains the Sketch tier + Z-Image-Turbo;
│                              deprecated Planner/Critic training kept for reference
├── inference/                # Unified Serving & Inference Subsystem:
│   ├── src/krisna_inference/ # SwapOrchestrator + real backends + FastAPI service
│   ├── frontend/             # Node.js control panel & Web Studio UI (Canvas2D Generalization Visualizer)
│   └── runtime/              # CLI session harness (run_agentic_session.py / krisna-session)
├── setup.sh                    # root orchestrator: phased environment setup
├── run_data_forge.sh / .ps1    # root orchestrator: data pipeline (Linux + Windows)
├── train.sh / train_all.ps1    # root orchestrator: training, all four tiers
├── run_inference.sh / .ps1     # root orchestrator: the FastAPI service + CLI session (-WithFrontend)
└── pytest.ini                  # root-level: makes `pytest tests/` work from here
```

See `docs/architecture/DIRECTORY_LAYOUT.md` for exactly why it's split
this way and the full old→new package mapping (this was restructured
from two separate repos — `data-forge` and `krisna-orchestrator` — with
the latter further split along a training/inference boundary).

## The real end-to-end path, verified stage by stage

This section exists because every step below was checked against the
actual code before being written down here, not assumed from a script's
own comment — see `docs/review/21_frontend_and_scripts_merge.md` for the
verification trail.

**1. Data pipeline** (`./run_data_forge.sh run`) — writes to
`$DATA_ROOT/model_data/` (`DATA_ROOT` defaults to `/data_krisna` on
Linux; override with the `DATA_ROOT` env var). See `data-forge/README.md`.

**2. Training** (`./train.sh <tier>`, or `--list` to see all four with
their real design intent and current hyperparameters) — each tier's
launcher prints exactly where its checkpoint lands and what to export
next once it finishes:
- Sketch Stage 1 → Stage 2 → `checkpoints/sketch_stage2_512/checkpoint_final.pt`
- Z-Image-Turbo dreambooth base → `checkpoints/polish_default_lora/`
- Z-Image-Turbo Diffusion-DPO refinement (on top of the base) →
  `models/dpo_checkpoints/<run>/final/`
- The VQGAN decoder (`./scripts/training/download_vqgan.sh`) — needed
  by **both** data-forge's Sketch-tier encoding **and** inference's
  VQ-token-to-pixel decode step at Finalize. Not a per-tier training
  artifact; download it once, independent of which tiers you train.

**3. Installing for inference** (`python3 scripts/inference/download_weights.py`,
or drive it from `inference/frontend/`'s Setup tab) — downloads the four
frozen HF models, auto-discovers your trained Sketch checkpoint and
Polish LoRA at the exact paths step 2 produces above, downloads the
VQGAN decoder automatically if `download_vqgan.sh` hasn't already been
run, and writes `.env.inference` at the repo root with every env var the
service needs (`KRISNA_SKETCH_CHECKPOINT`, `KRISNA_POLISH_DEFAULT_LORA_PATH`,
`KRISNA_VQGAN_CHECKPOINT`, `KRISNA_VQGAN_CONFIG`, `KRISNA_USE_REAL_BACKENDS=1`,
and more). This step is what makes "training → inference" a real,
connected path rather than two halves you have to wire together by hand.

**4. Running inference** (`./run_inference.sh`, or
`KRISNA_USE_REAL_BACKENDS=1 ./run_inference.sh` once step 3 is done) —
starts the real FastAPI service. `source .env.inference` first (or let
`inference/frontend/`'s own launcher do it — see below).

**5. Using it** — either directly against the API (`POST /session`,
`POST /session/{id}/message`, `POST /session/{id}/finalize`,
`POST /session/{id}/critique`, `GET /session/{id}/render` for the actual
rendered image bytes, `GET /orchestrator/status` for live VRAM/RAM —
both declared-budget and real hardware readings), or through
`inference/frontend/`'s Studio tab, which is a working UI over exactly
these endpoints: chat with the Planner, watch a phase-pipeline diagram
track Planner→Sketch→Polish→Critic, finalize and watch the pixel-forming
canvas reveal the real render, critique it, and watch live GPU/RAM
utilization (both the admission ledger's declared budget and the actual
hardware probe) the whole time.

**Yes, this is connected end to end** — verified by tracing every
producer/consumer pair above against the real code and configs, not
assumed from the scripts' own comments.

## Quick start

```bash
# One-time setup (Python venvs + Node frontend deps)
./setup.sh                        # base + backends + sketch + polish + critic + verifiers
./setup.sh frontend                # optional — the Node control panel (npm install)

# Data pipeline
./run_data_forge.sh run --dry-run

# Training (see what each tier actually needs first)
./train.sh --list

# Install for inference (downloads weights, locates your trained
# checkpoints, writes .env.inference) — real backends, needs a GPU
python3 scripts/inference/download_weights.py --dry-run   # check first
python3 scripts/inference/download_weights.py

# Preflight hardware & deployment diagnostics (NVIDIA GPU, CUDA, VRAM, Critic venv)
python scripts/inference/check_hardware.py --require-gpu
python scripts/inference/check_hardware.py --low-vram --json

# Docker deployment (production isolated multi-venv container + Web Studio)
./scripts/docker/run_docker.sh --build                     # Linux / macOS
.\scripts\docker\run_docker.ps1 -Build                    # Windows native PowerShell
docker compose up -d                                      # Or standard Docker Compose

# Inference service (Bare Metal Linux & Windows)
./run_inference.sh                                        # MockBackend — no GPU needed (test/CI)
set -a; source .env.inference; set +a; ./run_inference.sh  # real backends (fails fast if no GPU)
.\run_inference.ps1                                       # Windows native (PowerShell)
.\run_inference.ps1 -Real -LowVram                        # Windows GPU with low-VRAM CPU offload

# Control panel (optional — a UI over the same API; brings the real
# backend up itself if .env.inference already exists)
cd inference/frontend && npm start

# Test a real agentic session against MockBackend from the CLI
python inference/runtime/run_agentic_session.py --finalize --critique
# Or using the package CLI entrypoint:
krisna-session --finalize --critique

# Run every test in the monorepo in one pass (481 passed, 1 skipped, no GPU required
# — GPU-dependent tests are gated behind pytest.importorskip("torch")
# and skip cleanly if it's absent)
pytest tests/
```

## Where to go next

| I want to... | Go to |
|---|---|
| Read the canonical Product & Research Requirements Document | [docs/PRD.md](docs/PRD.md) |
| Understand the data pipeline (sources, licenses, preprocessing) | [data-forge/README.md](data-forge/README.md), [docs/data-forge/](docs/data-forge/) |
| Train the Sketch tier or Z-Image-Turbo | [training/README.md](training/README.md), [docs/training/](docs/training/) |
| Understand the swap orchestrator, backends, or low-VRAM mode | [inference/README.md](inference/README.md), [docs/inference/](docs/inference/) |
| Use the working control-panel UI & Canvas2D Visualizer | [inference/frontend/README.md](inference/frontend/README.md) |
| Manually test a trained checkpoint from the CLI | [inference/runtime/README.md](inference/runtime/README.md) |
| Understand the data-forge ↔ training sync contract | [docs/architecture/SYNC_DESIGN.md](docs/architecture/SYNC_DESIGN.md) |
| Understand the training ↔ inference multi-turn contract | [docs/architecture/TRAINING_INFERENCE_SYNC_DESIGN.md](docs/architecture/TRAINING_INFERENCE_SYNC_DESIGN.md) |
| See research findings, verification trail, and citations | [docs/architecture/RESEARCH_AND_CITATIONS.md](docs/architecture/RESEARCH_AND_CITATIONS.md) |
| See the full, phase-by-phase design-sync audit (Phase 1–27) | [docs/review/README.md](docs/review/README.md) |

## Test suite

465 tests across the three Python packages (`data_forge`,
`krisna_training`, `krisna_inference`), runnable together from the root:

```bash
pytest tests/                    # everything (465 tests)
pytest tests/data_forge/          # data pipeline only
pytest tests/training/             # training only
pytest tests/inference/             # inference/orchestrator only
```

No GPU required for any of the above — GPU-dependent tests use
`pytest.importorskip("torch")` and skip cleanly if it's absent, rather
than aborting collection for the whole run.

