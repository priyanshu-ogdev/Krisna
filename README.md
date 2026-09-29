# Krisna — Agentic Design System

A conversational, agentic UI-design system: a Planner discusses intent, a
Sketch tier generates a partial draft, a Polish tier finalizes a
high-fidelity render, and an on-demand Critic tier reviews it. This
monorepo covers the full pipeline from raw public datasets to a running
inference service.

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
├── docs/                # all documentation — start at docs/README.md
├── scripts/              # all shell scripts, split by package
├── tests/                # ALL tests, root-level — pytest tests/ runs everything
├── models/                # trained checkpoint artifacts (empty until you train)
├── data-forge/             # data pipeline: raw public datasets -> model_data/
├── training/                # trains the Sketch tier + Z-Image-Turbo;
│                              deprecated Planner/Critic training kept for reference
├── inference/                # SwapOrchestrator + real backends + FastAPI service
├── inference-runtime/         # CLI harness: run a real agentic session for testing
├── src/                       # frontend — scaffolded, design deferred
└── pytest.ini                 # root-level: makes `pytest tests/` work from here
```

See `docs/architecture/DIRECTORY_LAYOUT.md` for exactly why it's split
this way and the full old→new package mapping (this was restructured
from two separate repos — `data-forge` and `krisna-orchestrator` — with
the latter further split along a training/inference boundary).

## Quick start

```bash
# Data pipeline (produces data-forge/../model_data/ — see data-forge/README.md)
cd data-forge && pip install -e . && data-forge run --dry-run

# Inference service, MockBackend mode — no GPU, no trained checkpoints needed
./scripts/inference/setup_env.sh
./scripts/inference/run_service.sh

# Test a real agentic session against MockBackend (safe default)
python inference-runtime/run_agentic_session.py --finalize --critique

# Run every test in the monorepo in one pass (355+ tests, no GPU required)
pytest tests/
```

## Where to go next

| I want to... | Go to |
|---|---|
| Understand the data pipeline (sources, licenses, preprocessing) | `data-forge/README.md`, `docs/data-forge/` |
| Train the Sketch tier or Z-Image-Turbo | `training/README.md` |
| Understand the swap orchestrator, backends, or low-VRAM mode | `inference/README.md` |
| Manually test a trained checkpoint | `inference-runtime/README.md` |
| Understand the data-forge ↔ training sync contract | `docs/architecture/SYNC_DESIGN.md` |
| See exactly what was researched/verified and why each decision was made, with citations | `docs/architecture/RESEARCH_AND_CITATIONS.md` |
| See the full PRD-vs-implementation reasoning trail | `docs/architecture/`, each package's README's "what's genuinely still missing" sections |

## Test suite

355+ tests across all three Python packages (`data_forge`, `krisna_training`,
`krisna_inference`), runnable together from the root:

```bash
pytest tests/                    # everything
pytest tests/data_forge/          # data pipeline only
pytest tests/training/             # training only
pytest tests/inference/             # inference/orchestrator only
```

No GPU required for any of the above — GPU-dependent tests are gated
behind `torch` import availability and skip cleanly if it's absent.
