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
| Polish Default | `Qwen/Qwen-Image-2.1` native image-conditioned pipeline plus optional LoRA | LoRA only |
| Polish Quality | Frozen Qwen-Image-2.1 without an adapter | No |
| Data-Forge VLM | Local Qwen3.8-27B preprocessing/audit dependency | No |

The Qwen-Image-2.1 LoRA path is configured and contract-tested, but it is not
considered trained until a real source-disjoint pair manifest, native one-batch
GPU smoke test, adapter receipt, and evaluation receipt exist.

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
