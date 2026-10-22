# Directory Layout — Krivex Monorepo

This repo was restructured from two separate projects (`data-forge`,
`krivex-orchestrator`) into one monorepo, with `krivex-orchestrator`
further split into `training/` and `inference/` along an industry-standard
training/serving boundary. This doc explains why, and exactly what maps
to what — read this before touching import paths or scripts.

## Top-level layout

```
krivex/
├── docs/                  # all documentation, root-level, split by subsystem
├── scripts/                # all shell scripts, root-level, split by subsystem
├── tests/                  # ALL tests, root-level — not nested per-package
│   ├── data_forge/
│   ├── training/
│   └── inference/
├── models/                 # trained checkpoint artifacts (data, not code)
├── data-forge/              # data pipeline package (data_forge/)
├── training/                # training package (krivex_training/)
├── inference/                # serving package (krivex_inference/)
├── inference/
│   ├── runtime/               # CLI harness: run a real agentic session
│   │                           # against models/ for manual testing
│   └── frontend/              # Node/Tauri control plane
└── pytest.ini                # root-level: asyncio_mode + testpaths, so
                               # `pytest tests/` works from repo root
                               # regardless of which package a test covers
```

## Why training/inference is a real package split, not just folders

The original repository had `training/` and `inference/` as
subpackages of one single package. That was fine when it
was one deployable unit, but it conflates two things with genuinely
different runtime requirements:

- **`krivex_training`**: needs `torch`, `diffusers`, `peft`, and (for the
  now-deprecated Critic path) `unsloth` + `transformers==5.5.0` pinned
  exactly. Runs on a training box, possibly disconnected from anything
  serving real requests.
- **`krivex_inference`**: needs `fastapi`/`uvicorn` for serving, plus
  whichever backend deps are active. Runs on the serving box.

Splitting them into separate installable packages (`pip install
./training`, `pip install ./inference`) means a deployment that only
serves doesn't need to pull in training-only dependencies (and vice
versa) — this is the actual industry-standard reason for a training/
inference split, not just naming hygiene.

## Package → package mapping (old → new)

| Legacy Module Path | New Package Path | Notes |
|---|---|---|
| `swap_orchestrator`, `state_machine`, `model_registry`, `vram_budget`, `design_state`, `flows`, `service`, `store`, `exceptions`, `logging_setup` | `krivex_inference.orchestrator.*` | Coordination layer — no GPU code, fully testable with `MockBackend` |
| `inference.*` (backends, factory, planner_rag, common, blob_store_singleton, maskgit_model, sketch_handoff) | `krivex_inference.backends.*` | Real model-loading code |
| `verifiers.*` | `krivex_inference.verifiers.*` | Unchanged internals |
| `training.sketch`, `training.polish` | `krivex_training.sketch`, `krivex_training.polish` | Active training code |
| `training.planner`, `training.critic` | `krivex_training.planner`, `krivex_training.critic` | **Deprecated** — see their `__init__.py` docstrings. Planner/Critic ship frozen under the final PRD |
| `training.data_forge_bridge` | `krivex_training.data_forge_bridge` | Sync bridge — training-side, since its whole job is producing training data |
| `dpo.*` | `krivex_training.dpo.*` | DPO data prep (pair building, preference store) is training-data prep, not a serving concern |

## The one real cross-package dependency

`krivex_training.sketch.train` imports `krivex_inference.verifiers.common`
for a shared CLIP embedder utility, used for scoring during training. This
is real and intentional, not accidental coupling — `krivex-training`'s
`pyproject.toml` declares `krivex-inference` as a dependency for exactly
this reason. Nothing in `krivex_inference` depends on `krivex_training`.

## data-forge ↔ training ↔ inference sync contract

See `docs/architecture/SYNC_DESIGN.md` for the full, verified (not just
documented) contract — directory names, field names, and join keys that
must match exactly across all three packages for the sync bridge
(`krivex_training.data_forge_bridge`) to work.

## Tests live at the root, not nested per-package

`tests/data_forge/`, `tests/training/`, `tests/inference/` — run
`pytest tests/` from the repo root to run everything in one pass, or
`pytest tests/inference/` etc. to scope to one package. Each package's
own `pyproject.toml` still declares its own `testpaths` (pointing at
`../tests/<package>`) for IDE/tooling that runs pytest from inside a
package directory instead of the root — both entry points resolve to the
same files.

## What moved where, mechanically

This was a real restructuring, not a relabeling: every import statement
across ~150 Python files was rewritten to the correct new package path per the table above, every shell
script's path assumptions were fixed for the new root, and the full test
suite (355 tests across all three packages) was run and confirmed passing
from the new root before this restructuring was considered complete — not
just compiled.
