# Contributing to Krisna

## Scope and ownership

Krisna is a monorepo with explicit subsystem ownership:

| Area | Ownership boundary |
|---|---|
| `data-forge/` | Source governance, preprocessing, receipts, and model-data export |
| `training/` | Dataset bridges, native trainer contracts, adapter manifests |
| `inference/` | Model plane, orchestrator, native backends, verifiers, API |
| `inference-frontend/` | Node/Tauri control panel and local IPC client |
| `scripts/` | Operational entry points; scripts must call package APIs rather than duplicate logic |
| `tests/` | Contract and behavior tests grouped by subsystem |
| `docs/` | Binding product, architecture, research, and runbook documentation |

Do not mix generated data, credentials, model weights, or local databases into
source changes.

## Change workflow

1. Read the subsystem README and the relevant architecture/runbook document.
2. Inspect `git status` before editing. Preserve unrelated work.
3. Define the user-visible or research-visible acceptance criteria.
4. Make the smallest complete change at the owning boundary.
5. Add or update a focused test for changed behavior and failure paths.
6. Run the smallest relevant checks, then `git diff --check`.
7. Update the authoritative documentation when a contract, model, data source,
   or operational command changes.

## Required evidence rules

- A configured model is not a loaded model.
- A registry count is not a realized dataset count.
- A checkpoint path is not evidence of a completed training run.
- A model-card capability is not evidence of Krisna task quality.
- A synthetic test artifact must be labeled synthetic.
- Research claims require a receipt containing model revision, data hash,
  split policy, seed, evaluator, and hardware.

## Commit convention

Use one logical subsystem per commit where practical:

```text
feat(data-forge): add strict source provenance receipt
fix(inference): reject unsupported native Qwen controls
test(training): cover paired dataset preflight
docs(research): record publisher-backed flow-matching plan
chore(repo): standardize ignore and contributor policy
```

Before a commit, inspect:

```bash
git status --short
git diff --check
git diff --stat
git diff -- <intended files>
```

Never commit `.env`, tokens, credentials, model weights, datasets, databases,
blob stores, `__pycache__`, `node_modules`, or generated audit output.

## Verification commands

```bash
python -m pytest tests/data_forge -q
python -m pytest tests/training -q
python -m pytest tests/inference -q
python -m compileall -q data-forge/src inference/src training/src
git diff --check
```

GPU/model smoke tests are separate gates and must be reported as unrun when the
required hardware, checkpoints, or external trainer checkout is unavailable.
