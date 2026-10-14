# Repository structure and release policy

## Principles

The repository is organized as a Python/Node/Rust monorepo with package-local
ownership and root-level integration tests. Runtime state is externalized and
never committed. Documentation distinguishes implementation, design proposal,
primary-source evidence, and pending validation.

## Source tree

```text
data-forge/       governed data ingestion and export package
training/         training contracts, adapters, and receipts
inference/        Python model plane and API
inference-runtime CLI smoke/session harness
inference-frontend Node/Tauri product surface
scripts/          thin operational wrappers
tests/            subsystem and integration tests
docs/             canonical product, research, architecture, and runbooks
.github/          CI/repository automation
```

## Artifact policy

The following are local or generated artifacts and are ignored by policy:

- raw/scrubbed/processed datasets and model-data exports;
- model weights, checkpoints, LoRA adapters, GGUF/GGML files;
- virtual environments, caches, databases, blob stores, logs, and reports;
- downloaded third-party repositories;
- frontend dependencies and Rust build output.

If an artifact is required to reproduce a research result, commit its small
metadata receipt and immutable source/checkpoint identifier—not the binary.

## Release boundaries

Before a release or push to `main`:

1. all intended changes are separated into logical commits;
2. generated files are absent from the staged diff;
3. package tests and compile checks pass;
4. research docs do not present pending gates as measurements;
5. the final commit list and diff are reviewed against `origin/main`.
