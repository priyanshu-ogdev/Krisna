# scripts/data-forge/

Unified master launchers, diagnostic auditors, and maintenance scripts for the **Krivex Data-Forge** zero-touch data pipeline.

## Master Launchers (PowerShell & Bash Parity)

The pipeline provides two native 1:1 parity launchers that automatically resolve the Python virtual environment (`$env:PYTHON_EXECUTABLE`, active conda `torch_env`, or `.venv`), source root paths into `PYTHONPATH`, and configure environment variables from `.env`:

- **Windows PowerShell**: `.\scripts\data-forge\run_data_forge.ps1 <command> [options]`
- **Linux / WSL2 Bash**: `./scripts/data-forge/run_data_forge.sh <command> [options]`

### Master Commands

| Command | Action | Description |
| :--- | :--- | :--- |
| `doctor` | Pre-flight diagnostics | Stage graph ordering consistency, toolchain support, and API connectivity. |
| `audit` / `registry` | Forensic audit | Preflight audit: immutable commit SHAs, license policies, and record counting. |
| `pin-revisions` | SHA pinning | HuggingFace Hub resolution of unpinned `main` branches to 40-char commit SHAs. |
| `verify-schemas` | Schema check | Validates hand-authored JSON Schemas against Pydantic models. |
| `run` | Pipeline execution | Runs stage graph (supports `--dry-run`, `--resume`, `--stages 0,1,2`, `--limit`). |
| `manifest` | Manifest query | Queries SQLite database (`stats`, `query --status <status>`, `export`). |
| `sync-training` | Training bridge | Materializes `model_data/` exports into training-ready paired datasets. |
| `all` | Full automated flow | Executes `doctor` -> `audit` -> `run` -> `sync-training` in sequence. |
| `setup` | Environment setup | Sets up execution environment and dependencies. |

### Quick Start Examples

```powershell
# PowerShell (Windows)
.\scripts\data-forge\run_data_forge.ps1 audit
.\scripts\data-forge\run_data_forge.ps1 pin-revisions --apply
.\scripts\data-forge\run_data_forge.ps1 verify-schemas
.\scripts\data-forge\run_data_forge.ps1 run --dry-run
.\scripts\data-forge\run_data_forge.ps1 all
```

```bash
# Bash (Linux / WSL2)
./scripts/data-forge/run_data_forge.sh audit
./scripts/data-forge/run_data_forge.sh pin-revisions --apply
./scripts/data-forge/run_data_forge.sh verify-schemas
./scripts/data-forge/run_data_forge.sh run --dry-run
./scripts/data-forge/run_data_forge.sh all
```

---

## Utility Scripts

### `audit_registry.py`
Executes preflight audit receipts using `ManifestPlanningStage._build_preflight_receipt(config)`. Validates zero policy violations, verifies all active sources are commercial/academic permissible, counts effective records, and reports visible GPU hardware suitability.

### `pin_revisions.py`
Resolves every unpinned `revision: "main"` in `data-forge/configs/{datasets,models}.yaml` to a real immutable 40-character Git commit SHA via the HuggingFace Hub API. Uses `ruamel.yaml` to preserve comments and indentation byte-for-byte (validated via `tests/data_forge/test_pin_revisions.py`).

```bash
python scripts/data-forge/pin_revisions.py           # dry run (diff inspection)
python scripts/data-forge/pin_revisions.py --apply    # writes real SHAs to YAML
```

### `verify_schemas.py`
Pre-flight structural check of every hand-authored JSON Schema in `data-forge/configs/schemas/` against the corresponding Pydantic model's generated schema (required fields + property types/enums).

```bash
python scripts/data-forge/verify_schemas.py
```

### `sync_to_training.py`
Bridges exported `model_data/polish_qwen_image21` records into HuggingFace dataset format with metadata receipts, SHA-256 target/condition hash manifests, 7-bucket aspect ratio manifold binning (`1024,1024;768,1344;1344,768;832,1248;1248,832;896,1152;1152,896`), and wireframe renderer fingerprinting (`krivex.renderer.v2.7132a5a7`). Enforces group-disjoint train/validation splits for direct consumption by training and post-training evaluation scripts.

See [data-forge/README.md](file:///d:/Krisna/data-forge/README.md) for complete architecture and stage references.
