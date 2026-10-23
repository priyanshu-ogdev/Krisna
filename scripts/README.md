# scripts/

All shell scripts, split by which package they serve. Every script
assumes it's run from the monorepo root (e.g. `./scripts/inference/
run_service.sh`, not `cd scripts/inference && ./run_service.sh`).

- **`data-forge/`** — `pin_revisions.py` (resolves dataset/model revisions
  to real commit SHAs) and `verify_schemas.py` (checks the hand-authored
  JSON Schemas in `configs/schemas/` against their Pydantic models — see
  `data-forge/README.md`). Everything else is data-forge's own installed
  CLI (`data-forge run`, `data-forge manifest ...`), not shell scripts.
- **`training/`** — env setup (`setup_env_training.sh`,
  `setup_env_diffusers_training.sh`), dataset prep
  (`prepare_polish_dataset.sh`), the data-forge sync bridge
  (`sync_from_data_forge_polish.sh`), and the production Polish launcher
  (`train_polish_default_lora.sh` / `train_all.ps1 -Tier polish-default`).
  (The Sketch tier is deterministic layout code requiring no training,
  and Planner/Critic training launchers are deprecated/sealed frozen).
- **`inference/`** — env setup (`setup_env.sh`, `setup_env_inference.sh`,
  `setup_env_verifiers.sh`), `run_tests.sh`, `run_service.sh`.
- **`eval/`** — post-training evaluation and paper update harness:
  - `run_evaluation.py`: multi-hardware benchmark evaluation across KID, LPIPS, CMMD, DINOv2, OCR CER, Layout IoU, and latency profiles.
  - `ingest_eval_results.py`: validates empirical results against pre-registered hypothesis bounds, issues cryptographic receipts (`eval_results/evaluation_receipt.json`), and automatically populates post-training tables in `docs/RESEARCH_PAPER.md`.
