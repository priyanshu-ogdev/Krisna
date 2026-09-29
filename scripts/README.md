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
  `setup_env_critic.sh`, `setup_env_diffusers_training.sh`), dataset prep
  (`prepare_sketch_dataset.sh`, `prepare_polish_dataset.sh`,
  `download_vqgan.sh`), the data-forge sync bridge
  (`sync_from_data_forge_{sketch,polish,dpo}.sh`), and training launchers
  (`train_sketch_stage{1,2}.sh`, `train_polish_default_lora.sh`,
  `train_planner_lora.sh` — deprecated, `train_critic_qlora.sh` —
  deprecated).
- **`inference/`** — env setup (`setup_env.sh`, `setup_env_inference.sh`,
  `setup_env_verifiers.sh`), `run_tests.sh`, `run_service.sh`.
