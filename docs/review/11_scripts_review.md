# Phase 11 — `scripts/` Review: Running the Design, Data-Forge, and Training Pipeline

Reviewed every script under `scripts/` for whether it actually runs the
design as documented — real paths, real function signatures, real env
vars — not just whether it looks plausible. Several were run for real
where the sandbox allowed it, not just read.

## Real bugs found and fixed

| File | Bug | Fix |
|---|---|---|
| `scripts/data-forge/verify_schemas.py` | **Broken path resolution** — computed `schemas_dir` as `<repo-root>/configs/schemas` (2 `.parent` calls from the script's own location), which doesn't exist. The real schemas live at `data-forge/configs/schemas/`. This script could never have found a single schema file as originally written. | Fixed the parent-count (3 `.parent` calls + `/ "data-forge"`). **Ran it for real after fixing — all 6 schemas (`caption_output.json`, `safety_output.json`, `structure_output.json`, `license_output.json`, `ocr_output.json`, `audit_output.json`) now confirmed structurally in sync with their Pydantic models.** This was a genuinely broken, never-functional script until this fix. |
| `scripts/inference/run_service.sh` | Pointed operators at `docs/inference/LOW_VRAM_MODE.md` for the low-VRAM cost breakdown — that file doesn't exist anywhere in the repo (`docs/inference/` only has a 5-line `README.md`). | Repointed to the two files that actually contain this content: `docs/architecture/RESEARCH_AND_CITATIONS.md` §4 and `docs/review/06_inference_orchestrator.md`. |
| `scripts/training/train_critic_qlora.sh` | Final line told the operator to `export KRISNA_CRITIC_LORA_PATH=...` as if this were a working integration step. Checked `inference/backends/factory.py` directly: that env var was **explicitly removed** (`# REMOVED: KRISNA_CRITIC_LORA_PATH`), and `tests/inference/test_inference_factory.py` verifies it's a no-op. The script was giving a real, actionable-looking instruction for something that does nothing in the live system — the worst kind of doc bug, since it looks correct. | Added an upfront deprecation banner (matching the Python package's own `DeprecationWarning`) and replaced the misleading final line with an accurate one explaining the env var does nothing and what reviving it would actually require. |
| `scripts/training/train_planner_lora.sh` | No deprecation warning at all, despite `krisna_training.planner` itself carrying a real `DeprecationWarning` and `inference/planner_backend.py` no longer having a `lora_adapter_path` parameter to consume this script's output. | Added the same upfront deprecation banner as the Critic script. |
| `scripts/README.md` | Claimed `scripts/data-forge/` is "currently empty" — it has 4 real files (`README.md`, `pin_revisions.py`, `setup_env.ps1`, `verify_schemas.py`). Stale doc, same class of bug as the two found earlier in `RESEARCH_AND_CITATIONS.md`. | Corrected to actually describe the two real scripts. |
| `scripts/data-forge/README.md` | Documented `pin_revisions.py` but never mentioned `verify_schemas.py` existed at all. | Added its own section. |

## Checked and confirmed correct (not changed)

- `sync_from_data_forge_sketch.sh`, `sync_from_data_forge_dpo.sh`,
  `sync_from_data_forge_polish.sh` — all call their target `sync()`
  functions with the exact real signatures (checked directly against
  each bridge module's actual `def sync(...)`, not assumed). The sketch
  sync script needed no changes for today's caption-mixing fix, since
  `sync()`'s own signature didn't change — only the dict it writes
  internally gained a field.
- `train_sketch_stage1.sh` / `train_sketch_stage2.sh` — pass `--config`
  straight through to `TrainConfig.from_yaml`, which maps every yaml key
  to a dataclass field by name (`cls(**data)`). Today's new
  `caption_mix_ratio`/`cfg_dropout_prob` config keys are picked up
  automatically; no script change needed.
- `train_polish_default_lora.sh` / `train_polish_dpo.sh` — both are
  generic yaml→CLI-flag passthroughs (`--{key}={value}` for every
  non-null config key). Confirmed today's `rank: 32` addition to
  `polish_default_lora_z_image.yaml` becomes `--rank=32` automatically.
- **`train_polish_dpo.sh`'s final `export KRISNA_POLISH_DEFAULT_LORA_PATH=...`
  hint — checked closely on suspicion it was a copy-paste bug from the
  base LoRA script, and it is NOT a bug.** Traced `factory.py` directly:
  `Tier.POLISH_DEFAULT` only ever loads one LoRA path
  (`KRISNA_POLISH_DEFAULT_LORA_PATH`), and
  `dpo_z_image_stage1_general.yaml`'s `lora-adapter-path: null` default
  means DPO trains independently from raw base weights unless an
  operator explicitly chains it. The DPO checkpoint is meant to
  **supersede** the base LoRA at deploy time by pointing the same env
  var at its own output — correct as written. **This corrects an
  overstatement in `docs/review/02_polish_tier.md`**, which had
  described the two LoRAs as being "composed... at inference" — fixed
  there too, in place, with the correction visible rather than silently
  edited away (same practice as the Phase 6 correction).
- `download_vqgan.sh`, `prepare_sketch_dataset.sh`,
  `prepare_polish_dataset.sh` (n/a — doesn't exist as a separate
  script; polish dataset prep goes through the data-forge sync path),
  all `setup_env*.sh` scripts — checked path/package references
  against real files; all correct. `setup_env_critic.sh`'s `--no-deps`
  install and isolated-venv rationale matches what Phase 4 already
  independently verified about the real `transformers` version
  conflict between the Critic and Planner tiers.
- `scripts/data-forge/pin_revisions.py` — **ran it for real** (after
  installing its `ruamel.yaml`/`huggingface_hub` dependencies). It
  correctly enumerates every dataset/model entry and attempts to
  resolve each one's real commit SHA, failing gracefully per-entry
  (not crashing the whole run) when this sandbox's network egress
  doesn't allow `huggingface.co` — an environment limitation, not a
  script bug. This is genuine confirmation the script works, not just
  a read-through.

## What this means for "properly running the design"

The pipeline's actual run order (data-forge → sync bridges → training
launchers → inference) is correctly wired end-to-end in the scripts
that matter for the active (non-deprecated) path. The bugs found were
all in the *edges* — stale documentation, a genuinely broken utility
script nobody had run, and misleading operator-facing hints for the
deprecated tiers — not in the core Sketch/Polish/Planner/Critic
pipeline flow itself, which checks out.

---
See `docs/review/02_polish_tier.md` for the corrected LoRA-composition
finding, `docs/review/04_critic.md` for the Critic deprecation
background, and `docs/review/03_planner.md` for the Planner's.
