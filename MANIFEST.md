# Krisna — diffed files only

Exactly the 17 files touched across this review, each at its real repo-relative
path under `Krisna/`. Drop these into your working copy at the matching paths
to apply every fix/upgrade from this review in one go. No file was added or
removed — every entry below is an in-place edit to an existing file.

Verified before packaging: all `.py` files here compile (`python3 -m py_compile`),
all `.yaml` files here parse (`yaml.safe_load`).

## inference/ (root-cause fix for the prompt/image mixup bug)
- `inference/src/krisna_inference/orchestrator/design_state.py` — added
  `FinalizeOutput.prompt_used`, the field that was missing.
- `inference/src/krisna_inference/orchestrator/flows.py` — `finalize()` now
  freezes the real generation-time prompt onto `prompt_used`; `critique_pass()`
  reads it back instead of re-synthesizing from possibly-drifted live
  conversation state, closing a real (prompt, image) mixup that landed
  directly in DPO training data.

## training/configs/ (YAML knobs for every upgrade below)
- `dpo_z_image_stage1_general.yaml`, `polish_stage2_dpo_general.yaml` —
  lora-target-modules fix, lora-dropout, target-passes, val-fraction/every/
  batches, max-grad-norm, ema-decay, flip-prob, val-batch-size.
- `polish_stage1_default_lora.yaml`, `polish_default_lora_z_image.yaml` —
  re-enabled `validation_prompt`.
- `sketch_stage1_256.yaml`, `sketch_stage2_512.yaml`,
  `sketch_train_stage1_256.yaml`, `sketch_train_stage2_512.yaml` —
  val_manifest_path/val_every/val_batches, target_epochs,
  gradient_accumulation_steps.

## training/src/krisna_training/ (the actual logic)
- `data_forge_bridge/sync_dpo_pairs.py` — deterministic pair IDs (fixes
  duplicate-row-per-resync bug) + skip-if-exists before any image I/O.
- `preference/preference_store.py` — `INSERT OR IGNORE` + new `exists()`,
  making resyncs truly idempotent.
- `polish/dpo_dataset.py` — deterministic train/val split by hashed pair id;
  paired (chosen+rejected share one coin-flip) horizontal-flip augmentation,
  default off.
- `polish/dpo_loss.py` — added `apply_flow_matching_shift` (resolution-
  dependent timestep shift, Esser et al. 2024 Eq. 23) and `ema_warmup_decay`.
- `polish/train_dpo.py` — fixed missing `LoraConfig.target_modules` (would
  have crashed on first run); fixed `load_lora_weights()` not guaranteeing
  `requires_grad=True` on the continue-from-base-checkpoint path (would have
  silently trained nothing); added gradient clipping, EMA, heldout DPO
  validation, corpus-scale step rescaling, flip-prob/val-batch-size wiring,
  and a quantitative VRAM-budget note justifying why `--train-batch-size`
  stays at 1 by default.
- `sketch/dataset.py` — reverted `caption_mix_ratio` default back to 0.95
  (was silently drifted to 0.85 against every other config).
- `sketch/train.py` — fixed optimizer-state and LR-scheduler not resuming
  correctly on `resume_from`; added decoupled weight decay, gradient
  accumulation, and the heldout validation loop.

See `training_upgrades.diff` / `inference_upgrades.diff` for the exact
line-level changes.
