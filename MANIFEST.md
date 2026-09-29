# Krisna — final diffed files only

35 files total (32 modified + 3 new), each at its real repo-relative path
under `Krisna/`. This is the fully reconciled result of two parallel
review streams merged into one — see "How this was reconciled" below.

Verified before packaging (this exact file set, not just the working
tree): every `.py` file compiles (`python3 -m py_compile`), every `.yaml`
file parses (`yaml.safe_load`), the shell script passes `bash -n`. 68 real
tests executed (via a minimal pytest-compatible shim built this session,
since this sandbox has no network access to install real pytest/torch)
all pass — 64 pytest-style + 4 unittest-style.

## New files (3)
- `inference/src/krisna_inference/orchestrator/constraint_merge.py` —
  shared helper so Planner constraint-updates are applied identically
  by both the same-turn Sketch call and the persisted DesignState,
  closing a real mid-turn staleness bug (Sketch was generating from the
  turn's OLD constraints, one turn behind the Planner's own extraction).
- `scripts/training/dpo_lora_autodetect.py` — shell-wrapper-level
  auto-detection of the base LoRA fine-tune checkpoint, unit-tested,
  used by `train_polish_dpo.sh`.
- `training/src/krisna_training/polish/repa.py` — REPA (Representation
  Alignment, Yu et al. 2024, arXiv:2410.06940) loss, projection head,
  and DINOv2 preprocessing for `train_dpo.py`. Opt-in, disabled by
  default.

## Inference layer — bug fixes
- `critic_backend.py` — missing `import os` (guaranteed `NameError` on
  direct construction with default args).
- `polish_default_backend.py` — switched to `ZImageImg2ImgPipeline`;
  the previous txt2img-only pipeline silently discarded the sketch
  handoff on every "quick" finalize.
- `factory.py` — wires the new `strength` kwarg through.
- `design_state.py` / `flows.py` — added `FinalizeOutput.prompt_used`,
  frozen at generation time and read back by `critique_pass()` instead
  of re-synthesizing a prompt from possibly-drifted live conversation
  state — closes a real (prompt, image) mixup landing directly in DPO
  training data.
- `swap_orchestrator.py` / `flows.py` — both now call
  `constraint_merge.apply_constraint_updates` instead of two independent
  inline copies that could (and did) drift.
- `preference/pair_builder.py` (+ `dpo/pair_builder.py` shim) —
  `MismatchedPromptError`: candidates being paired with disagreeing
  `prompt_used` values now raise instead of silently writing an
  ambiguous pair.

## Training layer — bug fixes
- `sketch/dataset.py` — `caption_mix_ratio` default reverted 0.85→0.95
  (had silently drifted from every config/citation agreeing on 0.95).
- `sketch/train.py` — resume didn't restore Adam's moment buffers or
  fast-forward the LR scheduler (both silently restarted from scratch
  on every resume); added a resume config-mismatch guard, decoupled
  weight decay, gradient accumulation, the heldout validation loop, and
  a dropped empty-manifest guard that would have silently reported a
  fake `val_loss: 0.0`.
- `sketch/model.py` / `pos_embed.py` — added opt-in 2D axial RoPE
  (`use_2d_rope`, default off) as a SOTA-aligned alternative to the
  learned absolute position table, removing the need for bicubic
  interpolation between progressive-resolution stages.
- `sketch/vq_tokenizer.py` — corrected a real checkpoint-attribution
  error (docstring claimed boris/vqgan_f16_16384's CC3M/YFCC100M-
  fine-tuned weights; the actual download URLs point to the plain
  ImageNet-only base checkpoint) — verified via web search against the
  real HF model cards and CompVis's own README, not assumed.
- `polish/train_dpo.py` — missing `LoraConfig.target_modules` (would
  crash on first fresh-adapter run); `load_lora_weights()` not
  guaranteed to set `requires_grad=True` (could silently train
  nothing); no gradient clipping; `.iterdir()` vs `.safetensors` glob
  inconsistency between the shell-level and Python-level LoRA
  auto-detect checks. Added EMA (with warmup), flow-matching
  resolution-dependent timestep shift, heldout DPO validation, paired
  flip augmentation, corpus-scale step rescaling, and the REPA hook.
- `polish/dpo_dataset.py` — deterministic train/val split by hashed
  pair id; paired (shared coin-flip) horizontal-flip augmentation.
- `data_forge_bridge/sync_dpo_pairs.py` — non-deterministic pair IDs
  caused full duplicate re-imports on every resync; unguarded
  `meta["image_a"]` could crash the entire sync on one malformed record.
- `preference/preference_store.py` — `INSERT OR IGNORE` + `exists()`,
  making resyncs actually idempotent.

## How this was reconciled
This session picked up a prior review's diffed-files upload (34 files:
the inference-layer prompt-mixup/backend fixes above) partway through,
verified each claim against the actual code rather than trusting the
description, then continued independently into the training loop
(gradient accumulation, weight-decay grouping, resume fixes, 2D RoPE,
REPA) — producing a SECOND, divergent set of changes to some of the
same files (`train_dpo.py`, `sketch/train.py`, `sync_dpo_pairs.py`, four
YAML configs). Reconciling meant diffing every overlapping file between
both trees file-by-file: files where one side was a strict superset were
adopted wholesale (`flows.py`, `swap_orchestrator.py`, `critic_backend.py`,
etc.); files with genuine, non-overlapping fixes on both sides
(`train_dpo.py`'s `.safetensors` check, `sketch/train.py`'s empty-manifest
guard, `sync_dpo_pairs.py`'s KeyError guard) were merged by hand, then
re-verified by actually running the affected tests — not merely
re-compiling — to confirm the merge preserved both sides' fixes.
