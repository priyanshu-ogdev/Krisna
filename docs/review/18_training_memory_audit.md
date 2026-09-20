# Phase 18 — Training-Layer Memory Audit (A6000 48GB / 128GB RAM)

Triggered by confirming the actual training target hardware: a single
RTX A6000 (48GB VRAM) with 128GB system RAM for offload. Audited every
training script's real memory footprint against that budget, not the
inference-side 12GB/24GB envelopes covered in earlier phases.

## Findings and fixes

**Sketch Stage 2 (512px, 1024 tokens/seq)**: batch_size=16 (vs. Stage
1's 32) only offsets the ~16x attention-FLOPs increase (O(n²) in
sequence length, 4x tokens) to a net ~8x compute/memory increase per
step — not the ~2x a plain "halved batch" framing suggests. Memory-wise,
`nn.TransformerEncoderLayer` (`batch_first=True`, `norm_first=True`, no
explicit attention mask) likely dispatches to PyTorch's
`scaled_dot_product_attention` fast path (memory-linear flash attention,
head_dim=64 in range) even during training, but that depends on PyTorch
version/kernel selection this environment can't verify. Added an
optional `use_gradient_checkpointing` to `SketchModelConfig`/`build_model()`
— manually iterates `self.encoder.layers` with
`torch.utils.checkpoint.checkpoint()` rather than restructuring the
module, preserving `state_dict` key compatibility with existing
checkpoints exactly (`nn.TransformerEncoder` here has `norm=None`, so
this is a drop-in equivalent). Off by default (Stage 1 doesn't need it),
on for `sketch_train_stage2_512.yaml`.

**Planner (Qwen3.5-9B, unquantized bf16, batch=4/max_length=4096, no
gradient checkpointing at all)**: ran the actual math. Qwen3.5-9B's real
published specs (32 layers, hidden=4096, 16 heads, dense) plugged into
the standard dense-transformer activation-memory approximation
(Korthikanti et al. 2022) give an upper bound of **~245GB** without
checkpointing at these settings — nowhere close to 48GB even generously
discounting for this being a full-attention approximation of a cheaper
hybrid Gated-DeltaNet architecture. With checkpointing: ~7.7GB. Fixed —
but correctly: gradient checkpointing on a frozen-LoRA-base model has a
documented failure mode (confirmed via three real GitHub issues on the
actual peft/transformers repos, not assumed) where the backward pass
silently produces **zero gradient updates** for the adapter unless
`enable_input_require_grads()` is called on the base model, in the right
order relative to `get_peft_model()`. Implemented in that order, plus
`config.use_cache = False` (required companion setting for causal-LM
training with checkpointing).

**DPO (`train_dpo.py`)**: three separate real gaps.
1. **No LR scheduler at all** — plain constant `AdamW` from step 0,
   unlike every sibling config in this repo, against a `beta=2000` DPO
   loss (unusually high-magnitude). Added `--lr-warmup-steps`/
   `--lr-scheduler` via diffusers' `get_scheduler`, wired into the real
   training loop.
2. **A fully redundant model copy.** `ref_pipe` was loaded as the *whole
   pipeline* (transformer + VAE + text encoder(s)) via
   `AutoPipelineForText2Image.from_pretrained()`, but only
   `ref_pipe.transformer` is ever used — `prompt_embeds` come from
   `policy_pipe.encode_prompt()`, and the reference VAE is never called.
   The unused VAE/text-encoder(s) were freed immediately after load
   rather than held resident for the whole run.
3. **Both full bf16 transformer copies held on GPU simultaneously** —
   DPO's dominant VRAM cost, documented in the literature (Reg-DPO,
   arXiv:2511.01450, §5 "Model Offloading for Frozen Modules": moving a
   frozen reference model's weights to CPU between forward passes cuts
   peak memory ~10GB in their setup). `ref_transformer` needs no
   gradients and is only used inside `torch.no_grad()`, so it now lives
   on CPU by default, swapped to GPU only for its brief forward call
   each step, and was removed from `accelerator.prepare()` (which would
   have forced it back onto GPU immediately, undoing the placement).
   `enable_gradient_checkpointing()` added for the trainable policy
   transformer too.

## The LoRA-alpha correction — and a correction of my own correction

Found Critic (`lora_r=16, lora_alpha=16`, scaling=1.0) and DPO (same
ratio) used a different alpha/rank policy than Planner
(`lora_r=16, lora_alpha=32`, scaling=2.0), with no comment explaining
the split. First pass: "standardized" Critic and DPO to match Planner's
2x ratio, on the theory that the split was accidental drift.

That theory was wrong, and research (not assumption) caught it before
it shipped as a permanent claim. Two separate, real, ecosystem-specific
conventions apply here:

- **Diffusers' own dreambooth-LoRA scripts** (SDXL, SD3, vanilla
  dreambooth, and diffusers' own official `training/lora.md` guide) all
  construct `LoraConfig(r=args.rank, lora_alpha=args.rank, ...)` — a
  consistent, deliberate scaling=1.0 convention across their whole
  example suite. This governs DPO/Polish (diffusers-based).
- **Unsloth's own `FastVisionModel.get_peft_model()` API** — which
  `train_critic_qlora.py`'s config fields (`finetune_vision_layers`/
  `finetune_language_layers`/`finetune_attention_modules`/
  `finetune_mlp_modules`) are a literal match for — has its own official
  example using `r=16, lora_alpha=16` with the explicit comment
  "Recommended alpha == r at least." This governs Critic.

Reverted Critic and DPO back to `alpha=rank` (scaling=1.0), now backed
by real, model-API-specific citations instead of an unresearched
cross-tier consistency assumption. Planner was never touched (its
alpha=2x wasn't the flagged item, and general LLM-LoRA community
practice does support that ratio as a reasonable, if contested, choice).

## Verification

`torch` unavailable in this environment throughout. Verified: the
activation-memory arithmetic directly (plain Python), the
`enable_input_require_grads()` ordering against real upstream GitHub
issues (not memory), `SketchModelConfig`'s backward compatibility for
the new field empirically via Python's real `pickle`/`dataclasses`
modules (old checkpoints missing the field still correctly report the
default via the class-level attribute fallback dataclasses set for
simple immutable defaults — this was specifically checked because it's
exactly the kind of assumption that's broken code elsewhere in this
project). All files `ast.parse`-clean.
