"""Diffusion-DPO training for Z-Image-Turbo — CLOSES THE PREVIOUSLY-FLAGGED
GAP: `training/dpo/` built and exported real preference-pair data
(data-forge's human-labeled Pick-a-Pic v2/HPDv2/DesignSense-10k/
DesignPref), but nothing consumed it. This is that consumer.

Uses flow_matching_dpo_loss (see dpo_loss.py's module docstring for the
full derivation and citations — adapted from Wallace et al.'s Diffusion-
DPO, following MotionFlux's published velocity-prediction substitution
for flow-matching models). Reads real PreferencePair records from
krisna_training.dpo.preference_store.PreferenceStore, resolving
chosen_ref/rejected_ref through the shared BlobStore.

What this does NOT do: this is a real, structurally-correct training
loop — optimizer, gradient accumulation, reference-model freezing,
checkpointing — but it has NOT been run against a real GPU or validated
to produce a model that's actually better aligned. The loss math is
verified (see tests/training/test_dpo_loss.py); the end-to-end training
dynamics on real Z-Image-Turbo weights are not, and can't be from this
environment. Treat the defaults (beta, learning rate, fm_anchor_weight)
as starting points requiring a real sweep, not tuned values.

What's verified vs. flagged-as-unverified in the API surface used here,
checked directly rather than assumed: `transformer.add_adapter(LoraConfig
(...))` on a diffusers model is confirmed real (a diffusers maintainer's
own bug-report example calls this exact method on
`CogVideoXTransformer3DModel` — github.com/huggingface/peft/issues/2494),
and `save_pretrained()` as the save-side call once a model is PEFT-
wrapped is confirmed real (PEFT's own quickstart uses exactly this).
`encode_prompt()`'s return shape and the transformer's forward-call
signature are marked inline below as genuinely pipeline-specific and not
independently checked against Z-Image-Turbo's real pipeline class — same
"flag rather than guess" discipline as this project's other unverified
API surfaces (see `polish_quality_backend.py`'s `strength` kwarg note).

UPGRADE (this review pass) — why --train-batch-size stays at 1, with the
actual arithmetic rather than a hand-wave, so a real operator has a real
starting point to tune from: this project's own PRD (§7.3) reports
Z-Image-Turbo's FULL bf16 pipeline (transformer + VAE + text encoder(s))
at ~14GB resident for INFERENCE. Training keeps that same ~14GB resident
policy pipeline PLUS transiently swaps the ~14GB frozen reference
transformer onto the GPU for its brief forward call each micro-batch
(ref_transformer.to(accelerator.device) below) — so peak VRAM during
that window is already ~28GB of the real A6000 target's 48GB (PRD §3)
before counting the LoRA adapter's own (small — rank 32 on 4 attention
projections is a few tens of MB) optimizer state, or activation memory
for the batch itself. Gradient checkpointing is enabled on the policy
transformer specifically to keep that remaining ~20GB headroom mostly
available for activations rather than fixed weight/optimizer cost. This
arithmetic suggests batch_size=2 is PLAUSIBLE on the real target
hardware, not that it's unsafe — but "plausible from arithmetic" and
"verified to not OOM" are different claims, and only the second one is
safe to silently ship as a new default from an environment with no GPU
to actually test against. Raise this deliberately, watching real memory
usage, rather than treating the arithmetic above as a green light.

Usage:
    python -m krisna_training.polish.train_dpo \\
        --preference-db ./krisna_preference_pairs.db \\
        --pretrained-model-name-or-path Tongyi-MAI/Z-Image-Turbo \\
        --output-dir ./models/dpo_checkpoints/stage1_general \\
        --source pickapic_v2 --source hpdv2 \\
        --fm-anchor-weight 0.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger("krisna_training.polish.train_dpo")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preference-db", required=True, help="Path to the PreferenceStore sqlite db (see scripts/training/sync_from_data_forge_dpo.sh).")
    p.add_argument("--blob-root", default="krisna_blobs", help="Root dir chosen_ref/rejected_ref blob:// paths resolve against.")
    p.add_argument("--pretrained-model-name-or-path", default="Tongyi-MAI/Z-Image-Turbo")
    p.add_argument("--lora-adapter-path", default=None, help="Existing LoRA adapter to continue from (e.g. the base fine-tune's output). If unset, DPO trains a fresh adapter on top of the base pretrained weights.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source", action="append", default=None, help="Restrict to specific preference sources (e.g. pickapic_v2). Repeatable. Default: all sources in the db.")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--beta", type=float, default=2000.0, help="DPO inverse-temperature — see dpo_loss.py's docstring for why this default is a starting point, not a tuned value.")
    p.add_argument("--fm-anchor-weight", type=float, default=0.0, help="Flow-matching anchor regularization weight (MotionFlux-style). 0.0 disables it.")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument(
        "--lr-warmup-steps", type=int, default=100,
        help=(
            "This training loop previously had no LR scheduler at all — "
            "plain constant AdamW(lr=learning_rate) from step 0, unlike "
            "every sibling config in this repo. A randomly-initialized "
            "LoRA adapter taking full-LR AdamW steps from step 0, against "
            "a DPO loss with beta=2000, is exactly what warmup protects "
            "against. Default 100 matches the sibling dreambooth script's "
            "value."
        ),
    )
    p.add_argument(
        "--lr-scheduler", type=str, default="constant_with_warmup",
        choices=["constant", "constant_with_warmup", "linear", "cosine"],
    )
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument(
        "--lora-target-modules", type=str,
        default="to_q,to_k,to_v,to_out.0",
        help=(
            "UPGRADE (bug found this review pass): the previous "
            "LoraConfig(r=..., lora_alpha=...) call passed NO "
            "target_modules at all. PEFT only auto-resolves "
            "target_modules for model classes in its own built-in "
            "text/vision-transformer mapping — a custom diffusers DiT "
            "transformer (Z-Image-Turbo) is not in that mapping, so this "
            "would raise `ValueError: Please specify target_modules` at "
            "the very first fresh-adapter run, before any training "
            "happens. Default here targets the single-stream attention "
            "block's q/k/v/out projections (Z-Image-Turbo is documented "
            "in this project's own citations as a SINGLE-stream DiT, "
            "unlike FLUX/SD3's dual-stream joint attention, so there is "
            "no separate add_q_proj/add_k_proj/add_v_proj/to_add_out set "
            "to target the way a FLUX dreambooth-LoRA script would) — "
            "this exact module-name convention (to_q/to_k/to_v/to_out.0) "
            "is what diffusers' own attention-processor classes use "
            "across its DiT-family pipelines. UNVERIFIED against "
            "Z-Image-Turbo's specific module names in this environment "
            "(no network access to inspect the real class) — same "
            "'flag rather than guess' discipline as this file's other "
            "unverified API surfaces; override with a comma-separated "
            "list if the real module names differ."
        ),
    )
    p.add_argument(
        "--lora-dropout", type=float, default=0.0,
        help=(
            "UPGRADE: LoRA dropout on the adapter's own low-rank "
            "layers (separate from the base model's own dropout, which "
            "stays frozen either way). 0.0 preserves prior behavior "
            "exactly. A small value (0.0-0.1) is a standard, cheap "
            "generalization lever worth sweeping on a preference set "
            "this small relative to the base model's pretraining "
            "corpus — set only if overfitting shows up in the "
            "train/val implicit-reward-margin gap (finding #3's new "
            "validation loop is what would surface this)."
        ),
    )
    p.add_argument(
        "--flip-prob", type=float, default=0.0,
        help=(
            "UPGRADE (reconsidered this review pass): per-pair, PAIRED "
            "horizontal flip (chosen and rejected always flipped "
            "together, never independently — see dpo_dataset.py's "
            "__getitem__ for why independent flips would corrupt the "
            "preference signal). Defaults to 0.0 (no behavior change): "
            "this is the final high-fidelity Polish stage, and a "
            "horizontal flip mirrors any on-screen UI text into "
            "unreadable backward glyphs — a real risk this project has "
            "no per-record text-density signal to guard against. Set to "
            "a small nonzero value (0.1-0.3) only once you've confirmed "
            "your preference corpus's text density tolerates it; the "
            "heldout validation split (--val-every) never applies it "
            "regardless of this flag."
        ),
    )
    p.add_argument(
        "--val-batch-size", type=int, default=None,
        help=(
            "UPGRADE: defaults to --train-batch-size if unset, but can "
            "safely be set HIGHER — validation runs entirely under "
            "torch.no_grad() (no autograd graph, no Adam moment "
            "buffers, no gradient-accumulation bookkeeping), so it has "
            "none of --train-batch-size's memory overhead and a larger "
            "value here is a real, low-risk way to make each validation "
            "pass more representative without touching the memory "
            "budget that --train-batch-size (deliberately left at 1 by "
            "default — see this file's module docstring) is "
            "conservative about."
        ),
    )
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-train-steps", type=int, default=1000)
    p.add_argument("--checkpointing-steps", type=int, default=200)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help=(
            "UPGRADE (finding #6): this loop previously had NO gradient "
            "clipping at all, unlike sketch/train.py (grad_clip: 1.0). "
            "DPO's loss is beta * (implicit reward margin) with "
            "beta=2000 by default — a large multiplier on an already-"
            "noisy quantity (a difference of two MSE differences), "
            "exactly the setup where one outlier batch produces a "
            "destabilizing gradient spike. Diffusion-DPO reference "
            "implementations clip for this reason. Set to 0 to disable."
        ),
    )
    p.add_argument(
        "--ema-decay", type=float, default=0.9995,
        help=(
            "UPGRADE (finding #5): EMA of the trainable LoRA parameters, "
            "checkpointed and used for the FINAL saved adapter (not for "
            "the loss/backward path, which always uses the live weights "
            "— standard EMA usage). A 2024 ablation comparing diffusion "
            "vs. token-based (MaskGIT-style) generative training found "
            "EMA 'beneficial almost universally' for diffusion models, "
            "vs. negligible-or-harmful for token-based ones — this tier "
            "(Z-Image-Turbo, flow-matching/diffusion family) is exactly "
            "the regime where it helps; the Sketch tier (MaskGIT-style) "
            "deliberately does NOT get EMA for the same reason (see "
            "sketch/train.py — no code change there is not an oversight)."
            " Set to 0 to disable."
        ),
    )
    p.add_argument("--val-fraction", type=float, default=0.05, help="Deterministic heldout fraction of preference pairs, matching the PRD's 5%% stratified holdout convention.")
    p.add_argument("--val-every", type=int, default=100, help="Run heldout implicit-reward-margin eval every N optimizer steps. 0 disables validation.")
    p.add_argument("--val-batches", type=int, default=20, help="Cap on validation batches per eval call — mirrors sketch/train.py's val_batches for the same reason (a monitoring signal, not a full pass every time).")
    p.add_argument(
        "--target-passes", type=float, default=None,
        help=(
            "UPGRADE (finding #4): when set, --max-train-steps is IGNORED "
            "and recomputed as round(target_passes * len(train_pairs) / "
            "(train_batch_size * gradient_accumulation_steps)) from the "
            "actual synced preference-pair count. max_train_steps=1000 "
            "against Pick-a-Pic v2 + HPDv2's combined ~1.8M-pair volume "
            "sees only ~0.2%% of available pairs at effective batch 4 — "
            "calibrated as a smoke-test default (Oxen.ai's own reference "
            "table calls 1000 steps 'quick test runs'), not scaled up "
            "once real preference-pair volume was known. Leave unset to "
            "keep --max-train-steps exactly as given."
        ),
    )
    return p


def build_dataset(args, split: str = "train"):
    """Returns a torch.utils.data.Dataset yielding
    (chosen_pixel_values, rejected_pixel_values, prompt) triples from
    real PreferenceStore records — resolved through the shared BlobStore.
    split="val" returns the deterministic heldout complement (finding #3).
    Validation never applies flip augmentation (flip_prob=0.0 regardless
    of --flip-prob) — augmentation exists to widen the TRAINING
    distribution; the heldout eval should measure real performance on
    the actual (unflipped) held-out data, matching sketch/train.py's own
    choice to force cfg_dropout_prob=0.0 for its validation loader.
    """
    from krisna_training.polish.dpo_dataset import PreferencePairDataset

    return PreferencePairDataset(
        args.preference_db, args.blob_root, args.source, args.resolution,
        split=split, val_fraction=args.val_fraction,
        flip_prob=args.flip_prob if split == "train" else 0.0,
    )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)

    import torch
    from accelerate import Accelerator
    from diffusers import AutoPipelineForText2Image
    from diffusers.training_utils import compute_density_for_timestep_sampling
    from peft import LoraConfig
    from torch.utils.data import DataLoader

    from krisna_training.polish.dpo_loss import (
        apply_flow_matching_shift,
        ema_warmup_decay,
        flow_matching_dpo_loss,
        flow_matching_velocity_target,
        noise_latent_at_timestep,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)

    # UPGRADE (finding #2, [Critical]): this is the single most severe
    # bug this review found. The documented, canonical path (./train.sh
    # --all) is supposed to run polish-default (base LoRA fine-tune)
    # THEN polish-dpo continuing from it (train.sh --list's own text:
    # "via diffusers' own dreambooth-LoRA script, THEN refined further
    # by polish-dpo"; PRD §6's model-stack table describes one
    # continuous "LoRA fine-tune + Diffusion-DPO" line). But both
    # shipped DPO configs default lora-adapter-path to null, and neither
    # train_polish_dpo.sh nor this script ever threaded the base
    # fine-tune's output_dir through automatically — so the documented
    # default run silently DPO-aligns the vanilla pretrained weights,
    # never the UI-domain-adapted ones, and throws away the base
    # fine-tune step with no error and no log line. Fixed here (not just
    # in the shell wrapper) so this is loud and correct regardless of
    # how train_dpo.py is invoked: if the caller didn't pass
    # --lora-adapter-path, auto-detect the base fine-tune's conventional
    # output_dir (training/configs/polish_stage1_default_lora.yaml's
    # output_dir) and use it, unless it genuinely doesn't exist yet — in
    # which case this is loud about falling back to a fresh adapter
    # rather than silently doing so.
    if args.lora_adapter_path is None:
        _DEFAULT_BASE_LORA_OUTPUT_DIR = "./checkpoints/polish_default_lora"
        base_lora_path = Path(_DEFAULT_BASE_LORA_OUTPUT_DIR)
        # BUG FIX (this review pass): must check for the SAME criterion
        # scripts/training/dpo_lora_autodetect.py uses (a real
        # *.safetensors file, matching what diffusers' dreambooth-LoRA
        # scripts actually save and what load_lora_weights() actually
        # needs) — not merely "the directory exists and has any file in
        # it at all". This layer is a defense-in-depth duplicate for
        # anyone invoking this module directly (bypassing
        # train_polish_dpo.sh's shell wrapper, which resolves this first
        # via that same helper) — using a looser check here than the
        # wrapper uses would mean a directory with only a stray log file
        # or a half-written/failed run passes this check, gets handed to
        # load_lora_weights(), and fails with a confusing diffusers-level
        # error instead of this file's own clear warning.
        if base_lora_path.exists() and any(base_lora_path.glob("*.safetensors")):
            args.lora_adapter_path = str(base_lora_path)
            log.warning(
                "auto_detected_base_lora_adapter",
                extra={
                    "path": args.lora_adapter_path,
                    "note": "--lora-adapter-path was not set; auto-continuing DPO from the "
                             "base fine-tune's conventional output_dir. Pass "
                             "--lora-adapter-path explicitly to silence this, or to "
                             "point at a different checkpoint.",
                },
            )
        else:
            log.warning(
                "dpo_training_fresh_adapter_no_base_finetune",
                extra={
                    "checked_path": _DEFAULT_BASE_LORA_OUTPUT_DIR,
                    "note": "No --lora-adapter-path given and no base fine-tune checkpoint "
                             "found at the conventional output_dir. DPO will train a FRESH "
                             "adapter on top of the raw pretrained weights, NOT the "
                             "UI-domain-adapted ones — this means the base fine-tune stage "
                             "is being skipped for this run. This is intentional only if "
                             "that is really what you want; otherwise run polish-default "
                             "first (./train.sh polish-default) before polish-dpo.",
                },
            )

    dataset = build_dataset(args, split="train")
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    if args.target_passes is not None:
        effective_batch = max(1, args.train_batch_size * args.gradient_accumulation_steps)
        scaled_steps = max(1, round(args.target_passes * len(dataset) / effective_batch))
        log.info(
            "max_train_steps_rescaled_to_corpus",
            extra={
                "configured_max_train_steps": args.max_train_steps, "available_pairs": len(dataset),
                "effective_batch": effective_batch, "target_passes": args.target_passes,
                "scaled_max_train_steps": scaled_steps,
            },
        )
        args.max_train_steps = scaled_steps
    else:
        effective_batch = max(1, args.train_batch_size * args.gradient_accumulation_steps)
        approx_passes = (args.max_train_steps * effective_batch) / max(1, len(dataset))
        if approx_passes < 0.5:
            log.warning(
                "max_train_steps_may_be_undersized",
                extra={
                    "max_train_steps": args.max_train_steps, "available_pairs": len(dataset),
                    "approx_passes_covered": round(approx_passes, 4),
                    "note": "max_train_steps covers well under half a pass over the available "
                             "preference pairs at this effective batch size — consider "
                             "--target-passes instead of a fixed step count.",
                },
            )

    val_dataloader = None
    if args.val_every > 0:
        try:
            val_dataset = build_dataset(args, split="val")
            val_batch_size = args.val_batch_size if args.val_batch_size is not None else args.train_batch_size
            val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        except ValueError as e:
            log.warning("dpo_validation_split_empty", extra={"error": str(e)})

    # Policy pipeline — trainable LoRA adapter on the transformer only
    # (VAE and text encoder stay frozen, matching the base fine-tune's
    # own convention — see training/README.md's polish-tier section).
    policy_pipe = AutoPipelineForText2Image.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
    )

    # UPGRADE (quality/consistency finding, this review pass): read the
    # REAL pretrained scheduler's configured resolution-dependent shift
    # (see dpo_loss.py's apply_flow_matching_shift docstring) instead of
    # sampling raw, unshifted timesteps. Defaults to 1.0 (no-op, exactly
    # the previous behavior) if the scheduler doesn't declare one, so
    # this can never silently change behavior for a scheduler that
    # genuinely has no shift.
    fm_shift = float(getattr(policy_pipe.scheduler.config, "shift", 1.0))
    log.info("flow_matching_shift_detected", extra={"shift": fm_shift, "resolution": args.resolution})

    if args.lora_adapter_path:
        policy_pipe.load_lora_weights(args.lora_adapter_path)
        log.info("continuing_from_existing_adapter", extra={"path": args.lora_adapter_path})
        # BUG FOUND THIS REVIEW PASS: `load_lora_weights()` is diffusers'
        # INFERENCE-oriented loader — it does not guarantee the loaded
        # LoRA parameters come back with requires_grad=True. If they
        # don't, `[p for p in policy_transformer.parameters() if
        # p.requires_grad]` below silently (or, once AdamW rejects an
        # empty list, loudly-but-confusingly) trains NOTHING — precisely
        # on the "continue DPO from the base fine-tune" path that finding
        # #2's fix specifically exists to exercise. Explicitly force
        # requires_grad=True on every loaded LoRA parameter rather than
        # trusting the loader's default, and verify at least one
        # parameter actually is trainable before spending any compute.
        for name, param in policy_pipe.transformer.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad_(True)
    else:
        policy_pipe.transformer.add_adapter(
            LoraConfig(
                r=args.lora_rank, lora_alpha=args.lora_rank,
                lora_dropout=args.lora_dropout,
                target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
            )
        )
    _n_trainable = sum(p.numel() for p in policy_pipe.transformer.parameters() if p.requires_grad)
    if _n_trainable == 0:
        raise RuntimeError(
            "No trainable parameters found on policy_pipe.transformer after adapter setup — "
            "LoRA loading/attachment silently produced zero trainable weights. Check "
            "--lora-target-modules against the real module names on this pipeline's "
            "transformer class, or that --lora-adapter-path points at a real LoRA checkpoint."
        )
    log.info("policy_adapter_ready", extra={"trainable_params": _n_trainable})

    # Reference model — frozen COPY of the same base weights (NOT the
    # same object as the policy — the whole DPO formulation depends on
    # comparing policy-vs-reference on identical inputs, which requires
    # two independently-forward-passable models). If continuing from an
    # existing LoRA adapter, the reference is that already-fine-tuned
    # checkpoint frozen in place — DPO then aligns further from there,
    # not from the raw base weights.
    ref_pipe = AutoPipelineForText2Image.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
    )
    if args.lora_adapter_path:
        ref_pipe.load_lora_weights(args.lora_adapter_path)
    ref_pipe.transformer.requires_grad_(False)
    ref_pipe.transformer.eval()
    ref_transformer = ref_pipe.transformer

    # UPGRADE (A6000 48GB training-memory audit): only ref_pipe.transformer
    # is ever used below — prompt_embeds come from policy_pipe.encode_prompt(),
    # and ref_pipe's own VAE/text encoder(s) are never called. Loading the
    # whole pipeline built a second, entirely unused copy of them purely
    # to reach `.transformer` — freed immediately.
    del ref_pipe.vae
    if hasattr(ref_pipe, "text_encoder"):
        del ref_pipe.text_encoder
    if hasattr(ref_pipe, "text_encoder_2"):
        del ref_pipe.text_encoder_2
    if hasattr(ref_pipe, "text_encoder_3"):
        del ref_pipe.text_encoder_3
    del ref_pipe
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Holding BOTH full bf16 transformer copies resident on GPU
    # simultaneously is DPO's dominant VRAM cost — documented in Reg-DPO
    # (arXiv:2511.01450, §5 "Model Offloading for Frozen Modules": moving
    # a frozen reference model's weights to CPU between forward passes
    # cuts peak memory by ~10GB). ref_transformer needs no gradients and
    # is only used inside `with torch.no_grad()` below, so it stays on
    # CPU by default and is swapped to GPU only for its brief forward call.
    ref_transformer.to("cpu")

    vae = policy_pipe.vae
    vae.requires_grad_(False)
    vae.to(accelerator.device)
    # Move text encoders to accelerator device so encode_prompt runs on GPU
    for attr in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
        te = getattr(policy_pipe, attr, None)
        if te is not None and hasattr(te, "to"):
            te.to(accelerator.device)
            te.requires_grad_(False)
    policy_transformer = policy_pipe.transformer

    if hasattr(policy_transformer, "enable_gradient_checkpointing"):
        policy_transformer.enable_gradient_checkpointing()
        log.info("policy_transformer_gradient_checkpointing_enabled")
    else:
        log.warning("policy_transformer_gradient_checkpointing_unavailable")

    optimizer = torch.optim.AdamW(
        [p for p in policy_transformer.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    from diffusers.optimization import get_scheduler

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # ref_transformer removed from accelerator.prepare() — that call
    # would move/wrap it onto the accelerator device immediately,
    # undoing the CPU-resident placement above before training starts.
    policy_transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy_transformer, optimizer, dataloader, lr_scheduler
    )

    # UPGRADE (finding #5): EMA of the trainable (LoRA) parameters.
    # Standard usage — the live weights are always what's optimized and
    # used for the forward passes below; the EMA shadow is only read at
    # checkpoint/save time, giving the diffusion-family regime's
    # documented benefit without touching the loss/backward path at all.
    ema_params = None
    if args.ema_decay > 0.0:
        trainable_params = [p for p in accelerator.unwrap_model(policy_transformer).parameters() if p.requires_grad]
        ema_params = [p.detach().clone() for p in trainable_params]

    def _update_ema(step: int) -> None:
        if ema_params is None:
            return
        # UPGRADE (this review pass): warmup the decay rather than pinning
        # it at args.ema_decay from step 0 — see dpo_loss.py's
        # ema_warmup_decay docstring for why a fixed 0.9995-class decay
        # is close to a no-op improvement over ~1000-4000 step DPO runs
        # without this.
        decay = ema_warmup_decay(step, args.ema_decay)
        trainable_params = [p for p in accelerator.unwrap_model(policy_transformer).parameters() if p.requires_grad]
        with torch.no_grad():
            for ema_p, p in zip(ema_params, trainable_params):
                ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    def _save_adapter(path: str, use_ema: bool) -> None:
        unwrapped = accelerator.unwrap_model(policy_transformer)
        if use_ema and ema_params is not None:
            trainable_params = [p for p in unwrapped.parameters() if p.requires_grad]
            originals = [p.detach().clone() for p in trainable_params]
            with torch.no_grad():
                for p, ema_p in zip(trainable_params, ema_params):
                    p.copy_(ema_p)
            unwrapped.save_pretrained(path)
            with torch.no_grad():
                for p, orig in zip(trainable_params, originals):
                    p.copy_(orig)
        else:
            unwrapped.save_pretrained(path)

    def _run_dpo_validation() -> dict | None:
        """Heldout implicit-reward-margin eval (finding #3) — the metric
        dpo_loss.py already computes per-batch but this loop previously
        discarded entirely; no validation signal existed anywhere for
        the DPO stage. model.eval()/no_grad, never affects optimization.
        """
        if val_dataloader is None:
            return None
        policy_transformer.eval()
        margins, losses = [], []
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                if i >= args.val_batches:  # cap eval cost, same discipline as sketch/train.py's val_batches
                    break
                chosen_latents = vae.encode(batch["chosen_pixel_values"].to(vae.dtype).to(accelerator.device)).latent_dist.sample()
                rejected_latents = vae.encode(batch["rejected_pixel_values"].to(vae.dtype).to(accelerator.device)).latent_dist.sample()
                chosen_latents = chosen_latents * vae.config.scaling_factor
                rejected_latents = rejected_latents * vae.config.scaling_factor
                bsz = chosen_latents.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                )
                u = apply_flow_matching_shift(u, fm_shift)
                sigma_c = u.to(chosen_latents.device).view(bsz, 1, 1, 1)
                u_r = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                )
                u_r = apply_flow_matching_shift(u_r, fm_shift)
                sigma_r = u_r.to(rejected_latents.device).view(bsz, 1, 1, 1)
                noise_c, noise_r = torch.randn_like(chosen_latents), torch.randn_like(rejected_latents)
                noisy_c = noise_latent_at_timestep(chosen_latents, noise_c, sigma_c)
                noisy_r = noise_latent_at_timestep(rejected_latents, noise_r, sigma_r)
                target_c = flow_matching_velocity_target(chosen_latents, noise_c)
                target_r = flow_matching_velocity_target(rejected_latents, noise_r)
                prompt_embeds = policy_pipe.encode_prompt(batch["prompt"])
                if isinstance(prompt_embeds, torch.Tensor):
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                elif isinstance(prompt_embeds, (tuple, list)):
                    prompt_embeds = tuple(t.to(accelerator.device) if isinstance(t, torch.Tensor) else t for t in prompt_embeds)
                pv_c = policy_transformer(noisy_c, sigma_c.flatten(), prompt_embeds).sample
                pv_r = policy_transformer(noisy_r, sigma_r.flatten(), prompt_embeds).sample
                ref_transformer.to(accelerator.device)
                rv_c = ref_transformer(noisy_c, sigma_c.flatten(), prompt_embeds).sample
                rv_r = ref_transformer(noisy_r, sigma_r.flatten(), prompt_embeds).sample
                ref_transformer.to("cpu")
                out = flow_matching_dpo_loss(
                    policy_v_chosen=pv_c, policy_v_rejected=pv_r, ref_v_chosen=rv_c, ref_v_rejected=rv_r,
                    target_v_chosen=target_c, target_v_rejected=target_r, beta=args.beta, fm_anchor_weight=args.fm_anchor_weight,
                )
                margins.append(out.implicit_reward_margin.item())
                losses.append(out.loss.item())
        policy_transformer.train()
        if not margins:
            return None
        return {
            "val_loss": round(sum(losses) / len(losses), 4),
            "val_implicit_reward_margin": round(sum(margins) / len(margins), 4),
            "val_batches": len(margins),
        }

    global_step = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(policy_transformer):
                chosen_latents = vae.encode(batch["chosen_pixel_values"].to(vae.dtype)).latent_dist.sample()
                rejected_latents = vae.encode(batch["rejected_pixel_values"].to(vae.dtype)).latent_dist.sample()
                chosen_latents = chosen_latents * vae.config.scaling_factor
                rejected_latents = rejected_latents * vae.config.scaling_factor

                bsz = chosen_latents.shape[0]
                # Independent noise/timestep draws per side of the pair —
                # matching Wallace et al.'s own formulation, not a shared
                # draw (see dpo_loss.py's docstring for why).
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal", batch_size=bsz,
                    logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                )
                u = apply_flow_matching_shift(u, fm_shift)
                sigma_chosen = u.to(chosen_latents.device).view(bsz, 1, 1, 1)
                u_r = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal", batch_size=bsz,
                    logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                )
                u_r = apply_flow_matching_shift(u_r, fm_shift)
                sigma_rejected = u_r.to(rejected_latents.device).view(bsz, 1, 1, 1)

                noise_chosen = torch.randn_like(chosen_latents)
                noise_rejected = torch.randn_like(rejected_latents)
                noisy_chosen = noise_latent_at_timestep(chosen_latents, noise_chosen, sigma_chosen)
                noisy_rejected = noise_latent_at_timestep(rejected_latents, noise_rejected, sigma_rejected)
                target_chosen = flow_matching_velocity_target(chosen_latents, noise_chosen)
                target_rejected = flow_matching_velocity_target(rejected_latents, noise_rejected)

                # UNVERIFIED against Z-Image-Turbo's specific pipeline
                # class — encode_prompt()'s return shape genuinely varies
                # across diffusers pipelines (some return a single
                # tensor, many SD3/FLUX-lineage pipelines return a tuple
                # of (prompt_embeds, pooled_prompt_embeds) or similar).
                # Z-Image-Turbo's live pipeline code was not directly
                # inspected for this script — same "flag rather than
                # guess" discipline as polish_quality_backend.py's
                # `strength` kwarg elsewhere in this project. Check the
                # real return shape (`policy_pipe.encode_prompt.__doc__`
                # or the pipeline source) before trusting this call as
                # written; the fix is almost certainly just unpacking a
                # tuple here rather than a deeper problem.
                prompt_embeds = policy_pipe.encode_prompt(batch["prompt"])
                if isinstance(prompt_embeds, torch.Tensor):
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                elif isinstance(prompt_embeds, (tuple, list)):
                    prompt_embeds = tuple(
                        t.to(accelerator.device) if isinstance(t, torch.Tensor) else t
                        for t in prompt_embeds
                    )

                # Same caveat: the transformer forward signature
                # (positional args, `.sample` vs a plain tensor return)
                # is pipeline-specific. This matches the general
                # diffusers DiT-family calling convention
                # (hidden_states, timestep, encoder_hidden_states) with a
                # `.sample` output — verify against Z-Image-Turbo's real
                # transformer class before a production run.
                policy_v_chosen = policy_transformer(noisy_chosen, sigma_chosen.flatten(), prompt_embeds).sample
                policy_v_rejected = policy_transformer(noisy_rejected, sigma_rejected.flatten(), prompt_embeds).sample
                with torch.no_grad():
                    # Swap ref_transformer onto the accelerator device only
                    # for this forward call, then back to CPU immediately.
                    ref_transformer.to(accelerator.device)
                    ref_v_chosen = ref_transformer(noisy_chosen, sigma_chosen.flatten(), prompt_embeds).sample
                    ref_v_rejected = ref_transformer(noisy_rejected, sigma_rejected.flatten(), prompt_embeds).sample
                    ref_transformer.to("cpu")

                out = flow_matching_dpo_loss(
                    policy_v_chosen=policy_v_chosen, policy_v_rejected=policy_v_rejected,
                    ref_v_chosen=ref_v_chosen, ref_v_rejected=ref_v_rejected,
                    target_v_chosen=target_chosen, target_v_rejected=target_rejected,
                    beta=args.beta, fm_anchor_weight=args.fm_anchor_weight,
                )

                accelerator.backward(out.loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0.0:
                    # UPGRADE (finding #6): previously no clipping at all in
                    # this loop. Only meaningful once grads are fully
                    # accumulated (accelerator.sync_gradients), matching
                    # accelerate's own documented pattern for combining
                    # clip_grad_norm_ with gradient accumulation.
                    accelerator.clip_grad_norm_(
                        [p for p in policy_transformer.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                _update_ema(global_step)
                global_step += 1
                if args.val_every > 0 and global_step % args.val_every == 0:
                    val_metrics = _run_dpo_validation()
                    if val_metrics is not None:
                        log.info("dpo_val_step", extra={"step": global_step, **val_metrics})
                if global_step % 10 == 0:
                    log.info(
                        "dpo_step",
                        extra={
                            "step": global_step,
                            "loss": out.loss.item(),
                            "dpo_term": out.dpo_term.item(),
                            "fm_anchor_term": out.fm_anchor_term.item(),
                            # The real signal to watch, per dpo_loss.py's
                            # docstring — should trend positive and
                            # growing, not just "loss goes down".
                            "implicit_reward_margin": out.implicit_reward_margin.item(),
                        },
                    )
                if global_step % args.checkpointing_steps == 0:
                    # CONFIRMED via direct research, not assumed:
                    # transformer.add_adapter(LoraConfig(...)) is real —
                    # a diffusers maintainer's own bug-report example
                    # shows this exact call on a diffusers transformer
                    # model (github.com/huggingface/peft/issues/2494).
                    # save_lora_adapter() as its save-side counterpart was
                    # NOT found confirmed anywhere in the same research
                    # pass — only load_lora_adapter() (PeftAdapterMixin)
                    # and the pipeline-level save_lora_weights() are
                    # documented. Using save_pretrained() instead, which
                    # IS directly confirmed (PEFT's own quickstart: "now
                    # perform training... then save the model:
                    # model.save_pretrained(...)") as the standard save
                    # call once add_adapter()/get_peft_model() has
                    # PEFT-wrapped a model.
                    # Saved from the EMA shadow when enabled (finding #5)
                    # — checkpoints are a deployable artifact, not a
                    # debugging snapshot, so they should reflect the
                    # smoothed weights the ablation found beneficial for
                    # this (diffusion) tier, not the noisier live weights.
                    _save_adapter(str(output_dir / f"checkpoint-{global_step}"), use_ema=True)
                if global_step >= args.max_train_steps:
                    break

    _save_adapter(str(output_dir / "final"), use_ema=True)
    log.info("dpo_training_complete", extra={"output_dir": str(output_dir), "used_ema": ema_params is not None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
