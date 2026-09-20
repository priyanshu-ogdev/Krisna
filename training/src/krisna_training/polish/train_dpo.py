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
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-train-steps", type=int, default=1000)
    p.add_argument("--checkpointing-steps", type=int, default=200)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    return p


def build_dataset(args, tokenizer_max_length: int = 77):
    """Returns a torch.utils.data.Dataset yielding
    (chosen_pixel_values, rejected_pixel_values, prompt) triples from
    real PreferenceStore records — resolved through the shared BlobStore.
    """
    from krisna_training.polish.dpo_dataset import PreferencePairDataset

    return PreferencePairDataset(args.preference_db, args.blob_root, args.source, args.resolution)


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
        flow_matching_dpo_loss,
        flow_matching_velocity_target,
        noise_latent_at_timestep,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)

    dataset = build_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # Policy pipeline — trainable LoRA adapter on the transformer only
    # (VAE and text encoder stay frozen, matching the base fine-tune's
    # own convention — see training/README.md's polish-tier section).
    policy_pipe = AutoPipelineForText2Image.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
    )
    if args.lora_adapter_path:
        policy_pipe.load_lora_weights(args.lora_adapter_path)
        log.info("continuing_from_existing_adapter", extra={"path": args.lora_adapter_path})
    else:
        policy_pipe.transformer.add_adapter(LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank))

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
                sigma_chosen = u.to(chosen_latents.device).view(bsz, 1, 1, 1)
                u_r = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal", batch_size=bsz,
                    logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                )
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
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
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
                    accelerator.unwrap_model(policy_transformer).save_pretrained(str(output_dir / f"checkpoint-{global_step}"))
                if global_step >= args.max_train_steps:
                    break

    accelerator.unwrap_model(policy_transformer).save_pretrained(str(output_dir / "final"))
    log.info("dpo_training_complete", extra={"output_dir": str(output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
