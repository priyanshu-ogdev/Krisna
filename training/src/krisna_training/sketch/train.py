"""Sketch tier training loop. Single-GPU (matches the whole project's
single-RTX-A6000 design — no distributed training here), bf16 autocast,
AdamW + cosine LR schedule with warmup, checkpointing in the exact format
inference/maskgit_model.py's MaskGITSketchModel.from_checkpoint expects
(see checkpoint_io.py).

Run via scripts/train_sketch_stage1.sh / train_sketch_stage2.sh, or:
    python -m krisna_training.sketch.train --config configs/sketch_train_stage1_256.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("krisna_training.sketch.train")


@dataclass
class TrainConfig:
    manifest_path: str
    output_dir: str
    grid_h: int = 16
    grid_w: int = 16
    vocab_size: int = 16384
    hidden_dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    prompt_dim: int = 768              # CLIP ViT-L/14 text-embedding dim by default
    # UPGRADE (SOTA research pass): forwarded to SketchModelConfig by
    # build_model_config() below. See that class's docstring for the
    # full citation trail. false/10000.0 = exact prior behavior.
    use_2d_rope: bool = False
    rope_base: float = 10000.0
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.96            # Chang et al. 2022 (MaskGIT) use Adam
                                         # (beta1=0.9, beta2=0.96), a lower
                                         # second-moment decay than PyTorch's
                                         # default 0.999 — a real stabilization
                                         # choice for masked-token transformer
                                         # training, not an arbitrary pick.
                                         # Previously this fell through to
                                         # AdamW's default 0.999 unintentionally
                                         # (no explicit betas= was passed at
                                         # all) — caught during a design-sync
                                         # review, not a considered deviation.
    warmup_steps: int = 1000
    caption_mix_ratio: float = 0.95   # see dataset.py::SketchTokenDataset
    cfg_dropout_prob: float = 0.1     # see make_collate_fn
    total_steps: int = 100_000
    critic_loss_weight: float = 0.5
    log_every: int = 50
    checkpoint_every: int = 2000
    grad_clip: float = 1.0
    resume_from: str | None = None
    init_from: str | None = None       # progressive training: load weights from a smaller-grid checkpoint
    num_workers: int = 4
    seed: int = 42
    use_gradient_checkpointing: bool = False   # see model.py's build_model() docstring

    # UPGRADE (generalization, this review pass): effective batch size
    # was previously fixed at exactly `batch_size` (32 by default) with
    # no way to raise it without more VRAM. A larger effective batch
    # reduces gradient-estimate noise for a from-scratch 44M-param
    # transformer trained on a 100K-500K image corpus (PRD §8.3) — the
    # exact regime where noisy small-batch gradients hurt convergence
    # and final generalization most. 1 preserves prior behavior exactly
    # (no accumulation). Raise this rather than `batch_size` itself when
    # VRAM is the constraint, matching how train_dpo.py already does
    # gradient accumulation via Accelerate.
    gradient_accumulation_steps: int = 1

    # --- UPGRADE: real validation loop (finding #3) ---
    # data-forge's s09_heldout stage carves out a 5% stratified holdout
    # specifically so training can measure generalization instead of only
    # training loss (a poor proxy for a masked-token objective — a model
    # can drive training loss down by memorizing the ~1-4% of the corpus
    # it actually sees at max_train_steps=1000-scale step budgets without
    # this ever showing up as a token-loss regression). Point this at the
    # manifest.jsonl produced by syncing that heldout split through the
    # same data_forge_bridge path as the training manifest.
    val_manifest_path: str | None = None
    val_every: int = 1000
    val_batches: int = 20              # cap eval cost; heldout loss is a monitoring signal, not a full pass every time

    # --- UPGRADE: corpus-scale-aware step count (finding #4) ---
    # max_train_steps=1000-class defaults in the shipped configs were
    # calibrated against small (hundreds-of-images) style-transfer LoRA
    # guidance, not against this project's own PRD §8.3 target of
    # 100K-500K images. "Total steps = (Images * Repetitions * Epochs) /
    # Batch Size" (standard LoRA/diffusion step-count guidance) scales
    # with dataset size, not a fixed constant. Rather than hand-tune
    # total_steps per corpus snapshot, set target_epochs and this
    # computes total_steps from the ACTUAL synced dataset length at
    # train-time — self-correcting as the corpus grows, matching how
    # sync_to_training.py already prints real corpus counts once known.
    # None preserves the exact configured total_steps (opt-in, so this
    # never silently changes a deliberately-set value).
    target_epochs: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)


def build_model_config(cfg: TrainConfig):
    from krisna_training.sketch.model import SketchModelConfig

    return SketchModelConfig(
        vocab_size=cfg.vocab_size,
        grid_h=cfg.grid_h,
        grid_w=cfg.grid_w,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        prompt_dim=cfg.prompt_dim,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        use_2d_rope=cfg.use_2d_rope,
        rope_base=cfg.rope_base,
    )


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`. Restores run-to-run reproducibility for
    the caption-mix (dataset.py) and mask ratio/position (masking.py,
    called from collate_fn below) draws, both of which use Python's global
    `random` module. CPython's `random` module auto-reseeds itself from OS
    entropy after every fork (os.register_at_fork, since Python 3.9) —
    which means sibling DataLoader workers were never actually correlated
    (a prior version of this fix wrongly assumed they were), but it does
    mean the SAME cfg.seed produces DIFFERENT draws on separate runs,
    since that auto-reseed discards whatever seed was set beforehand.
    Fixed by explicitly reseeding from torch's already-correct,
    already-deterministic worker_info.seed.
    """
    import random

    import numpy as np
    import torch

    worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_collate_fn(mask_token_id: int, text_embedder, prompt_dim: int, cfg_dropout_prob: float = 0.1):
    """cfg_dropout_prob: probability of replacing a sample's real text
    embedding with the zero/null embedding, independent of which caption
    (dense or source) was used. Standard classifier-free-guidance
    conditioning dropout (Ho & Salimans, 2022, "Classifier-Free Diffusion
    Guidance") — training the model to also handle unconditional
    generation is what makes CFG usable at inference at all, and 10% is
    the commonly-used default in the text-to-image literature this
    project's own citations doc already draws on (e.g. Imagen, Saharia et
    al. 2022). Previously this project had NO conditioning-dropout
    mechanism anywhere — the model only ever saw real captions during
    training, meaning inference-time CFG (if used) would be running on a
    model that never learned the unconditional branch it needs. Caught
    during a design-sync review, not a considered omission — see
    docs/review/10_synthetic_data_generalization_fix.md.
    """
    def collate(batch):
        import torch

        from krisna_training.sketch.masking import apply_random_mask

        token_lists = [b["tokens"] for b in batch]
        captions = [b["caption"] for b in batch]

        masked_list, target_list, mask_pos_list = [], [], []
        for tokens in token_lists:
            masked, positions = apply_random_mask(tokens, mask_token_id)
            masked_list.append(masked)
            target_list.append(tokens)
            mask_pos_list.append(positions)

        tokens_tensor = torch.tensor(masked_list, dtype=torch.long)
        targets_tensor = torch.tensor(target_list, dtype=torch.long)
        mask_tensor = torch.zeros_like(tokens_tensor, dtype=torch.float)
        for i, positions in enumerate(mask_pos_list):
            mask_tensor[i, positions] = 1.0

        if text_embedder is not None:
            embeds = torch.cat([text_embedder.embed_text(c or "UI design") for c in captions], dim=0)
            if cfg_dropout_prob > 0:
                drop = torch.rand(embeds.shape[0]) < cfg_dropout_prob
                embeds[drop] = 0.0
        else:
            # No embedder configured — zero conditioning (still trains the
            # unconditional token-filling objective, just without style
            # steering). Useful for a quick smoke run without downloading CLIP.
            embeds = torch.zeros(len(batch), prompt_dim)

        return tokens_tensor, mask_tensor, targets_tensor, embeds

    return collate


def train(cfg: TrainConfig) -> None:
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    from krisna_training.sketch.checkpoint_io import (
        load_checkpoint_for_resume,
        save_checkpoint,
    )
    from krisna_training.sketch.dataset import SketchTokenDataset
    from krisna_training.sketch.losses import compute_loss
    from krisna_training.sketch.model import build_model

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("no_cuda_available", extra={"note": "training on CPU will be extremely slow"})

    model_cfg = build_model_config(cfg)
    dataset = SketchTokenDataset(cfg.manifest_path, cfg.grid_h, cfg.grid_w, cfg.caption_mix_ratio)
    effective_batch = max(1, cfg.batch_size * cfg.gradient_accumulation_steps)

    # Corpus-scale-aware step count (finding #4). Computed from the real,
    # just-loaded dataset length — not a guess made ahead of sync time.
    # Uses effective_batch (not raw batch_size) so target_epochs means
    # what it says regardless of gradient_accumulation_steps.
    if cfg.target_epochs is not None:
        steps_per_epoch = max(1, len(dataset) // effective_batch)
        scaled_steps = max(1, round(cfg.target_epochs * steps_per_epoch))
        log.info(
            "total_steps_rescaled_to_corpus",
            extra={
                "configured_total_steps": cfg.total_steps, "dataset_size": len(dataset),
                "effective_batch": effective_batch, "steps_per_epoch": steps_per_epoch,
                "target_epochs": cfg.target_epochs, "scaled_total_steps": scaled_steps,
            },
        )
        cfg.total_steps = scaled_steps
    else:
        approx_epochs = (cfg.total_steps * effective_batch) / max(1, len(dataset))
        if approx_epochs < 1.0:
            log.warning(
                "max_train_steps_may_be_undersized",
                extra={
                    "total_steps": cfg.total_steps, "effective_batch": effective_batch,
                    "dataset_size": len(dataset), "approx_epochs_covered": round(approx_epochs, 3),
                    "note": "total_steps covers well under one full pass over the corpus at this "
                             "effective batch size — consider setting target_epochs instead of a "
                             "fixed step count.",
                },
            )

    val_dataset = None
    if cfg.val_manifest_path:
        val_dataset = SketchTokenDataset(
            cfg.val_manifest_path, cfg.grid_h, cfg.grid_w,
            # Heldout eval should reflect real inference-time caption style
            # exposure the same way training does, not silently deviate.
            cfg.caption_mix_ratio,
        )
        # BUG FOUND ON REVIEW: this empty-manifest guard existed in an
        # earlier draft of this validation loop but was dropped when
        # val_dataset/val_loader construction got split across two places
        # during the target_epochs refactor. Without it, an empty
        # val_manifest_path (0 usable records — e.g. data-forge's heldout
        # sync hasn't run yet, or a bad path) silently built an empty
        # DataLoader; _run_validation's own `n_batches = max(1, n_batches)`
        # guard against a ZeroDivisionError would then report a fake,
        # suspiciously-perfect "val_loss: 0.0" every single validation
        # step instead of skipping validation and saying so loudly.
        if len(val_dataset) == 0:
            log.warning(
                "val_manifest_empty",
                extra={"path": cfg.val_manifest_path, "note": "validation will be skipped entirely, not silently reported as a perfect val_loss"},
            )
            val_dataset = None

    text_embedder = None
    try:
        from krisna_inference.verifiers.common import get_clip_embedder

        text_embedder = get_clip_embedder()
        text_embedder.load()
    except ImportError:
        log.warning("no_clip_available", extra={"note": "training unconditionally — install verifier deps for real prompt conditioning"})

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(model_cfg.mask_token_id, text_embedder, cfg.prompt_dim, cfg.cfg_dropout_prob),
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate_fn(model_cfg.mask_token_id, text_embedder, cfg.prompt_dim, cfg_dropout_prob=0.0),
            drop_last=False,
        )

    start_step = 0
    resumed_optimizer_state = None
    if cfg.resume_from:
        model, resumed_model_cfg, start_step, resumed_optimizer_state = load_checkpoint_for_resume(cfg.resume_from)
        # BUG FOUND ON REVIEW: the training/val DataLoaders above were
        # already built (collate_fn closes over `model_cfg.mask_token_id`,
        # captured from the config FILE's fresh SketchModelConfig) before
        # this resume branch ever runs. If a resumed checkpoint's saved
        # config disagrees with the current YAML (vocab_size/grid changed
        # between runs, or the wrong checkpoint path given), masking would
        # silently use the WRONG mask_token_id against the actually-
        # resumed model — an out-of-vocabulary embedding lookup or a
        # mask_token_id the resumed model never trained on, with no error,
        # just quietly worse convergence. Fail loudly instead of resuming
        # into a mismatched setup.
        if (
            resumed_model_cfg.mask_token_id != model_cfg.mask_token_id
            or resumed_model_cfg.grid_h != model_cfg.grid_h
            or resumed_model_cfg.grid_w != model_cfg.grid_w
            # UPGRADE (SOTA research pass): use_2d_rope changes the actual
            # module graph (RotaryMultiheadSelfAttention layers + no
            # pos_embed parameter, vs. nn.TransformerEncoder + pos_embed)
            # — a mismatch here isn't a subtly-wrong-but-loadable config,
            # it's a state_dict key mismatch that load_state_dict(strict=True)
            # would fail on outright. Caught here for the same reason as
            # mask_token_id/grid above: fail with a clear message instead
            # of a confusing raw key-mismatch traceback three lines deeper.
            or resumed_model_cfg.use_2d_rope != model_cfg.use_2d_rope
        ):
            raise ValueError(
                f"Resume config mismatch: checkpoint at {cfg.resume_from} was trained with "
                f"mask_token_id={resumed_model_cfg.mask_token_id}, grid={resumed_model_cfg.grid_h}x"
                f"{resumed_model_cfg.grid_w}, use_2d_rope={resumed_model_cfg.use_2d_rope}, but the "
                f"current YAML config produces mask_token_id={model_cfg.mask_token_id}, "
                f"grid={model_cfg.grid_h}x{model_cfg.grid_w}, use_2d_rope={model_cfg.use_2d_rope}. "
                "Resuming would silently mask/collate against the wrong vocabulary/grid, or fail "
                "to load a mismatched architecture. Fix the YAML to match the checkpoint's real "
                "config, or use init_from (progressive-resolution init) instead of resume_from if "
                "you deliberately changed grid size."
            )
        model_cfg = resumed_model_cfg
        model.to(device)
        log.info("resumed", extra={"path": cfg.resume_from, "step": start_step})
    else:
        model = build_model(model_cfg)
        if cfg.init_from:
            _init_from_smaller_grid(model, model_cfg, cfg.init_from)
        model.to(device)

    # UPGRADE (generalization, this review pass): decoupled weight decay
    # — 1D parameters (biases, LayerNorm/RMSNorm weights) and embedding
    # tables were previously decayed at the same rate as every 2D weight
    # matrix. This is well-established practice across transformer
    # training (GPT-2, BERT, ViT all exclude these; see Loshchilov &
    # Hutter 2019 "Decoupled Weight Decay Regularization" — the AdamW
    # paper itself — and its widely-followed convention of decaying only
    # matrix-multiply weights). Decaying LayerNorm scale/bias toward zero
    # has no principled justification (it directly fights the norm's
    # learned rescaling) and decaying embedding rows shrinks token
    # representations toward the origin for tokens seen rarely per
    # epoch — a 44M-param from-scratch transformer on a 100K-500K image
    # corpus (PRD §8.3) is exactly the small-enough-to-matter regime
    # where this default has a real effect on generalization, not just a
    # theoretical nicety.
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "norm" in name.lower() or "embed" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2),
    )
    log.info(
        "optimizer_param_groups",
        extra={"decayed_params": sum(p.numel() for p in decay_params), "no_decay_params": sum(p.numel() for p in no_decay_params)},
    )

    # BUG FOUND THIS REVIEW PASS: the optimizer's Adam moment buffers
    # (m, v — the whole point of using Adam over plain SGD) were loaded
    # from the checkpoint by load_checkpoint_for_resume(), then silently
    # discarded (previously assigned to `_`). Every resume therefore
    # restarted Adam from cold moments — a real transient destabilization
    # right after every resume, worse the more often training is
    # resumed (long runs on preemptible/interruptible hardware being
    # exactly the case resume_from exists for). Restore it when shapes
    # match; degrade to a loud warning rather than crashing if an older
    # checkpoint's param-group structure doesn't match (e.g. saved
    # before this review's decay/no-decay param-group split existed).
    if resumed_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resumed_optimizer_state)
            log.info("optimizer_state_restored", extra={"step": start_step})
        except (ValueError, KeyError) as e:
            log.warning(
                "optimizer_state_restore_failed",
                extra={
                    "error": str(e), "step": start_step,
                    "note": "Resuming with fresh Adam moments instead — likely an older "
                             "checkpoint saved before the decay/no-decay optimizer param-group "
                             "split, or a genuinely incompatible checkpoint.",
                },
            )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        import math

        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    if start_step > 0:
        # BUG FOUND THIS REVIEW PASS: LambdaLR tracks its own internal
        # `last_epoch` counter starting at 0, independent of the
        # training loop's `step` variable — it has no way to know
        # training is resuming from start_step rather than from
        # scratch. Left as-is, EVERY resume silently restarted the
        # cosine LR schedule from the beginning (back through warmup,
        # or back to peak LR, depending on warmup_steps) instead of
        # continuing from wherever the true step count had reached —
        # a real LR discontinuity on every resume, exactly the kind of
        # bug that degrades final convergence without ever showing up
        # as a crash. Fast-forward it to match.
        scheduler.last_epoch = start_step - 1
        scheduler.step()

    model.train()
    step = start_step
    t0 = time.time()
    data_iter = iter(loader)

    def _next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    while step < cfg.total_steps:
        optimizer.zero_grad(set_to_none=True)
        # UPGRADE (generalization, this review pass): gradient
        # accumulation — see gradient_accumulation_steps' docstring
        # above. Loss is divided by the accumulation count so the
        # accumulated gradient matches what a single large batch of
        # size effective_batch would have produced (mean reduction
        # composes linearly across accumulated micro-batches this way);
        # clipping and the optimizer step happen exactly once per
        # OPTIMIZER step, after every micro-batch's gradient has been
        # accumulated — not once per micro-batch, which would clip and
        # step on partial gradients and silently defeat the point of
        # accumulating in the first place.
        loss_sum, token_loss_sum, critic_loss_sum = 0.0, 0.0, 0.0
        for micro_step in range(cfg.gradient_accumulation_steps):
            tokens, mask, targets, prompt_embeds = _next_batch()
            tokens, mask, targets, prompt_embeds = (
                tokens.to(device), mask.to(device), targets.to(device), prompt_embeds.to(device)
            )

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                logits, critic_scores = model(tokens, mask, prompt_embeds)
                loss, token_loss, critic_loss = compute_loss(
                    logits, critic_scores, targets, mask, critic_loss_weight=cfg.critic_loss_weight
                )
            (loss / cfg.gradient_accumulation_steps).backward()
            # Logging uses the true mean across accumulated micro-batches,
            # not just whichever micro-batch happened to run last.
            loss_sum += loss.item()
            token_loss_sum += token_loss.item()
            critic_loss_sum += critic_loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            n_micro = cfg.gradient_accumulation_steps
            log.info(
                "train_step",
                extra={
                    "step": step, "loss": round(loss_sum / n_micro, 4),
                    "token_loss": round(token_loss_sum / n_micro, 4),
                    "critic_loss": round(critic_loss_sum / n_micro, 4),
                    "lr": round(scheduler.get_last_lr()[0], 6),
                    "steps_per_sec": round(cfg.log_every / max(elapsed, 1e-6), 3) if step > start_step else None,
                },
            )
            t0 = time.time()

        if step > 0 and step % cfg.checkpoint_every == 0:
            ckpt_path = Path(cfg.output_dir) / f"checkpoint_step{step}.pt"
            save_checkpoint(ckpt_path, model, model_cfg, step, optimizer)
            log.info("checkpoint_saved", extra={"path": str(ckpt_path)})

        if val_loader is not None and step > 0 and step % cfg.val_every == 0:
            val_metrics = _run_validation(model, val_loader, device, cfg)
            log.info("val_step", extra={"step": step, **val_metrics})
            model.train()

        step += 1

    final_path = Path(cfg.output_dir) / "checkpoint_final.pt"
    save_checkpoint(final_path, model, model_cfg, step, optimizer)
    log.info("training_complete", extra={"path": str(final_path), "total_steps": step})


def _run_validation(model, val_loader, device: str, cfg: "TrainConfig") -> dict:
    """Heldout masked-token loss (finding #3) — the metric no prior
    version of this loop ever computed. Deliberately model.eval()'d and
    no_grad'd: this is purely a monitoring/early-stopping signal, never
    part of the optimization step. Capped at cfg.val_batches so a large
    heldout split doesn't stall training every val_every steps; the
    subset is still a real random sample since val_loader's underlying
    dataset order is whatever the manifest wrote it in and batches are
    consumed in order — good enough for a trend signal, not a claim of
    a perfectly i.i.d. estimate on every call.
    """
    import torch

    from krisna_training.sketch.losses import compute_loss

    model.eval()
    total_loss, total_token_loss, total_critic_loss, n_batches = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for i, (tokens, mask, targets, prompt_embeds) in enumerate(val_loader):
            if i >= cfg.val_batches:
                break
            tokens, mask, targets, prompt_embeds = (
                tokens.to(device), mask.to(device), targets.to(device), prompt_embeds.to(device)
            )
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                logits, critic_scores = model(tokens, mask, prompt_embeds)
                loss, token_loss, critic_loss = compute_loss(
                    logits, critic_scores, targets, mask, critic_loss_weight=cfg.critic_loss_weight
                )
            total_loss += loss.item()
            total_token_loss += token_loss.item()
            total_critic_loss += critic_loss.item()
            n_batches += 1

    n_batches = max(1, n_batches)
    return {
        "val_loss": round(total_loss / n_batches, 4),
        "val_token_loss": round(total_token_loss / n_batches, 4),
        "val_critic_loss": round(total_critic_loss / n_batches, 4),
        "val_batches": n_batches,
    }


def _init_from_smaller_grid(model, new_cfg, checkpoint_path: str) -> None:
    """Progressive-resolution init (§6.1: "256->512px progressive
    training"): load a checkpoint trained at a smaller grid, transfer all
    weights as-is EXCEPT the positional embedding, which is bicubically
    interpolated up to the new grid size via pos_embed.py."""
    from krisna_training.sketch.checkpoint_io import load_checkpoint_for_resume
    from krisna_training.sketch.pos_embed import interpolate_pos_embed

    old_model, old_cfg, _, _ = load_checkpoint_for_resume(checkpoint_path)
    old_state = old_model.state_dict()

    new_pos = interpolate_pos_embed(
        old_state["pos_embed"], old_cfg.grid_h, old_cfg.grid_w, new_cfg.grid_h, new_cfg.grid_w
    )
    old_state["pos_embed"] = new_pos

    missing, unexpected = model.load_state_dict(old_state, strict=False)
    log.info(
        "progressive_init",
        extra={"from": checkpoint_path, "missing_keys": missing, "unexpected_keys": unexpected},
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
