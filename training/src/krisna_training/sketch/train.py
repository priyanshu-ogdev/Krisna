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

    start_step = 0
    if cfg.resume_from:
        model, model_cfg, start_step, _ = load_checkpoint_for_resume(cfg.resume_from)
        model.to(device)
        log.info("resumed", extra={"path": cfg.resume_from, "step": start_step})
    else:
        model = build_model(model_cfg)
        if cfg.init_from:
            _init_from_smaller_grid(model, model_cfg, cfg.init_from)
        model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        import math

        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    model.train()
    step = start_step
    t0 = time.time()
    data_iter = iter(loader)

    while step < cfg.total_steps:
        try:
            tokens, mask, targets, prompt_embeds = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            tokens, mask, targets, prompt_embeds = next(data_iter)

        tokens, mask, targets, prompt_embeds = (
            tokens.to(device), mask.to(device), targets.to(device), prompt_embeds.to(device)
        )

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            logits, critic_scores = model(tokens, mask, prompt_embeds)
            loss, token_loss, critic_loss = compute_loss(
                logits, critic_scores, targets, mask, critic_loss_weight=cfg.critic_loss_weight
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            log.info(
                "train_step",
                extra={
                    "step": step, "loss": round(loss.item(), 4),
                    "token_loss": round(token_loss.item(), 4),
                    "critic_loss": round(critic_loss.item(), 4),
                    "lr": round(scheduler.get_last_lr()[0], 6),
                    "steps_per_sec": round(cfg.log_every / max(elapsed, 1e-6), 3) if step > start_step else None,
                },
            )
            t0 = time.time()

        if step > 0 and step % cfg.checkpoint_every == 0:
            ckpt_path = Path(cfg.output_dir) / f"checkpoint_step{step}.pt"
            save_checkpoint(ckpt_path, model, model_cfg, step, optimizer)
            log.info("checkpoint_saved", extra={"path": str(ckpt_path)})

        step += 1

    final_path = Path(cfg.output_dir) / "checkpoint_final.pt"
    save_checkpoint(final_path, model, model_cfg, step, optimizer)
    log.info("training_complete", extra={"path": str(final_path), "total_steps": step})


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
