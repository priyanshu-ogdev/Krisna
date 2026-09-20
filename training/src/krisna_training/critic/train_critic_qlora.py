"""Critic (Gemma 4 31B Dense) QLoRA training via Unsloth.

MUST run under ./venv-critic (see this package's __init__.py). Loads the
same `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` checkpoint
critic_worker.py's `_load()` does, via the same `FastModel.from_pretrained`
call, then injects QLoRA adapters via `FastModel.get_peft_model` (Unsloth's
VLM LoRA API — confirmed real via docs.unsloth.ai/basics/vision-fine-tuning,
which documents `finetune_vision_layers`/`finetune_language_layers`/
`finetune_attention_modules`/`finetune_mlp_modules` as the VLM-specific
controls beyond the plain-LLM `r`/`lora_alpha`/`target_modules` args).

Both vision and language layers are trained by default — the Critic's job
(§5.4: "judges the finished render, not a generator") genuinely needs to
both perceive the image AND reason/write about it well, unlike a
text-only adapter.

Run: ./venv-critic/bin/python -m krisna_training.critic.train_critic_qlora --config configs/critic_qlora_train.yaml
(or scripts/train_critic_qlora.sh)
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("krisna_training.critic.train")


@dataclass
class CriticTrainConfig:
    manifest_path: str
    output_dir: str
    model_id: str = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
    max_seq_length: int = 8192
    max_length: int = 4096          # per-example token cap (dataset.py's tokenize function)

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    batch_size: int = 1             # VLM + 31B — expect this to stay small on one A6000
    grad_accum_steps: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    total_steps: int = 2000
    grad_clip: float = 1.0

    log_every: int = 20
    checkpoint_every: int = 500
    seed: int = 3407                # Unsloth's own convention default
    resume_adapter_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CriticTrainConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)


def build_lora_model(cfg: CriticTrainConfig):
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    model = FastModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        finetune_vision_layers=cfg.finetune_vision_layers,
        finetune_language_layers=cfg.finetune_language_layers,
        finetune_attention_modules=cfg.finetune_attention_modules,
        finetune_mlp_modules=cfg.finetune_mlp_modules,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    return model, tokenizer


def make_collate_fn(pad_token_id: int):
    """Same padding/label-masking shape as training/planner/train.py's
    collate — duplicated rather than imported, per this package's
    self-contained-under-venv-critic design (see __init__.py)."""

    def collate(batch):
        import torch

        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        pixel_values = []

        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_len)
            row_labels = [
                tok if m == 1 else -100 for tok, m in zip(b["input_ids"], b["loss_mask"])
            ] + [-100] * pad_len
            labels.append(row_labels)
            pixel_values.append(b["pixel_values"])

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if pixel_values and pixel_values[0] is not None:
            out["pixel_values"] = torch.cat([torch.as_tensor(p) for p in pixel_values], dim=0)
        return out

    return collate


def train(cfg: CriticTrainConfig) -> None:
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    from krisna_training.critic.dataset import CriticSFTDataset

    torch.manual_seed(cfg.seed)

    model, tokenizer = build_lora_model(cfg)
    if cfg.resume_adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model.get_base_model(), cfg.resume_adapter_path, is_trainable=True
        )
        log.info("resumed_adapter", extra={"path": cfg.resume_adapter_path})

    dataset = CriticSFTDataset(cfg.manifest_path, tokenizer, max_length=cfg.max_length)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=make_collate_fn(pad_token_id), drop_last=True,
    )

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        import math

        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    model.train()
    step, micro_step = 0, 0
    t0 = time.time()
    data_iter = iter(loader)

    while step < cfg.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / cfg.grad_accum_steps
        loss.backward()
        micro_step += 1

        if micro_step % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), cfg.grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                elapsed = time.time() - t0
                log.info(
                    "train_step",
                    extra={
                        "step": step, "loss": round(outputs.loss.item(), 4),
                        "lr": round(scheduler.get_last_lr()[0], 8),
                        "steps_per_sec": round(cfg.log_every / max(elapsed, 1e-6), 3) if step > 0 else None,
                    },
                )
                t0 = time.time()

            if step > 0 and step % cfg.checkpoint_every == 0:
                ckpt_dir = Path(cfg.output_dir) / f"checkpoint_step{step}"
                model.save_pretrained(str(ckpt_dir))
                log.info("checkpoint_saved", extra={"path": str(ckpt_dir)})

            step += 1

    final_dir = Path(cfg.output_dir) / "checkpoint_final"
    model.save_pretrained(str(final_dir))
    log.info("training_complete", extra={"path": str(final_dir), "total_steps": step})


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = CriticTrainConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
