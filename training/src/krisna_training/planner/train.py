"""Planner BF16 LoRA training loop.

BF16 throughout, no quantization anywhere in this file — §6.1's own
finding is that QLoRA degrades Qwen3.5's hybrid Gated DeltaNet + Gated
Attention architecture, so this trains a standard peft LoRA adapter on top
of the full-precision (bf16) base model. Loss is the base model's own
built-in causal-LM loss (labels with -100 at non-assistant positions),
not reimplemented here — HF's standard CausalLM forward already handles
the shift-by-one and ignore_index correctly.

Run via scripts/train_planner_lora.sh, or:
    python -m krisna_training.planner.train --config configs/planner_lora_train.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("krisna_training.planner.train")


@dataclass
class PlannerTrainConfig:
    manifest_path: str
    output_dir: str
    base_model_id: str = "Qwen/Qwen3.5-9B"
    max_length: int = 4096

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_patterns: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    total_steps: int = 5000
    grad_clip: float = 1.0

    log_every: int = 20
    checkpoint_every: int = 500
    num_workers: int = 2
    seed: int = 42
    resume_adapter_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlannerTrainConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text())
        if "lora_target_patterns" in data:
            data["lora_target_patterns"] = tuple(data["lora_target_patterns"])
        return cls(**data)


def make_collate_fn(pad_token_id: int):
    def collate(batch):
        import torch

        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            ids, mask = b["input_ids"], b["loss_mask"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append([tok if m else -100 for tok, m in zip(ids, mask)] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def build_lora_model(cfg: PlannerTrainConfig):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from krisna_training.planner.lora_config import discover_target_modules

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, dtype=torch.bfloat16)
    base_model.config.use_cache = False   # required alongside gradient
                                            # checkpointing for causal-LM
                                            # training — KV-cache is an
                                            # inference-only optimization,
                                            # incompatible with activation
                                            # recomputation during training.

    # UPGRADE (A6000 48GB training-memory audit): this 9B model loads
    # fully unquantized in bf16 (~18GB weights alone) with no gradient
    # checkpointing at all. At batch_size=4/max_length=4096
    # (planner_lora_train.yaml), the standard dense-transformer activation
    # memory approximation (Korthikanti et al. 2022) gives an upper bound
    # around 245GB without checkpointing — nowhere close to fitting a
    # 48GB card even accounting for this being a full-attention
    # approximation of a cheaper hybrid architecture. With checkpointing,
    # ~7.7GB.
    #
    # Order matters and is a documented pitfall: enable_input_require_grads()
    # must be called on the BASE model before get_peft_model() wraps it.
    # Without this, frozen-base + LoRA + gradient-checkpointing silently
    # breaks backward — the LoRA adapter receives zero gradient updates
    # the whole run (confirmed via huggingface/peft#522,
    # huggingface/transformers#26334, huggingface/transformers#42489).
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    target_modules = discover_target_modules(base_model, include_patterns=cfg.lora_target_patterns)
    log.info("lora_target_modules", extra={"modules": target_modules})

    lora_config = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=target_modules, task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def train(cfg: PlannerTrainConfig) -> None:
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    from krisna_training.planner.dataset import PlannerSFTDataset

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("no_cuda_available", extra={"note": "training a 9B model on CPU is not realistic — this path exists for tests/smoke-checks only"})

    model, tokenizer = build_lora_model(cfg)
    model.to(device)

    if cfg.resume_adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model.get_base_model(), cfg.resume_adapter_path, is_trainable=True)
        model.to(device)
        log.info("resumed_adapter", extra={"path": cfg.resume_adapter_path})

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dataset = PlannerSFTDataset(cfg.manifest_path, tokenizer, max_length=cfg.max_length)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=make_collate_fn(pad_token_id), drop_last=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_lambda(step: int) -> float:
        import math

        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    model.train()
    step = 0
    micro_step = 0
    t0 = time.time()
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)

    while step < cfg.total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
            out = model(**batch)
            loss = out.loss / cfg.grad_accum_steps

        loss.backward()
        micro_step += 1

        if micro_step % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                elapsed = time.time() - t0
                log.info(
                    "train_step",
                    extra={
                        "step": step, "loss": round(out.loss.item(), 4),
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
    cfg = PlannerTrainConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
