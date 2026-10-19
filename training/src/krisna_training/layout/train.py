"""LayoutFlow Trainer and Validation Routines.

Grounding:
- Guerreiro, J. A., et al. (ECCV 2024). "LayoutFlow: Flow Matching for Layout Generation."
- Lipman, Y., et al. (ICLR 2023). "Flow Matching for Generative Modeling."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from krisna_training.layout.loss import continuous_flow_matching_loss
from krisna_training.layout.model import LayoutFlow, LayoutFlowConfig


class LayoutTensorDataset(Dataset):
    """In-memory or list-based tensor dataset for LayoutFlow training."""

    def __init__(self, records: list[dict[str, Any]], max_nodes: int = 64, type_count: int = 10):
        self.records = records
        self.max_nodes = max_nodes
        self.type_count = type_count
        self.node_types = (
            "container", "text", "image", "button", "input",
            "list", "card", "icon", "divider", "other",
        )
        self.type_to_id = {name: i for i, name in enumerate(self.node_types)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        nodes = rec.get("nodes", [])[: self.max_nodes]
        n = len(nodes)

        boxes = torch.zeros(self.max_nodes, 4, dtype=torch.float32)
        type_oh = torch.zeros(self.max_nodes, self.type_count, dtype=torch.float32)
        cond_mask = torch.zeros(self.max_nodes, dtype=torch.float32)
        pad_mask = torch.ones(self.max_nodes, dtype=torch.bool)

        for i, node in enumerate(nodes):
            raw_b = node.get("bbox") or {}
            if isinstance(raw_b, dict):
                x1 = float(raw_b.get("x1", 0.0))
                y1 = float(raw_b.get("y1", 0.0))
                x2 = float(raw_b.get("x2", 1.0))
                y2 = float(raw_b.get("y2", 1.0))
            elif isinstance(raw_b, (list, tuple)) and len(raw_b) == 4:
                x1, y1, x2, y2 = [float(v) for v in raw_b]
            else:
                x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0

            boxes[i] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            t_idx = self.type_to_id.get(str(node.get("type")), self.type_count - 1)
            type_oh[i, t_idx] = 1.0
            cond_mask[i] = 1.0 if node.get("locked") else 0.0
            pad_mask[i] = False

        return {
            "boxes": boxes,
            "type_one_hot": type_oh,
            "condition_mask": cond_mask,
            "padding_mask": pad_mask,
        }


def train_layout_flow_epoch(
    model: LayoutFlow,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    lambda_cls: float = 0.5,
) -> dict[str, float]:
    """Runs a single training epoch of OT-CFM."""
    model.train()
    total_loss = 0.0
    total_v_loss = 0.0
    total_cls_loss = 0.0
    total_ot_coh = 0.0
    count = 0

    for batch in dataloader:
        boxes = batch["boxes"].to(device)
        type_oh = batch["type_one_hot"].to(device)
        cond_mask = batch["condition_mask"].to(device)
        pad_mask = batch["padding_mask"].to(device)

        optimizer.zero_grad()
        loss_dict = continuous_flow_matching_loss(
            model=model,
            boxes_1=boxes,
            type_one_hot=type_oh,
            condition_mask=cond_mask,
            padding_mask=pad_mask,
            lambda_cls=lambda_cls,
        )

        loss = loss_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        b_size = boxes.size(0)
        total_loss += loss.item() * b_size
        total_v_loss += loss_dict["loss_velocity"].item() * b_size
        total_cls_loss += loss_dict["loss_cls"].item() * b_size
        total_ot_coh += loss_dict["ot_coherence"].item() * b_size
        count += b_size

    n = max(1, count)
    return {
        "loss": round(total_loss / n, 5),
        "loss_velocity": round(total_v_loss / n, 5),
        "loss_cls": round(total_cls_loss / n, 5),
        "ot_coherence": round(total_ot_coh / n, 4),
    }


def evaluate_layout_flow(
    model: LayoutFlow,
    dataloader: DataLoader,
    device: torch.device | str,
    lambda_cls: float = 0.5,
) -> dict[str, float]:
    """Evaluates OT-CFM loss on a validation split."""
    model.eval()
    total_loss = 0.0
    total_v_loss = 0.0
    total_cls_loss = 0.0
    total_ot_coh = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            boxes = batch["boxes"].to(device)
            type_oh = batch["type_one_hot"].to(device)
            cond_mask = batch["condition_mask"].to(device)
            pad_mask = batch["padding_mask"].to(device)

            loss_dict = continuous_flow_matching_loss(
                model=model,
                boxes_1=boxes,
                type_one_hot=type_oh,
                condition_mask=cond_mask,
                padding_mask=pad_mask,
                lambda_cls=lambda_cls,
            )

            b_size = boxes.size(0)
            total_loss += loss_dict["loss"].item() * b_size
            total_v_loss += loss_dict["loss_velocity"].item() * b_size
            total_cls_loss += loss_dict["loss_cls"].item() * b_size
            total_ot_coh += loss_dict["ot_coherence"].item() * b_size
            count += b_size

    n = max(1, count)
    return {
        "val_loss": round(total_loss / n, 5),
        "val_loss_velocity": round(total_v_loss / n, 5),
        "val_loss_cls": round(total_cls_loss / n, 5),
        "val_ot_coherence": round(total_ot_coh / n, 4),
    }
