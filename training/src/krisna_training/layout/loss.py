"""Continuous Flow Matching (CFM) Loss and Geometry Regularizers for LayoutFlow.

Grounding:
- Guerreiro, J. A., et al. (ECCV 2024). "LayoutFlow: Flow Matching for Layout Generation."
- Lipman, Y., et al. (ICLR 2023). "Flow Matching for Generative Modeling."
- Albergo, M. S., & Vanden-Eijnden, E. (ICLR 2023). "Building Normalizing Flows with Stochastic Interpolants."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def continuous_flow_matching_loss(
    model: nn.Module,
    boxes_1: torch.Tensor,
    type_one_hot: torch.Tensor,
    condition_mask: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    lambda_cls: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Computes Optimal Transport Conditional Flow Matching (OT-CFM) loss.

    Args:
        model: LayoutFlow model instance.
        boxes_1: Ground-truth target bounding boxes of shape (B, N, 4) in [0, 1].
        type_one_hot: Ground-truth element categories one-hot of shape (B, N, K).
        condition_mask: Tensor of shape (B, N) where 1 indicates conditioned/locked nodes, 0 mutable.
        padding_mask: Optional boolean Tensor of shape (B, N) where True indicates padded dummy nodes.
        lambda_cls: Weight coefficient for element category classification loss.

    Returns:
        Dict with 'loss', 'loss_velocity', 'loss_cls', and 'ot_coherence'.
    """
    device = boxes_1.device
    B, N, _ = boxes_1.shape

    # 1. Sample source noise b_0 ~ N(0, I) and continuous timestep t ~ U(0, 1)
    boxes_0 = torch.randn_like(boxes_1)
    t = torch.rand(B, device=device)  # (B,)

    # 2. Linear probability path interpolation: b_t = (1 - t) * b_0 + t * b_1
    t_expanded = t.view(B, 1, 1)
    boxes_t = (1.0 - t_expanded) * boxes_0 + t_expanded * boxes_1

    # Overwrite conditioned/locked elements with their ground-truth coordinates across all t
    cond_expanded = condition_mask.unsqueeze(-1).bool()
    boxes_t = torch.where(cond_expanded, boxes_1, boxes_t)

    # Target velocity vector field: u_t = b_1 - b_0
    target_velocity = boxes_1 - boxes_0

    # 3. Model forward pass
    v_pred, type_logits = model(
        boxes=boxes_t,
        type_one_hot=type_one_hot,
        condition_mask=condition_mask,
        timestep=t,
        padding_mask=padding_mask,
    )

    # 4. Velocity Mean-Squared Error loss (computed only on mutable, non-padded elements)
    active_mask = (1.0 - condition_mask)
    if padding_mask is not None:
        active_mask = active_mask * (~padding_mask).float()

    active_count = active_mask.sum().clamp(min=1.0)
    sq_err = (v_pred - target_velocity) ** 2  # (B, N, 4)
    loss_velocity = (sq_err.sum(dim=-1) * active_mask).sum() / active_count

    # 5. Node category classification loss
    target_labels = type_one_hot.argmax(dim=-1)  # (B, N)
    logits_flat = type_logits.view(-1, type_logits.shape[-1])
    labels_flat = target_labels.view(-1)
    if padding_mask is not None:
        valid_flat = (~padding_mask).view(-1)
        loss_cls = F.cross_entropy(logits_flat[valid_flat], labels_flat[valid_flat])
    else:
        loss_cls = F.cross_entropy(logits_flat, labels_flat)

    # 6. Optimal Transport velocity alignment metric (cosine similarity)
    cos_sim = F.cosine_similarity(v_pred, target_velocity, dim=-1)  # (B, N)
    ot_coherence = (cos_sim * active_mask).sum() / active_count

    total_loss = loss_velocity + lambda_cls * loss_cls

    return {
        "loss": total_loss,
        "loss_velocity": loss_velocity,
        "loss_cls": loss_cls,
        "ot_coherence": ot_coherence,
    }
