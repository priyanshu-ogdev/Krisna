"""
Direct Preference Optimization (DPO) Loss for Continuous Flow Matching (MMDiT).

Theoretical Grounding:
- Wallace et al., "Diffusion Model Alignment Using Direct Preference Optimization", IEEE/CVF CVPR 2024.
- Kim et al., "Preference Alignment with Flow Matching", NeurIPS 2024.
- Croitoru et al., "Curriculum Direct Preference Optimization for Diffusion and Consistency Models", IEEE/CVF CVPR 2025.
- Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024.

Formulation:
Implicit reward r_theta(x) = -(beta / 2) * [ ||v_theta(x) - u(x)||^2 - ||v_ref(x) - u(x)||^2 ]
Implicit reward margin Delta r = r_theta(x_w) - r_theta(x_l)
Loss_DPO = -log sigma(Delta r) = softplus(-Delta r)
Loss_Total = Loss_DPO + fm_anchor_weight * Loss_anchor(x_w)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F


@dataclass
class FlowMatchingDPOLossOutput:
    loss: torch.Tensor
    dpo_term: torch.Tensor
    fm_anchor_term: torch.Tensor
    implicit_reward_margin: torch.Tensor
    chosen_policy_err: torch.Tensor | None = None
    rejected_policy_err: torch.Tensor | None = None
    chosen_ref_err: torch.Tensor | None = None
    rejected_ref_err: torch.Tensor | None = None


# Alias for backward compatibility
DPOOutput = FlowMatchingDPOLossOutput


def flow_matching_velocity_target(clean_latent: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Computes target velocity field u(x_t) = noise - clean_latent for continuous flow matching."""
    return noise - clean_latent


def noise_latent_at_timestep(
    clean_latent: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Optimal transport linear path interpolation x_t = (1 - sigma) * clean_latent + sigma * noise."""
    return (1.0 - sigma) * clean_latent + sigma * noise


def apply_flow_matching_shift(timesteps: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """Resolution-dependent logit-normal time shifting (Esser et al., ICML 2024):
    tilde{t} = (s * t) / (1 + (s - 1) * t)
    """
    if shift == 1.0:
        return timesteps
    return (shift * timesteps) / (1.0 + (shift - 1.0) * timesteps)


def ema_warmup_decay(step: int, target_decay: float = 0.9995, warmup_steps: int = 2000) -> float:
    """Computes dynamic exponential moving average (EMA) decay rate during initial training steps."""
    if step < warmup_steps:
        return min(target_decay, (1.0 + step) / (10.0 + step))
    return target_decay


def flow_matching_dpo_loss(
    policy_v_chosen: torch.Tensor,
    policy_v_rejected: torch.Tensor,
    ref_v_chosen: torch.Tensor,
    ref_v_rejected: torch.Tensor,
    target_v_chosen: torch.Tensor,
    target_v_rejected: torch.Tensor,
    beta: float = 2000.0,
    fm_anchor_weight: float = 0.0,
    curriculum_margin: float | None = None,
    pair_margins: torch.Tensor | float | None = None,
) -> FlowMatchingDPOLossOutput:
    """Calculates flow-matching direct preference optimization loss with optional anchor regularization
    and curriculum margin scheduling (Croitoru et al., CVPR 2025).

    Args:
        policy_v_chosen: Policy MMDiT velocity prediction on chosen UI latent.
        policy_v_rejected: Policy MMDiT velocity prediction on rejected UI latent.
        ref_v_chosen: Frozen reference MMDiT velocity prediction on chosen UI latent.
        ref_v_rejected: Frozen reference MMDiT velocity prediction on rejected UI latent.
        target_v_chosen: Optimal transport ground-truth velocity for chosen UI latent.
        target_v_rejected: Optimal transport ground-truth velocity for rejected UI latent.
        beta: Inverse-temperature preference scale parameter (default: 2000.0).
        fm_anchor_weight: Anchor MSE regularization on chosen preferred path (Wallace et al., CVPR 2024; Kim et al., NeurIPS 2024).
        curriculum_margin: Optional CVPR 2025 curriculum margin difficulty threshold.
        pair_margins: Optional scalar or 1D tensor score differences between chosen and rejected UI candidates.
    """
    # Per-sample MSE loss across spatial and channel dimensions
    chosen_policy_err = F.mse_loss(policy_v_chosen, target_v_chosen, reduction="none").mean(
        dim=list(range(1, policy_v_chosen.ndim))
    )
    chosen_ref_err = F.mse_loss(ref_v_chosen, target_v_chosen, reduction="none").mean(
        dim=list(range(1, ref_v_chosen.ndim))
    )
    rejected_policy_err = F.mse_loss(policy_v_rejected, target_v_rejected, reduction="none").mean(
        dim=list(range(1, policy_v_rejected.ndim))
    )
    rejected_ref_err = F.mse_loss(ref_v_rejected, target_v_rejected, reduction="none").mean(
        dim=list(range(1, ref_v_rejected.ndim))
    )

    # Implicit reward difference per sample
    chosen_reward = -(chosen_policy_err - chosen_ref_err)
    rejected_reward = -(rejected_policy_err - rejected_ref_err)

    reward_diff = chosen_reward - rejected_reward
    margin = beta * reward_diff

    # Numerically stable softplus loss
    per_sample_dpo = F.softplus(-margin)

    # Curriculum DPO modulation (Croitoru et al., CVPR 2025)
    if curriculum_margin is not None and pair_margins is not None:
        if not isinstance(pair_margins, torch.Tensor):
            pair_margins = torch.tensor(pair_margins, device=margin.device, dtype=margin.dtype)
        curriculum_weights = torch.clamp(
            pair_margins / max(1e-4, curriculum_margin), min=0.1, max=1.0
        )
        per_sample_dpo = per_sample_dpo * curriculum_weights

    dpo_term = per_sample_dpo.mean()

    # Flow matching anchor regularization on preferred path
    if fm_anchor_weight > 0.0:
        fm_anchor_term = fm_anchor_weight * chosen_policy_err.mean()
    else:
        fm_anchor_term = torch.zeros((), device=policy_v_chosen.device, dtype=policy_v_chosen.dtype)

    total_loss = dpo_term + fm_anchor_term

    return FlowMatchingDPOLossOutput(
        loss=total_loss,
        dpo_term=dpo_term,
        fm_anchor_term=fm_anchor_term,
        implicit_reward_margin=reward_diff.mean(),
        chosen_policy_err=chosen_policy_err.mean(),
        rejected_policy_err=rejected_policy_err.mean(),
        chosen_ref_err=chosen_ref_err.mean(),
        rejected_ref_err=rejected_ref_err.mean(),
    )
