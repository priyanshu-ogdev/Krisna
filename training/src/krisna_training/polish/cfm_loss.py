"""
Consistency Flow Matching (CFM) Distillation Loss.

Theoretical Grounding:
- Luo et al., "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference", Springer ECCV 2024.
- Song et al., "Consistency Models", ICML 2023.
- Song & Dhariwal, "Improved Techniques for Training Consistency Models", ICLR 2024.
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023.
- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.

Formulation:
A teacher flow-matching model predicts velocity field v_phi(x_t, t, c) along ODE dx/dt = v_phi(x_t, t).
The student consistency network f_theta(x_t, t, c) directly estimates the clean trajectory origin x_0:
    f_theta(x_t, t, c) = c_skip(t) * x_t + c_out(t) * F_theta(x_t, t, c)
Boundary condition: c_skip(0) = 1.0, c_out(0) = 0.0 ensuring f_theta(x_0, 0, c) == x_0.
Teacher rollout from t_{n+1} to t_n:
    x_hat_{t_n} = x_{t_{n+1}} + (t_n - t_{n+1}) * v_phi(x_{t_{n+1}}, t_{n+1}, c)
Self-Consistency Distillation Loss:
    L_CD(theta) = PseudoHuber(f_theta(x_{t_{n+1}}, t_{n+1}, c), f_{theta^-}(x_hat_{t_n}, t_n, c))
where theta^- denotes the Exponential Moving Average (EMA) student parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class CFMLossOutput:
    loss: torch.Tensor
    consistency_term: torch.Tensor
    pseudo_huber_term: torch.Tensor
    student_pred: torch.Tensor
    target_pred: torch.Tensor


def pseudo_huber_loss(
    pred: torch.Tensor, target: torch.Tensor, c: float = 0.001, reduction: str = "mean"
) -> torch.Tensor:
    """Computes the Pseudo-Huber loss d(x, y) = sqrt(||x - y||^2 + c^2) - c.
    Song & Dhariwal (ICLR 2024) prove this metric eliminates gradient spikes at high noise
    while preserving sub-pixel visual sharpness for few-step generative distillation.
    """
    diff_sq = (pred - target) ** 2
    # Spatial sum / mean per sample
    if diff_sq.ndim > 1:
        dim_list = list(range(1, diff_sq.ndim))
        per_sample_sq = diff_sq.mean(dim=dim_list)
    else:
        per_sample_sq = diff_sq

    huber = torch.sqrt(per_sample_sq + (c ** 2)) - c
    if reduction == "mean":
        return huber.mean()
    elif reduction == "sum":
        return huber.sum()
    return huber


def consistency_scaling(
    timesteps: torch.Tensor, sigma_data: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes analytical boundary condition coefficients c_skip(t) and c_out(t)
    guaranteeing that f_theta(x_0, 0, c) = x_0 at t=0.

    c_skip(t) = sigma_data^2 / ((t - 0)^2 + sigma_data^2)
    c_out(t) = (t - 0) * sigma_data / sqrt(t^2 + sigma_data^2)
    """
    t = timesteps.clamp(min=0.0, max=1.0)
    sigma_data_sq = sigma_data ** 2
    c_skip = sigma_data_sq / (t ** 2 + sigma_data_sq)
    c_out = (t * sigma_data) / torch.sqrt(t ** 2 + sigma_data_sq)
    return c_skip, c_out


def consistency_flow_step(
    x_next: torch.Tensor,
    t_next: float,
    t_curr: float,
    teacher_v: torch.Tensor,
    solver: str = "euler",
) -> torch.Tensor:
    """Takes a numerical ODE step from t_{n+1} (t_next) backwards to t_n (t_curr)
    using the teacher flow vector field v_phi:
    x_{t_n} = x_{t_{n+1}} + (t_curr - t_next) * v_phi(x_{t_{n+1}}, t_{n+1})
    """
    dt = t_curr - t_next
    return x_next + dt * teacher_v


def consistency_distillation_loss(
    student_net: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target_net: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    teacher_v_fn: Callable[[torch.Tensor, float], torch.Tensor],
    clean_latent: torch.Tensor,
    noise: torch.Tensor,
    t_next: float,
    t_curr: float,
    huber_c: float = 0.001,
) -> CFMLossOutput:
    """Calculates the Consistency Flow Matching distillation loss for one batch step.

    Args:
        student_net: Trainable student model F_theta(x_t, t).
        target_net: Frozen / EMA target model F_theta^-(x_t, t).
        teacher_v_fn: Teacher ODE vector field evaluator v_phi(x_t, t).
        clean_latent: x_0 UI latent tensor.
        noise: Standard Gaussian noise tensor.
        t_next: Upper discrete timestep t_{n+1} in (0, 1].
        t_curr: Lower discrete timestep t_n in [0, 1), t_curr < t_next.
        huber_c: Pseudo-Huber loss curvature parameter (default: 0.001).
    """
    device = clean_latent.device
    dtype = clean_latent.dtype
    bsz = clean_latent.shape[0]

    # Optimal transport linear flow interpolation at t_{n+1}:
    # x_t = (1 - t) * clean + t * noise
    x_next = (1.0 - t_next) * clean_latent + t_next * noise

    # Teacher takes one ODE step from t_{n+1} to t_n
    with torch.no_grad():
        v_teacher = teacher_v_fn(x_next, t_next)
        x_curr_teacher = consistency_flow_step(x_next, t_next, t_curr, v_teacher)

    # Student evaluates at (x_next, t_next)
    t_next_tensor = torch.full((bsz, 1, 1, 1), t_next, device=device, dtype=dtype)
    c_skip_next, c_out_next = consistency_scaling(t_next_tensor)
    f_student = c_skip_next * x_next + c_out_next * student_net(x_next, t_next_tensor)

    # Target EMA network evaluates at (x_curr_teacher, t_curr)
    with torch.no_grad():
        t_curr_tensor = torch.full((bsz, 1, 1, 1), t_curr, device=device, dtype=dtype)
        c_skip_curr, c_out_curr = consistency_scaling(t_curr_tensor)
        f_target = c_skip_curr * x_curr_teacher + c_out_curr * target_net(x_curr_teacher, t_curr_tensor)

    # Pseudo-Huber consistency loss
    loss = pseudo_huber_loss(f_student, f_target.detach(), c=huber_c)

    return CFMLossOutput(
        loss=loss,
        consistency_term=loss.detach(),
        pseudo_huber_term=loss.detach(),
        student_pred=f_student.detach(),
        target_pred=f_target.detach(),
    )
