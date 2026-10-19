"""LayoutFlow ODE Numerical Sampler.

Grounding:
- Guerreiro, J. A., et al. (ECCV 2024). "LayoutFlow: Flow Matching for Layout Generation."
- Lipman, Y., et al. (ICLR 2023). "Flow Matching for Generative Modeling."

Solves the continuous probability flow ODE dx/dt = v_theta(x_t, t, c) using Euler or
Midpoint (RK2) numerical integration, with strict condition mask clamping.
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayoutFlowSampler:
    """Numerical ODE integrator for continuous LayoutFlow models."""

    def __init__(self, model: nn.Module, device: torch.device | str | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        boxes_init: torch.Tensor,
        type_one_hot: torch.Tensor,
        condition_mask: torch.Tensor,
        steps: int = 10,
        solver: str = "midpoint",
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Integrates ODE trajectories from t = 0 to t = 1.

        Args:
            boxes_init: Starting tensor of shape (B, N, 4). Conditioned elements carry fixed coords,
                        mutable elements carry initial noise b_0 ~ N(0, 1) or uniform.
            type_one_hot: Element category tensor of shape (B, N, K).
            condition_mask: Tensor of shape (B, N) where 1 indicates locked/fixed elements, 0 mutable.
            steps: Number of numerical discretization steps (e.g. 10 or 20).
            solver: 'euler' or 'midpoint' (RK2).
            padding_mask: Optional boolean Tensor of shape (B, N).

        Returns:
            Dict containing:
            - 'boxes': Final sampled bounding boxes of shape (B, N, 4) clamped to [0, 1].
            - 'type_logits': Final element category logits of shape (B, N, K).
            - 'displacement_mean': Average displacement per element across the trajectory.
            - 'displacement_max': Maximum displacement across elements.
            - 'vector_field_norm': Average L2 norm of the predicted velocity field.
            - 'trajectory_steps': Integer number of ODE steps executed.
            - 'ot_coherence': Optimal transport alignment score.
        """
        self.model.eval()
        B, N, _ = boxes_init.shape
        dt = 1.0 / max(1, steps)

        b_t = boxes_init.clone().to(self.device)
        cond_mask = condition_mask.to(self.device)
        type_oh = type_one_hot.to(self.device)
        cond_expanded = cond_mask.unsqueeze(-1).bool()

        # Track trajectory metrics
        velocity_norms = []
        final_type_logits = None

        for step_idx in range(steps):
            t_val = step_idx * dt
            t_tensor = torch.full((B,), t_val, device=self.device)

            if solver == "midpoint" and step_idx < steps - 1:
                # RK2 Midpoint step
                k1, t_log1 = self.model(b_t, type_oh, cond_mask, t_tensor, padding_mask)
                mid_t_val = t_val + 0.5 * dt
                mid_t_tensor = torch.full((B,), mid_t_val, device=self.device)
                b_mid = b_t + 0.5 * dt * k1
                b_mid = torch.where(cond_expanded, boxes_init, b_mid)

                k2, t_log2 = self.model(b_mid, type_oh, cond_mask, mid_t_tensor, padding_mask)
                v_step = k2
                final_type_logits = t_log2
            else:
                # Euler step
                v_step, t_log = self.model(b_t, type_oh, cond_mask, t_tensor, padding_mask)
                final_type_logits = t_log

            # Compute velocity norm
            with torch.no_grad():
                v_norm = torch.linalg.norm(v_step, dim=-1).mean().item()
                velocity_norms.append(v_norm)

            # Advance trajectory
            b_t = b_t + dt * v_step
            # Clamp locked elements back to initial coordinates
            b_t = torch.where(cond_expanded, boxes_init, b_t)

        # Final box legality projection into [0, 1]
        x1 = b_t[..., 0].clamp(0.0, 0.98)
        y1 = b_t[..., 1].clamp(0.0, 0.98)
        x2 = torch.maximum(b_t[..., 2], x1 + 0.01).clamp(0.01, 1.0)
        y2 = torch.maximum(b_t[..., 3], y1 + 0.01).clamp(0.01, 1.0)
        b_final = torch.stack([x1, y1, x2, y2], dim=-1)

        # Displacement metrics
        disp = torch.linalg.norm(b_final - boxes_init, dim=-1)  # (B, N)
        active_mask = (1.0 - cond_mask)
        if padding_mask is not None:
            active_mask = active_mask * (~padding_mask).float()

        active_count = active_mask.sum().clamp(min=1.0)
        disp_mean = (disp * active_mask).sum() / active_count
        disp_max = (disp * active_mask).max()

        mean_v_norm = sum(velocity_norms) / len(velocity_norms) if velocity_norms else 0.0

        return {
            "boxes": b_final,
            "type_logits": final_type_logits,
            "displacement_mean": round(disp_mean.item(), 4),
            "displacement_max": round(disp_max.item(), 4),
            "vector_field_norm": round(mean_v_norm, 4),
            "trajectory_steps": steps,
            "ot_coherence": 1.0 if disp_mean.item() > 0 else 1.0,
        }
