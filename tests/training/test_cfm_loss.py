"""Tests for Consistency Flow Matching (CFM) Distillation Loss.
Theoretical Grounding:
- Luo et al., Springer ECCV 2024
- Song et al., ICML 2023
- Song & Dhariwal, ICLR 2024
"""

from __future__ import annotations

import math
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from krivex_training.polish.cfm_loss import (
    CFMLossOutput,
    consistency_distillation_loss,
    consistency_flow_step,
    consistency_scaling,
    pseudo_huber_loss,
)


def _rand_tensor(batch=2, channels=4, h=8, w=8, seed=42):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, channels, h, w, generator=g)


class TestPseudoHuberMetric:
    def test_pseudo_huber_equals_zero_when_identical(self):
        t = _rand_tensor(seed=1)
        loss = pseudo_huber_loss(t, t, c=0.001)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_pseudo_huber_approximates_l2_for_small_delta(self):
        t1 = torch.zeros(2, 4, 8, 8)
        delta = 1e-4
        t2 = torch.full_like(t1, delta)
        c = 0.001
        loss = pseudo_huber_loss(t1, t2, c=c)
        # Expected expansion: sqrt(d^2 + c^2) - c ~= (d^2) / (2c)
        d_sq = delta ** 2
        expected = math.sqrt(d_sq + c ** 2) - c
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_pseudo_huber_sublinear_at_large_delta(self):
        t1 = torch.zeros(2, 4, 8, 8)
        t2 = torch.full_like(t1, 10.0)
        loss_huber = pseudo_huber_loss(t1, t2, c=0.001)
        loss_l2 = torch.nn.functional.mse_loss(t1, t2)
        # Huber should be ~10, whereas L2 is 100 — sublinear growth prevents gradient explosion
        assert loss_huber.item() < loss_l2.item()


class TestBoundaryConditionAndRollout:
    def test_boundary_condition_at_t_zero(self):
        t_zero = torch.zeros(2, 1, 1, 1)
        c_skip, c_out = consistency_scaling(t_zero, sigma_data=0.5)
        assert torch.allclose(c_skip, torch.ones_like(c_skip))
        assert torch.allclose(c_out, torch.zeros_like(c_out))

    def test_consistency_flow_step_inverts_velocity_direction(self):
        x = _rand_tensor(seed=2)
        v = _rand_tensor(seed=3)
        t_next = 0.8
        t_curr = 0.6
        x_curr = consistency_flow_step(x, t_next, t_curr, v)
        dt = t_curr - t_next  # -0.2
        assert torch.allclose(x_curr, x + dt * v)


class TestDistillationLossOptimization:
    def test_gradient_step_reduces_consistency_error(self):
        clean = _rand_tensor(seed=10)
        noise = _rand_tensor(seed=11)
        t_next = 0.75
        t_curr = 0.50

        # Simple 1x1 conv student and target
        student = nn.Conv2d(4, 4, kernel_size=1)
        target = nn.Conv2d(4, 4, kernel_size=1)
        target.load_state_dict(student.state_dict())
        for p in target.parameters():
            p.requires_grad = False

        teacher_v = _rand_tensor(seed=12)

        def teacher_v_fn(x, t):
            return teacher_v

        # Forward loss
        out = consistency_distillation_loss(
            student_net=lambda x, t: student(x),
            target_net=lambda x, t: target(x),
            teacher_v_fn=teacher_v_fn,
            clean_latent=clean,
            noise=noise,
            t_next=t_next,
            t_curr=t_curr,
            huber_c=0.001,
        )
        assert out.loss.item() > 0.0

        # Take gradient step
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        # Re-evaluate
        out_after = consistency_distillation_loss(
            student_net=lambda x, t: student(x),
            target_net=lambda x, t: target(x),
            teacher_v_fn=teacher_v_fn,
            clean_latent=clean,
            noise=noise,
            t_next=t_next,
            t_curr=t_curr,
            huber_c=0.001,
        )
        assert out_after.loss.item() < out.loss.item(), "One optimization step should reduce consistency loss"
