"""Tests for polish/dpo_loss.py — pure math, no GPU or real model needed.

These aren't just shape/smoke tests. The loss function was adapted from a
published formulation (see dpo_loss.py's module docstring), not verified
against a reference implementation directly — these tests check the
actual mathematical properties the adaptation is supposed to have, the
same way you'd sanity-check a hand-derived formula before trusting it.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from krisna_training.polish.dpo_loss import (
    flow_matching_dpo_loss,
    flow_matching_velocity_target,
    noise_latent_at_timestep,
)


def _rand_latent(batch=2, channels=4, h=8, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, channels, h, w, generator=g)


class TestKnownClosedFormValues:
    def test_loss_equals_log2_when_policy_matches_reference_exactly(self):
        """The standard DPO sanity check: if the policy hasn't diverged
        from the reference at all (identical predictions on both chosen
        and rejected), the implicit reward margin is exactly zero, and
        -log(sigmoid(0)) = log(2) — this is true for ANY correctly-
        implemented DPO-family loss, LLM or diffusion, and is the single
        most important check that the sigmoid/softplus math wasn't
        transcribed wrong.
        """
        target_chosen = _rand_latent(seed=1)
        target_rejected = _rand_latent(seed=2)
        # Policy and reference predict THE SAME (arbitrary, imperfect)
        # velocity on both sides — not necessarily equal to target.
        shared_chosen_pred = _rand_latent(seed=3)
        shared_rejected_pred = _rand_latent(seed=4)

        out = flow_matching_dpo_loss(
            policy_v_chosen=shared_chosen_pred, policy_v_rejected=shared_rejected_pred,
            ref_v_chosen=shared_chosen_pred, ref_v_rejected=shared_rejected_pred,
            target_v_chosen=target_chosen, target_v_rejected=target_rejected,
            beta=2000.0,
        )
        assert out.dpo_term.item() == pytest.approx(math.log(2), abs=1e-5)
        assert out.implicit_reward_margin.item() == pytest.approx(0.0, abs=1e-6)

    def test_dpo_term_approaches_zero_when_policy_strongly_prefers_chosen(self):
        """Policy fits the chosen target perfectly and the rejected
        target terribly, while the reference is mediocre at both —
        margin should be strongly positive, loss should be near zero
        (softplus(-large) -> 0)."""
        target_chosen = _rand_latent(seed=5)
        target_rejected = _rand_latent(seed=6)
        ref_chosen_pred = _rand_latent(seed=7)   # mediocre, unrelated to target
        ref_rejected_pred = _rand_latent(seed=8)

        out = flow_matching_dpo_loss(
            policy_v_chosen=target_chosen,          # PERFECT prediction on chosen
            policy_v_rejected=target_rejected + 50.0,  # TERRIBLE prediction on rejected
            ref_v_chosen=ref_chosen_pred, ref_v_rejected=ref_rejected_pred,
            target_v_chosen=target_chosen, target_v_rejected=target_rejected,
            beta=2000.0,
        )
        assert out.dpo_term.item() < 1e-3
        assert out.implicit_reward_margin.item() > 0

    def test_dpo_term_grows_when_policy_strongly_prefers_rejected(self):
        """The exact opposite of the above — policy has learned the WRONG
        preference. Loss should be large, margin strongly negative."""
        target_chosen = _rand_latent(seed=9)
        target_rejected = _rand_latent(seed=10)
        ref_chosen_pred = _rand_latent(seed=11)
        ref_rejected_pred = _rand_latent(seed=12)

        out = flow_matching_dpo_loss(
            policy_v_chosen=target_chosen + 50.0,   # TERRIBLE on chosen
            policy_v_rejected=target_rejected,       # PERFECT on rejected — backwards!
            ref_v_chosen=ref_chosen_pred, ref_v_rejected=ref_rejected_pred,
            target_v_chosen=target_chosen, target_v_rejected=target_rejected,
            beta=2000.0,
        )
        assert out.dpo_term.item() > 5.0
        assert out.implicit_reward_margin.item() < 0


class TestGradientDirection:
    def test_gradient_step_reduces_chosen_prediction_error(self):
        """The actual behavioral claim being made: optimizing this loss
        should push the policy's chosen-side prediction TOWARD the
        target, relative to the (frozen) reference. Verified by taking
        one real gradient step and checking chosen_policy_err actually
        decreased — not just that the loss scalar went down (which could
        happen for the wrong reason, e.g. only the rejected side moving)."""
        target_chosen = _rand_latent(seed=13)
        target_rejected = _rand_latent(seed=14)
        ref_chosen_pred = _rand_latent(seed=15).detach()
        ref_rejected_pred = _rand_latent(seed=16).detach()

        policy_chosen = (_rand_latent(seed=17)).clone().requires_grad_(True)
        policy_rejected = (_rand_latent(seed=18)).clone().requires_grad_(True)

        def chosen_err(pred):
            return torch.nn.functional.mse_loss(pred, target_chosen).item()

        err_before = chosen_err(policy_chosen)

        out = flow_matching_dpo_loss(
            policy_v_chosen=policy_chosen, policy_v_rejected=policy_rejected,
            ref_v_chosen=ref_chosen_pred, ref_v_rejected=ref_rejected_pred,
            target_v_chosen=target_chosen, target_v_rejected=target_rejected,
            beta=2000.0,
        )
        out.loss.backward()
        with torch.no_grad():
            policy_chosen -= 0.01 * policy_chosen.grad

        err_after = chosen_err(policy_chosen)
        assert err_after < err_before, "one gradient step should move the chosen-side prediction closer to its target"


class TestAnchorRegularization:
    def test_anchor_disabled_by_default_contributes_nothing(self):
        out = flow_matching_dpo_loss(
            policy_v_chosen=_rand_latent(seed=19), policy_v_rejected=_rand_latent(seed=20),
            ref_v_chosen=_rand_latent(seed=21), ref_v_rejected=_rand_latent(seed=22),
            target_v_chosen=_rand_latent(seed=23), target_v_rejected=_rand_latent(seed=24),
        )
        assert out.fm_anchor_term.item() == 0.0
        assert out.loss.item() == pytest.approx(out.dpo_term.item())

    def test_anchor_enabled_adds_exactly_the_chosen_side_mse(self):
        policy_chosen = _rand_latent(seed=25)
        target_chosen = _rand_latent(seed=26)
        expected_anchor = 0.5 * torch.nn.functional.mse_loss(policy_chosen, target_chosen).item()

        out = flow_matching_dpo_loss(
            policy_v_chosen=policy_chosen, policy_v_rejected=_rand_latent(seed=27),
            ref_v_chosen=_rand_latent(seed=28), ref_v_rejected=_rand_latent(seed=29),
            target_v_chosen=target_chosen, target_v_rejected=_rand_latent(seed=30),
            fm_anchor_weight=0.5,
        )
        assert out.fm_anchor_term.item() == pytest.approx(expected_anchor, rel=1e-4)
        assert out.loss.item() == pytest.approx((out.dpo_term + out.fm_anchor_term).item(), rel=1e-4)


class TestFlowMatchingHelpers:
    def test_velocity_target_is_noise_minus_clean(self):
        clean = _rand_latent(seed=31)
        noise = _rand_latent(seed=32)
        target = flow_matching_velocity_target(clean, noise)
        assert torch.allclose(target, noise - clean)

    def test_noising_at_sigma_zero_returns_clean_latent(self):
        clean = _rand_latent(seed=33)
        noise = _rand_latent(seed=34)
        sigma = torch.zeros(clean.shape[0], 1, 1, 1)
        noisy = noise_latent_at_timestep(clean, noise, sigma)
        assert torch.allclose(noisy, clean)

    def test_noising_at_sigma_one_returns_pure_noise(self):
        clean = _rand_latent(seed=35)
        noise = _rand_latent(seed=36)
        sigma = torch.ones(clean.shape[0], 1, 1, 1)
        noisy = noise_latent_at_timestep(clean, noise, sigma)
        assert torch.allclose(noisy, noise)

    def test_target_and_noising_convention_are_mutually_consistent(self):
        """A real correctness check, not just individually-correct
        pieces: predicting flow_matching_velocity_target at any sigma
        and taking a full Euler step from the noisy latent should recover
        the clean latent — this is what "consistent sign convention"
        actually cashes out to for a flow-matching sampler."""
        clean = _rand_latent(seed=37)
        noise = _rand_latent(seed=38)
        sigma = torch.full((clean.shape[0], 1, 1, 1), 0.7)
        noisy = noise_latent_at_timestep(clean, noise, sigma)
        v_target = flow_matching_velocity_target(clean, noise)
        # Euler step of size sigma, in the direction that REDUCES sigma
        # toward 0 (denoising direction): x_0_pred = x_t - sigma * v
        recovered_clean = noisy - sigma * v_target
        assert torch.allclose(recovered_clean, clean, atol=1e-5)
