from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from krisna_training.sketch.losses import compute_loss


def test_loss_is_finite_and_positive():
    B, N, V = 2, 8, 16
    logits = torch.randn(B, N, V, requires_grad=True)
    critic_scores = torch.sigmoid(torch.randn(B, N))
    targets = torch.randint(0, V, (B, N))
    mask = torch.ones(B, N)

    total, token_loss, critic_loss = compute_loss(logits, critic_scores, targets, mask)
    assert torch.isfinite(total)
    assert total.item() > 0
    assert token_loss.item() > 0


def test_only_masked_positions_contribute_to_token_loss():
    B, N, V = 1, 4, 16
    targets = torch.tensor([[0, 1, 2, 3]])
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # only position 0 counts

    # Make logits perfect everywhere EXCEPT the one masked position, where
    # they're maximally wrong — loss should still be large (driven only by
    # the masked position), not diluted by the correct unmasked ones.
    logits = torch.zeros(B, N, V)
    for i in range(N):
        logits[0, i, targets[0, i]] = 100.0  # confidently correct everywhere
    logits[0, 0, targets[0, 0]] = -100.0     # except the masked position: confidently WRONG
    logits[0, 0, (targets[0, 0].item() + 1) % V] = 100.0

    critic_scores = torch.full((B, N), 0.5)
    total, token_loss, _ = compute_loss(logits, critic_scores, targets, mask)
    assert token_loss.item() > 10  # large — driven entirely by the one wrong masked position


def test_critic_target_matches_generator_correctness():
    # If the generator's top-1 prediction is correct everywhere, and the
    # critic confidently predicts "correct" (score near 1) everywhere, critic
    # loss should be near zero.
    B, N, V = 1, 4, 8
    targets = torch.tensor([[0, 1, 2, 3]])
    logits = torch.zeros(B, N, V)
    for i in range(N):
        logits[0, i, targets[0, i]] = 100.0  # argmax == target everywhere

    critic_scores = torch.full((B, N), 0.999)
    mask = torch.ones(B, N)

    _, _, critic_loss = compute_loss(logits, critic_scores, targets, mask)
    assert critic_loss.item() < 0.01


def test_zero_mask_uses_epsilon_denominator_not_nan():
    B, N, V = 1, 4, 8
    logits = torch.randn(B, N, V)
    critic_scores = torch.sigmoid(torch.randn(B, N))
    targets = torch.randint(0, V, (B, N))
    mask = torch.zeros(B, N)  # nothing masked — degenerate case

    total, token_loss, critic_loss = compute_loss(logits, critic_scores, targets, mask)
    assert torch.isfinite(total)
