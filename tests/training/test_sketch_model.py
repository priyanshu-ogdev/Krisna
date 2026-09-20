from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from krisna_training.sketch.model import SketchModelConfig, build_model


@pytest.fixture
def tiny_config() -> SketchModelConfig:
    return SketchModelConfig(
        vocab_size=64, grid_h=4, grid_w=4, hidden_dim=32, n_layers=2, n_heads=2,
        ffn_dim=64, dropout=0.0, prompt_dim=16,
    )


def test_mask_token_id_is_one_past_vocab(tiny_config):
    assert tiny_config.mask_token_id == 64


def test_seq_len_is_grid_area(tiny_config):
    assert tiny_config.seq_len == 16


def test_forward_output_shapes(tiny_config):
    model = build_model(tiny_config)
    B, N = 3, tiny_config.seq_len
    tokens = torch.randint(0, tiny_config.mask_token_id + 1, (B, N))
    mask = torch.zeros(B, N)
    prompt_embedding = torch.randn(B, tiny_config.prompt_dim)

    logits, critic_scores = model(tokens, mask, prompt_embedding)

    assert logits.shape == (B, N, tiny_config.vocab_size)
    assert critic_scores.shape == (B, N)


def test_critic_scores_in_unit_interval(tiny_config):
    model = build_model(tiny_config)
    B, N = 2, tiny_config.seq_len
    tokens = torch.randint(0, tiny_config.mask_token_id + 1, (B, N))
    mask = torch.zeros(B, N)
    prompt_embedding = torch.randn(B, tiny_config.prompt_dim)

    _, critic_scores = model(tokens, mask, prompt_embedding)
    assert (critic_scores >= 0).all() and (critic_scores <= 1).all()


def test_all_masked_input_still_produces_valid_logits(tiny_config):
    model = build_model(tiny_config)
    B, N = 2, tiny_config.seq_len
    tokens = torch.full((B, N), tiny_config.mask_token_id)
    mask = torch.ones(B, N)
    prompt_embedding = torch.randn(B, tiny_config.prompt_dim)

    logits, critic_scores = model(tokens, mask, prompt_embedding)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(critic_scores).any()


def test_wrong_seq_len_raises(tiny_config):
    model = build_model(tiny_config)
    tokens = torch.randint(0, tiny_config.mask_token_id + 1, (1, tiny_config.seq_len + 1))
    mask = torch.zeros(1, tiny_config.seq_len + 1)
    prompt_embedding = torch.randn(1, tiny_config.prompt_dim)

    with pytest.raises(AssertionError):
        model(tokens, mask, prompt_embedding)


def test_gradients_flow_to_both_heads(tiny_config):
    model = build_model(tiny_config)
    B, N = 2, tiny_config.seq_len
    tokens = torch.randint(0, tiny_config.mask_token_id + 1, (B, N))
    mask = torch.ones(B, N)
    targets = torch.randint(0, tiny_config.vocab_size, (B, N))
    prompt_embedding = torch.randn(B, tiny_config.prompt_dim)

    from krisna_training.sketch.losses import compute_loss

    logits, critic_scores = model(tokens, mask, prompt_embedding)
    loss, _, _ = compute_loss(logits, critic_scores, targets, mask)
    loss.backward()

    assert model.token_head.weight.grad is not None
    assert model.critic_head.weight.grad is not None
    assert not torch.isnan(model.token_head.weight.grad).any()
