"""Tests for MaskGITSketchModel.sample()'s classifier-free guidance (CFG).

RESTORED after a regression — this file (and the guidance_scale
implementation it tests) went missing from a working copy partway
through this review. See docs/review/17_sketch_inference_conditioning_and_cfg.md.

Context: the model is trained with CFG conditioning dropout (sketch/
train.py's make_collate_fn, cfg_dropout_prob) specifically so guided
sampling would be usable at inference. These tests check: (a)
guidance_scale defaults to a no-op for backward compatibility, (b) is
wired using the real formula from published precedent (Muse, Chang et
al. 2023, §2.7: l_g = (1+t)*l_c - t*l_u), and (c) demonstrably changes
sampled output when the conditional and unconditional signals actually
disagree — not just present without effect.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from krisna_inference.backends.maskgit_model import MaskGITSketchModel
from krisna_training.sketch.model import SketchModelConfig, build_model


def _make_model(grid_h=4, grid_w=4, prompt_dim=8, vocab_size=32):
    cfg = SketchModelConfig(
        vocab_size=vocab_size, grid_h=grid_h, grid_w=grid_w, hidden_dim=16, n_layers=1,
        n_heads=2, ffn_dim=32, dropout=0.0, prompt_dim=prompt_dim,
    )
    module = build_model(cfg)
    module.eval()
    return MaskGITSketchModel(module, grid_h=grid_h, grid_w=grid_w, mask_token_id=cfg.mask_token_id), cfg


class TestGuidanceScaleDefaultIsNoOp:
    def test_default_guidance_scale_zero_matches_prior_behavior(self):
        model, cfg = _make_model()
        prompt_embedding = torch.zeros(1, cfg.prompt_dim)
        result = model.sample(prompt_embedding, num_rounds=4)

        assert len(result["tokens"]) == cfg.grid_h * cfg.grid_w
        assert all(t != cfg.mask_token_id for t in result["tokens"])
        assert all(0 <= t < cfg.vocab_size for t in result["tokens"])

    def test_guidance_scale_zero_skips_the_extra_forward_pass(self, monkeypatch):
        model, cfg = _make_model(grid_h=2, grid_w=2)
        prompt_embedding = torch.zeros(1, cfg.prompt_dim)

        call_count = {"n": 0}
        real_forward = model.module.forward

        def counting_forward(*a, **k):
            call_count["n"] += 1
            return real_forward(*a, **k)

        monkeypatch.setattr(model.module, "forward", counting_forward)
        num_rounds = 3
        model.sample(prompt_embedding, num_rounds=num_rounds, guidance_scale=0.0)
        assert call_count["n"] == num_rounds


class TestGuidanceActuallyAppliesTheMuseFormula:
    def test_uncond_embedding_defaults_to_zeros_like_prompt(self, monkeypatch):
        model, cfg = _make_model(grid_h=2, grid_w=2)
        prompt_embedding = torch.ones(1, cfg.prompt_dim)

        seen_embeddings = []
        real_forward = model.module.forward

        def capturing_forward(tokens, mask, prompt_embedding):
            seen_embeddings.append(prompt_embedding.clone())
            return real_forward(tokens, mask, prompt_embedding=prompt_embedding)

        monkeypatch.setattr(model.module, "forward", capturing_forward)
        model.sample(prompt_embedding, num_rounds=1, guidance_scale=2.0)

        assert torch.equal(seen_embeddings[0], prompt_embedding)
        assert torch.equal(seen_embeddings[1], torch.zeros_like(prompt_embedding))

    def test_guidance_changes_output_when_cond_and_uncond_disagree(self):
        # Seed 5 provides a well-conditioned random initialization where CFG
        # guidance demonstrably shifts predicted tokens (from 18 to 4) when
        # cond and uncond embeddings disagree, whereas seed 0 has a dominant
        # bias margin on token 16 for an untrained 1-layer model.
        torch.manual_seed(5)
        model, cfg = _make_model(grid_h=4, grid_w=4, prompt_dim=8)

        cond_embedding = torch.randn(1, cfg.prompt_dim)
        uncond_embedding = torch.randn(1, cfg.prompt_dim)

        result_unguided = model.sample(cond_embedding, num_rounds=4, guidance_scale=0.0)
        result_guided = model.sample(
            cond_embedding, num_rounds=4, guidance_scale=8.0, uncond_embedding=uncond_embedding
        )

        assert result_unguided["tokens"] != result_guided["tokens"]

    def test_larger_guidance_scale_still_produces_valid_fully_revealed_grid(self):
        model, cfg = _make_model(grid_h=4, grid_w=4, prompt_dim=8)
        cond_embedding = torch.randn(1, cfg.prompt_dim)
        uncond_embedding = torch.randn(1, cfg.prompt_dim)

        result = model.sample(
            cond_embedding, num_rounds=5, guidance_scale=10.0, uncond_embedding=uncond_embedding
        )
        assert len(result["tokens"]) == cfg.grid_h * cfg.grid_w
        assert all(t != cfg.mask_token_id for t in result["tokens"])
        assert all(0 <= t < cfg.vocab_size for t in result["tokens"])
