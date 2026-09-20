from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from krisna_training.sketch.checkpoint_io import (
    load_checkpoint_for_resume,
    save_checkpoint,
)
from krisna_training.sketch.model import SketchModelConfig, build_model


@pytest.fixture
def tiny_config() -> SketchModelConfig:
    return SketchModelConfig(
        vocab_size=32, grid_h=2, grid_w=2, hidden_dim=16, n_layers=1, n_heads=2,
        ffn_dim=32, dropout=0.0, prompt_dim=8,
    )


def test_save_and_resume_roundtrip(tmp_path, tiny_config):
    model = build_model(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tiny_config, step=42, optimizer=optimizer)

    loaded_model, loaded_cfg, step, opt_state = load_checkpoint_for_resume(path)
    assert step == 42
    assert loaded_cfg.grid_h == 2
    assert loaded_cfg.vocab_size == 32
    assert opt_state is not None
    assert isinstance(loaded_model, torch.nn.Module)


def test_checkpoint_has_required_inference_side_keys(tmp_path, tiny_config):
    """MaskGITSketchModel.from_checkpoint() (inference/maskgit_model.py)
    does ckpt["model_state_dict"] / ckpt["config"] / ckpt["grid_h"] /
    ckpt["grid_w"] / ckpt["mask_token_id"] directly — these keys must
    exist, unrenamed."""
    model = build_model(tiny_config)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tiny_config, step=0)

    raw = torch.load(path, weights_only=False)
    assert "model_state_dict" in raw
    assert "config" in raw
    assert raw["grid_h"] == tiny_config.grid_h
    assert raw["grid_w"] == tiny_config.grid_w
    assert raw["mask_token_id"] == tiny_config.mask_token_id


def test_loaded_model_produces_same_output_as_saved(tmp_path, tiny_config):
    model = build_model(tiny_config)
    model.eval()

    tokens = torch.randint(0, tiny_config.mask_token_id + 1, (1, tiny_config.seq_len))
    mask = torch.zeros(1, tiny_config.seq_len)
    prompt = torch.randn(1, tiny_config.prompt_dim)

    with torch.no_grad():
        original_logits, _ = model(tokens, mask, prompt)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tiny_config, step=0)
    loaded_model, _, _, _ = load_checkpoint_for_resume(path)
    loaded_model.eval()

    with torch.no_grad():
        loaded_logits, _ = loaded_model(tokens, mask, prompt)

    assert torch.allclose(original_logits, loaded_logits, atol=1e-6)


def test_inference_side_loader_actually_loads_this_checkpoint(tmp_path, tiny_config):
    """End-to-end contract check: a checkpoint written by THIS training
    code loads successfully through the actual inference-side
    MaskGITSketchModel.from_checkpoint — not just a shape-compatible
    lookalike."""
    from krisna_inference.backends.maskgit_model import MaskGITSketchModel

    model = build_model(tiny_config)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tiny_config, step=0)

    sketch_model = MaskGITSketchModel.from_checkpoint(str(path), device="cpu")
    assert sketch_model.grid_h == tiny_config.grid_h
    assert sketch_model.grid_w == tiny_config.grid_w
    assert sketch_model.mask_token_id == tiny_config.mask_token_id

    prompt_embedding = torch.zeros(1, tiny_config.prompt_dim)
    result = sketch_model.sample(prompt_embedding, num_rounds=2)
    assert len(result["tokens"]) == tiny_config.seq_len
    assert len(result["confidence_map"]) == tiny_config.seq_len
    assert all(0 <= t < tiny_config.vocab_size for t in result["tokens"])
