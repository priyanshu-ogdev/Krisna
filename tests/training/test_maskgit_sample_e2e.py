from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from krisna_inference.backends.maskgit_model import MaskGITSketchModel
from krisna_training.sketch.model import SketchModelConfig, build_model


@pytest.mark.parametrize("grid_h,grid_w,num_rounds", [(2, 2, 2), (4, 4, 3), (4, 4, 8), (8, 8, 1), (3, 3, 5)])
def test_sample_never_leaves_masked_tokens(grid_h, grid_w, num_rounds):
    """Regression test: the reveal-count schedule previously undercounted
    due to rounding, leaving mask_token_id values in the output on small
    grids / low round counts. sample() must ALWAYS fully reveal by the
    final round, for any grid size or round count including edge cases
    (num_rounds=1, non-power-of-2 grids)."""
    cfg = SketchModelConfig(
        vocab_size=32, grid_h=grid_h, grid_w=grid_w, hidden_dim=16, n_layers=1,
        n_heads=2, ffn_dim=32, dropout=0.0, prompt_dim=8,
    )
    module = build_model(cfg)
    module.eval()
    sketch_model = MaskGITSketchModel(module, grid_h=grid_h, grid_w=grid_w, mask_token_id=cfg.mask_token_id)

    prompt_embedding = torch.zeros(1, cfg.prompt_dim)
    result = sketch_model.sample(prompt_embedding, num_rounds=num_rounds)

    assert len(result["tokens"]) == grid_h * grid_w
    assert all(t != cfg.mask_token_id for t in result["tokens"]), (
        f"mask_token_id survived sampling with grid={grid_h}x{grid_w}, num_rounds={num_rounds}"
    )
    assert all(0 <= t < cfg.vocab_size for t in result["tokens"])
