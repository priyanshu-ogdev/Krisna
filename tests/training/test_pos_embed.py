from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from krisna_training.sketch.pos_embed import interpolate_pos_embed


def test_output_shape_matches_new_grid():
    pos_embed = torch.randn(16, 32)  # 4x4 grid, dim=32
    new_pos = interpolate_pos_embed(pos_embed, old_grid_h=4, old_grid_w=4, new_grid_h=8, new_grid_w=8)
    assert new_pos.shape == (64, 32)


def test_downscale_also_works():
    pos_embed = torch.randn(64, 16)  # 8x8 grid
    new_pos = interpolate_pos_embed(pos_embed, old_grid_h=8, old_grid_w=8, new_grid_h=4, new_grid_w=4)
    assert new_pos.shape == (16, 16)


def test_identity_when_grid_unchanged():
    pos_embed = torch.randn(16, 8)
    new_pos = interpolate_pos_embed(pos_embed, old_grid_h=4, old_grid_w=4, new_grid_h=4, new_grid_w=4)
    assert new_pos.shape == pos_embed.shape
    assert torch.allclose(new_pos, pos_embed, atol=1e-5)


def test_no_nans_produced():
    pos_embed = torch.randn(16, 32)
    new_pos = interpolate_pos_embed(pos_embed, old_grid_h=4, old_grid_w=4, new_grid_h=32, new_grid_w=32)
    assert not torch.isnan(new_pos).any()
