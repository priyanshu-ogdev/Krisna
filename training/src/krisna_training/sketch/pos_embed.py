"""Bicubic positional-embedding interpolation — the standard ViT trick
(used by DeiT, MAE, etc.) for transferring a transformer trained at one
grid resolution to a larger one, which is what PRD §6.1's "256->512px
progressive training" needs: train at 16x16 (256px @ f16), then continue
training at 32x32 (512px) initialized from the 256px run rather than from
scratch.
"""

from __future__ import annotations


def interpolate_pos_embed(pos_embed, old_grid_h: int, old_grid_w: int, new_grid_h: int, new_grid_w: int):
    """pos_embed: a torch.Tensor of shape [old_grid_h * old_grid_w, dim].
    Returns a new tensor of shape [new_grid_h * new_grid_w, dim]."""
    import torch
    import torch.nn.functional as F

    dim = pos_embed.shape[-1]
    grid = pos_embed.reshape(1, old_grid_h, old_grid_w, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_grid_h, new_grid_w), mode="bicubic", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(new_grid_h * new_grid_w, dim)
