"""Bicubic positional-embedding interpolation — the standard ViT trick
(used by DeiT, MAE, etc.) for transferring a transformer trained at one
grid resolution to a larger one, which is what PRD §6.1's "256->512px
progressive training" needs: train at 16x16 (256px @ f16), then continue
training at 32x32 (512px) initialized from the 256px run rather than from
scratch.

UPGRADE, SOTA research pass (documented here, not silently applied — see
model.py's SketchModelConfig.use_2d_rope and its full citation trail):
this whole interpolation trick exists only because the model's default
path uses a FIXED-SIZE learned absolute position table (model.py's
`pos_embed` parameter), which has no defined values for a grid size it
wasn't trained at — hence needing to manufacture them by resizing. The
model.py's opt-in `use_2d_rope=True` path replaces that table with 2D
axial rotary position embeddings, which are a CLOSED-FORM function of
(row, col) valid for any grid size with no interpolation and no
extra parameters at all — the real SOTA-aligned way most current
DiT-family image models (FLUX, SD3-class, and Seedream 3.0's own
"cross-modality RoPE" per its technical report, arXiv:2504.11346)
avoid this problem entirely rather than working around it. This module
is NOT obsolete — it's still exactly correct and necessary for the
default (use_2d_rope=False) path, and switching an ALREADY-TRAINED
256px checkpoint to the RoPE path isn't a drop-in operation (see that
config field's docstring) — but a NEW from-scratch training run has a
real, cited reason to consider starting on the RoPE path instead of
ever needing this interpolation step at all.
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
