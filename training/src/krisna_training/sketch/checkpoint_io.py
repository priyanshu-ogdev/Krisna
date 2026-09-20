"""Checkpoint format.

Stores `model.state_dict()` (a plain dict of tensors) plus the
`SketchModelConfig` needed to reconstruct the architecture via
`training.sketch.model.build_model()` — NOT a pickled model object.
Pickling live `nn.Module` instances directly is fragile (breaks across
refactors, and torch.save can't pickle a class that's defined inside a
function, which build_model()'s SketchTransformer deliberately is, to keep
this package's lazy-import discipline). state_dict + config is the
standard, robust PyTorch practice, and it's what
inference/maskgit_model.py's MaskGITSketchModel.from_checkpoint() expects:
it rebuilds the architecture from `ckpt["config"]` via the SAME
build_model() this training code uses, then loads
`ckpt["model_state_dict"]` into it — so the architecture definition lives
in exactly one place (model.py), not duplicated between training and
inference.

Required keys (from_checkpoint reads these directly): "model_state_dict",
"config", "grid_h", "grid_w", "mask_token_id".
"""

from __future__ import annotations

from pathlib import Path


def save_checkpoint(path: str | Path, model, config, step: int, optimizer=None) -> None:
    import torch

    payload = {
        # Required by MaskGITSketchModel.from_checkpoint — do not rename.
        "model_state_dict": model.state_dict(),
        "config": config,
        "grid_h": config.grid_h,
        "grid_w": config.grid_w,
        "mask_token_id": config.mask_token_id,
        # Extra metadata — training-side only, inference loader ignores this.
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint_for_resume(path: str | Path):
    """Training-side resume — returns (model, config, step, optimizer_state
    | None). Rebuilds the architecture via build_model(config) and loads
    the state dict into it, same as the inference-side loader does."""
    import torch

    from krisna_training.sketch.model import build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt["config"], ckpt["step"], ckpt.get("optimizer_state")
