"""REPA (REPresentation Alignment) — optional auxiliary training signal for
polish/train_dpo.py. UPGRADE (SOTA research pass, grounded in the real
paper rather than reasoned from the name alone):

Yu, Sihyun, et al. "Representation Alignment for Generation: Training
Diffusion Transformers Is Easier Than You Think." arXiv:2410.06940, 2024.
Official repo: github.com/sihyun-yu/REPA.

THE REAL METHOD (Eq. 8 of the paper, verified via the paper and its
official repo's README before writing this, not assumed from the name):
while training a diffusion transformer on noisy latents, add an auxiliary
loss that pulls the transformer's OWN intermediate hidden states (at one
specific, fairly early layer — the paper's official defaults use
`encoder-depth=8` out of SiT-XL's 28 layers, finding "regularizing only
the first few layers is sufficient... and yields the best results") toward
a FROZEN, pretrained self-supervised visual encoder's patch features
(DINOv2 in the paper) computed on the corresponding CLEAN image. The loss
is a negative mean patch-wise similarity (cosine similarity in the paper):

    L_REPA(theta, phi) = -E[ (1/N) * sum_n sim( y*[n], h_phi(h_t[n]) ) ]

y*[n]: patch n's DINOv2 feature of the clean image (frozen, no gradient).
h_t[n]: patch n's hidden state from the diffusion transformer at the
        hooked layer, processing the NOISY latent at timestep t.
h_phi: a trainable MLP projection head mapping the transformer's hidden
       dim to DINOv2's feature dim, since they're not required to match.
Total loss = L_diffusion + lambda * L_REPA, with the paper's own default
lambda (their `--proj-coeff`) = 0.5.

WHY THIS IS AN HONEST EXTRAPOLATION, NOT A DIRECT PORT — stated plainly
rather than left implicit: REPA's proven result (up to ~17.5x faster
convergence, better FID) is for FROM-SCRATCH DiT/SiT PRETRAINING on
ImageNet, where early-layer representations start from nothing. This
project's train_dpo.py does a comparatively short LoRA-based DPO
fine-tune of an ALREADY-PRETRAINED backbone (Z-Image-Turbo) — a model
whose early/mid layers should already carry reasonably good semantic
representations from its own real pretraining. Whether REPA's benefit
carries over to THIS regime (short-horizon preference fine-tuning of a
capable pretrained model, rather than pretraining from random init) is
NOT established by the paper and is not verified here — this ships as a
real, correctly-implemented, opt-in (default OFF, --repa-weight 0.0)
regularizer worth trying and measuring against the heldout validation
loop (finding #3, an earlier review pass) added for exactly this purpose
— not as a guaranteed win asserted from the paper's from-scratch-pretraining
numbers.

WHAT THIS FILE DELIBERATELY DOES NOT DO: guess Z-Image-Turbo's real
internal module names to auto-select a hook point (the same "flag rather
than guess" discipline as this project's other unverified third-party
API surfaces, e.g. train_dpo.py's --lora-target-modules). The caller must
supply --repa-hook-module (a dotted attribute path into the transformer,
e.g. "transformer_blocks.8"); this file raises a clear error if that path
doesn't resolve to a real submodule, rather than silently doing nothing.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Standard ImageNet normalization — used by facebook/dinov2-base's own
# AutoImageProcessor default (verified against HF's model docs before
# writing this), not assumed. This project's own DPO image pipeline
# (polish/dpo_dataset.py) normalizes with Normalize([0.5], [0.5]) — i.e.
# maps pixels to [-1, 1] for the VAE/diffusion path — which is a DIFFERENT
# normalization than DINOv2 expects, so it must be undone and redone
# rather than fed straight through.
_DINOV2_MEAN = (0.485, 0.456, 0.406)
_DINOV2_STD = (0.229, 0.224, 0.225)


class RepaProjectionHead:
    """Lazily-shaped: real construction happens in `build()` once the
    hooked hidden state's actual feature dimension is observed from a
    real forward pass, since this project has no verified, static
    knowledge of Z-Image-Turbo's internal hidden_dim to hardcode against
    (same reasoning as this module's docstring re: --repa-hook-module).
    A 3-layer MLP with SiLU activations — matches the general shape
    described for REPA's official projection head (a small trainable
    MLP, not a linear probe) — hidden width is a reasonable, documented
    choice (matching the larger of the two dims) rather than a value
    lifted from the paper's own SiT-XL-specific configuration, which
    doesn't apply to an architecture (Z-Image-Turbo) whose real hidden
    size isn't verified here.
    """

    @staticmethod
    def build(in_dim: int, out_dim: int):
        import torch.nn as nn

        hidden = max(in_dim, out_dim)
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )


def repa_loss(h_proj: "torch.Tensor", y_target: "torch.Tensor") -> "torch.Tensor":
    """Eq. 8 of the REPA paper: negative mean patch-wise cosine
    similarity. h_proj/y_target: [B, N, D] (same N, D after
    align_patch_grids + the projection head). y_target must already be
    detached / come from a no_grad context — this function does not
    detach it itself, since the caller (train_dpo.py) is the one that
    knows whether it ran the frozen encoder under no_grad.
    """
    import torch.nn.functional as F

    sim = F.cosine_similarity(h_proj, y_target, dim=-1)  # [B, N]
    return -sim.mean()


def align_patch_grids(source: "torch.Tensor", target_len: int) -> "torch.Tensor":
    """source: [B, N_src, D]. Returns [B, target_len, D].

    REPA's own setup (SiT + DINOv2, both ViT-style with comparable patch
    grids at the paper's chosen resolutions) lets patch counts line up
    naturally. Z-Image-Turbo's real latent patchification grid is not
    verified here, so N_src (DINOv2's patch count) and the diffusion
    transformer's hooked sequence length will not generally match.
    Rather than require the caller to know and hardcode both grids'
    exact (H, W) shapes (unverifiable for Z-Image-Turbo in this
    environment), this does a 1D linear interpolation along the
    sequence dimension. This is a deliberate simplification: a true 2D-
    aware resize (like pos_embed.py's bicubic interpolation) would
    better preserve spatial structure, but requires knowing both grids'
    real (H, W) — which requires exactly the kind of unverified
    assumption this project's discipline avoids. 1D interpolation
    is a strictly weaker but honest, dimension-agnostic fallback: it
    still aligns the two sequences' lengths so cosine similarity is
    well-defined, at the cost of not perfectly preserving 2D adjacency
    for whichever side has the coarser grid. Flagged here, not silently
    assumed to be spatially exact.
    """
    import torch.nn.functional as F

    if source.shape[1] == target_len:
        return source
    # F.interpolate expects [B, C, N] for 1D linear interpolation.
    x = source.transpose(1, 2)  # [B, D, N_src]
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.transpose(1, 2)  # [B, target_len, D]


def preprocess_for_dinov2(pixel_values_neg1_to_1: "torch.Tensor", size: int = 224) -> "torch.Tensor":
    """pixel_values_neg1_to_1: [B, 3, H, W], normalized as
    polish/dpo_dataset.py does (Normalize([0.5],[0.5]) -> range [-1, 1]).
    Returns a tensor preprocessed the way facebook/dinov2-base's own
    AutoImageProcessor default does (resize + ImageNet mean/std),
    decoupled from whatever resolution the diffusion model itself trains
    at — DINOv2's encoder doesn't need to see the same resolution the
    policy transformer is being trained on; `size` (default 224, a
    standard DINOv2 input size, divisible by its patch_size=14 — 224/14
    = 16, giving a clean 16x16=256-patch grid) just needs to be a
    reasonable size for the FROZEN encoder to extract good features
    from, independent of the diffusion training resolution.
    """
    import torch.nn.functional as F

    x = pixel_values_neg1_to_1.clamp(-1.0, 1.0) * 0.5 + 0.5  # undo Normalize([0.5],[0.5]) -> [0, 1]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = x.new_tensor(_DINOV2_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_DINOV2_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def resolve_submodule(root, dotted_path: str):
    """root.transformer_blocks.8 style lookup, supporting both attribute
    access and integer indexing into a ModuleList/Sequential (the "8" in
    "transformer_blocks.8"). Raises a clear, actionable error rather than
    an opaque AttributeError — this is exactly the kind of unverified
    path this project's discipline says to fail loudly on, not guess
    around.
    """
    obj = root
    for part in dotted_path.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (TypeError, IndexError, KeyError) as e:
                raise ValueError(
                    f"--repa-hook-module {dotted_path!r}: {part!r} is not a valid index into "
                    f"{type(obj).__name__} — check the real module structure for this pipeline's "
                    "transformer class before setting this flag."
                ) from e
        else:
            if not hasattr(obj, part):
                raise ValueError(
                    f"--repa-hook-module {dotted_path!r}: {type(obj).__name__} has no attribute "
                    f"{part!r}. This path is NOT verified against Z-Image-Turbo's real module "
                    "names in this environment — inspect the actual loaded transformer "
                    "(e.g. `print(policy_pipe.transformer)`) to find the real dotted path to a "
                    "real intermediate block before setting --repa-hook-module."
                )
            obj = getattr(obj, part)
    return obj
