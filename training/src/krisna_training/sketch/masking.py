"""Training-time masking. This is deliberately NOT the same as inference-
time sampling (inference_layer's maskgit_model.py uses Halton-ordered,
confidence-gated reveal across several rounds) — training uses the
standard MaskGIT recipe: sample a random mask RATIO via the cosine
schedule, then mask a uniformly random subset of positions at that ratio,
once per example, per training step. The model has to learn to fill in
masks at every ratio and every spatial arrangement, not just the specific
reveal order the inference sampler happens to use.
"""

from __future__ import annotations

import math
import random


def sample_mask_ratio() -> float:
    """r ~ Uniform(0,1), ratio = cos(r * pi/2) — same cosine SHAPE as
    inference/maskgit_model.py's cosine_mask_schedule, but here r is
    sampled per training example rather than derived from a fixed step
    count, per the standard MaskGIT training recipe."""
    r = random.random()
    return math.cos(r * math.pi / 2)


def apply_random_mask(tokens: "list[int]", mask_token_id: int, mask_ratio: float | None = None):
    """tokens: flat list of ground-truth token ids, length N.
    Returns (masked_tokens, mask_positions) where mask_positions is the
    sorted list of indices that were replaced with mask_token_id."""
    n = len(tokens)
    ratio = mask_ratio if mask_ratio is not None else sample_mask_ratio()
    num_masked = max(1, min(n, round(n * ratio)))

    mask_positions = sorted(random.sample(range(n), num_masked))
    masked_tokens = list(tokens)
    for i in mask_positions:
        masked_tokens[i] = mask_token_id
    return masked_tokens, mask_positions
