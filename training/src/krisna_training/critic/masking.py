"""Assistant-turn loss masking — intentionally the same pure logic as
training/planner/masking.py, duplicated rather than imported. Both are
pure stdlib functions (no cross-package import needed for either to work),
but this copy exists specifically so `train_critic_qlora.py` (which must
run under venv-critic, an entirely separate Python environment from the
Planner training code) never needs krisna_training.planner
importable at all — the critic package is self-contained end to end.
"""

from __future__ import annotations


def build_loss_mask_from_boundaries(
    roles: list[str], boundaries: list[int], total_len: int
) -> list[int]:
    """See training/planner/masking.py's docstring for the full
    explanation — identical contract. roles: one entry per turn (for the
    critic, typically just ["system", "user", "assistant"] — one prompt,
    one target critique JSON). boundaries: cumulative token count after
    each turn, including a leading 0 (len == len(roles)+1). total_len:
    actual tokenized sequence length, for clipping trailing template
    tokens."""
    if len(boundaries) != len(roles) + 1:
        raise ValueError(
            f"boundaries must have len(roles)+1 entries, got {len(boundaries)} "
            f"for {len(roles)} roles"
        )
    if boundaries != sorted(boundaries):
        raise ValueError("boundaries must be non-decreasing")

    mask = [0] * total_len
    for i, role in enumerate(roles):
        start, end = boundaries[i], min(boundaries[i + 1], total_len)
        if role != "assistant":
            continue
        for j in range(max(0, start), end):
            mask[j] = 1
    return mask


def mask_has_any_loss_tokens(mask: list[int]) -> bool:
    return any(mask)
