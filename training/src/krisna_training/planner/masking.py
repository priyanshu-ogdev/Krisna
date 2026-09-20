"""Assistant-turn loss masking for chat SFT — kept as pure functions over
token-count/boundary data, deliberately decoupled from tokenizer specifics,
so the actual masking LOGIC is unit-testable without downloading Qwen3.5's
tokenizer. `dataset.py` is the thin, tokenizer-dependent layer that turns a
real conversation into the boundaries this module consumes.
"""

from __future__ import annotations


def build_loss_mask_from_boundaries(
    roles: list[str], boundaries: list[int], total_len: int
) -> list[int]:
    """roles: one entry per conversation turn, e.g. ["system", "user",
    "assistant", "user", "assistant"].
    boundaries: cumulative token count after each turn INCLUDING a leading
    0, so len(boundaries) == len(roles) + 1 — e.g. [0, 12, 20, 40, 45, 60]
    means turn 0 (system) spans tokens [0,12), turn 1 (user) spans
    [12,20), etc.
    total_len: length of the actual tokenized sequence — boundaries are
    clipped to this (a chat template can add trailing tokens after the
    last turn's content that don't belong to any turn; those get mask=0).

    Returns a flat 0/1 list of length total_len — 1 where loss should be
    computed (assistant-turn tokens), 0 elsewhere.
    """
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
    """A conversation with an empty/degenerate mask (no assistant turns,
    or all-truncated-away) would silently contribute a NaN/zero-gradient
    loss if trained on — callers should check this and skip such examples
    rather than let it happen unnoticed."""
    return any(mask)
