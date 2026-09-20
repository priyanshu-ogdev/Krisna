from __future__ import annotations

import pytest

from krisna_training.planner.masking import (
    build_loss_mask_from_boundaries,
    mask_has_any_loss_tokens,
)


def test_masks_only_assistant_turns():
    roles = ["system", "user", "assistant", "user", "assistant"]
    boundaries = [0, 5, 10, 20, 25, 30]
    mask = build_loss_mask_from_boundaries(roles, boundaries, total_len=30)

    assert mask[0:5] == [0] * 5     # system
    assert mask[5:10] == [0] * 5    # user
    assert mask[10:20] == [1] * 10  # assistant
    assert mask[20:25] == [0] * 5   # user
    assert mask[25:30] == [1] * 5   # assistant


def test_no_assistant_turns_gives_all_zero_mask():
    roles = ["system", "user"]
    boundaries = [0, 5, 10]
    mask = build_loss_mask_from_boundaries(roles, boundaries, total_len=10)
    assert mask == [0] * 10
    assert not mask_has_any_loss_tokens(mask)


def test_mismatched_boundaries_length_raises():
    with pytest.raises(ValueError, match="boundaries must have"):
        build_loss_mask_from_boundaries(["user", "assistant"], [0, 5], total_len=5)


def test_non_monotonic_boundaries_raises():
    with pytest.raises(ValueError, match="non-decreasing"):
        build_loss_mask_from_boundaries(["user", "assistant"], [0, 10, 5], total_len=10)


def test_truncation_clips_final_boundary():
    # total_len shorter than the last boundary — e.g. sequence got truncated
    # to max_length after boundaries were computed on the full sequence.
    roles = ["user", "assistant"]
    boundaries = [0, 5, 20]
    mask = build_loss_mask_from_boundaries(roles, boundaries, total_len=12)
    assert len(mask) == 12
    assert mask[5:12] == [1] * 7  # assistant span clipped to available length


def test_mask_has_any_loss_tokens():
    assert mask_has_any_loss_tokens([0, 0, 1, 0])
    assert not mask_has_any_loss_tokens([0, 0, 0])
    assert not mask_has_any_loss_tokens([])
