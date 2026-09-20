from __future__ import annotations

import pytest

from krisna_training.critic.masking import (
    build_loss_mask_from_boundaries,
    mask_has_any_loss_tokens,
)


def test_single_turn_prompt_then_assistant_json():
    # The critic's actual use case: one user turn (image+prompt), one
    # assistant turn (target critique JSON).
    mask = build_loss_mask_from_boundaries(["user", "assistant"], [0, 50, 80], total_len=80)
    assert mask[:50] == [0] * 50
    assert mask[50:80] == [1] * 30


def test_no_assistant_turn_gives_all_zero_mask():
    mask = build_loss_mask_from_boundaries(["user"], [0, 20], total_len=20)
    assert mask == [0] * 20
    assert not mask_has_any_loss_tokens(mask)


def test_mismatched_boundaries_length_raises():
    with pytest.raises(ValueError, match="boundaries must have"):
        build_loss_mask_from_boundaries(["user", "assistant"], [0, 5], total_len=5)


def test_non_monotonic_boundaries_raises():
    with pytest.raises(ValueError, match="non-decreasing"):
        build_loss_mask_from_boundaries(["user", "assistant"], [0, 10, 5], total_len=10)


def test_truncation_clips_final_boundary():
    mask = build_loss_mask_from_boundaries(["user", "assistant"], [0, 50, 200], total_len=100)
    assert len(mask) == 100
    assert mask[50:100] == [1] * 50


def test_mask_has_any_loss_tokens():
    assert mask_has_any_loss_tokens([0, 1, 0])
    assert not mask_has_any_loss_tokens([0, 0, 0])
