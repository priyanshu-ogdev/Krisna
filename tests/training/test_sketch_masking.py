from __future__ import annotations

from krisna_training.sketch.masking import apply_random_mask, sample_mask_ratio


def test_sample_mask_ratio_in_unit_interval():
    for _ in range(50):
        r = sample_mask_ratio()
        assert 0.0 <= r <= 1.0


def test_apply_random_mask_replaces_correct_count():
    tokens = list(range(100))
    masked, positions = apply_random_mask(tokens, mask_token_id=999, mask_ratio=0.3)
    assert len(positions) == 30
    assert sum(1 for t in masked if t == 999) == 30


def test_apply_random_mask_preserves_unmasked_tokens():
    tokens = list(range(20))
    masked, positions = apply_random_mask(tokens, mask_token_id=999, mask_ratio=0.5)
    for i, (orig, m) in enumerate(zip(tokens, masked)):
        if i not in positions:
            assert m == orig


def test_apply_random_mask_always_masks_at_least_one():
    tokens = list(range(10))
    masked, positions = apply_random_mask(tokens, mask_token_id=999, mask_ratio=0.0)
    assert len(positions) >= 1


def test_apply_random_mask_full_ratio_masks_everything():
    tokens = list(range(10))
    masked, positions = apply_random_mask(tokens, mask_token_id=999, mask_ratio=1.0)
    assert len(positions) == 10
    assert all(t == 999 for t in masked)
