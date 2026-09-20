from __future__ import annotations

import math

from krisna_inference.backends.maskgit_model import (
    cosine_mask_schedule,
    halton_sequence,
    halton_token_order,
)


def test_cosine_schedule_endpoints():
    # step=total_steps -> ratio=1 -> cos(pi/2) = 0 (fully revealed)
    assert cosine_mask_schedule(8, 8) == math.cos(math.pi / 2)
    assert abs(cosine_mask_schedule(8, 8)) < 1e-10


def test_cosine_schedule_monotonically_decreasing():
    values = [cosine_mask_schedule(s, 8) for s in range(1, 9)]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))


def test_cosine_schedule_first_step_mostly_masked():
    # step=1 of 8 should still show most tokens masked (schedule close to 1).
    assert cosine_mask_schedule(1, 8) > 0.9


def test_halton_sequence_in_unit_interval():
    seq = halton_sequence(50, base=2)
    assert all(0.0 <= v < 1.0 for v in seq)
    assert len(seq) == 50
    assert len(set(seq)) == 50  # Halton sequence terms are distinct


def test_halton_token_order_covers_full_grid_exactly_once():
    order = halton_token_order(grid_h=8, grid_w=8)
    assert sorted(order) == list(range(64))
    assert len(order) == 64


def test_halton_token_order_is_spatially_dispersed_not_raster():
    # The first several picks should not just be the raster-order top row —
    # that's the whole point of using Halton instead of scanning order.
    order = halton_token_order(grid_h=16, grid_w=16)
    first_ten = order[:10]
    raster_first_ten = list(range(10))
    assert first_ten != raster_first_ten


def test_halton_token_order_deterministic():
    a = halton_token_order(grid_h=10, grid_w=10)
    b = halton_token_order(grid_h=10, grid_w=10)
    assert a == b
