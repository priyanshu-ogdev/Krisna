from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from krisna_training.planner.train import make_collate_fn


def test_collate_pads_to_longest_in_batch():
    collate = make_collate_fn(pad_token_id=0)
    batch = [
        {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]},
        {"input_ids": [4, 5], "loss_mask": [0, 1]},
    ]
    out = collate(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"].shape == (2, 3)
    assert out["labels"].shape == (2, 3)


def test_collate_pads_input_ids_with_pad_token():
    collate = make_collate_fn(pad_token_id=99)
    batch = [{"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1]}, {"input_ids": [4], "loss_mask": [1]}]
    out = collate(batch)
    assert out["input_ids"][1].tolist() == [4, 99, 99]


def test_collate_attention_mask_zero_on_padding():
    collate = make_collate_fn(pad_token_id=0)
    batch = [{"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1]}, {"input_ids": [4], "loss_mask": [1]}]
    out = collate(batch)
    assert out["attention_mask"][1].tolist() == [1, 0, 0]


def test_collate_labels_use_ignore_index_where_unmasked_or_padded():
    collate = make_collate_fn(pad_token_id=0)
    batch = [{"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]}]
    out = collate(batch)
    assert out["labels"][0].tolist() == [-100, 2, 3]


def test_collate_padding_region_is_ignore_index():
    collate = make_collate_fn(pad_token_id=0)
    batch = [{"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1]}, {"input_ids": [4], "loss_mask": [1]}]
    out = collate(batch)
    assert out["labels"][1].tolist() == [4, -100, -100]
