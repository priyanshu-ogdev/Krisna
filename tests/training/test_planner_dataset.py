from __future__ import annotations

import json

import pytest

from krisna_training.planner.dataset import (
    PlannerSFTDataset,
    tokenize_conversation_with_mask,
)


class FakeTokenizer:
    """Deterministic stand-in for a real chat tokenizer — this sandbox has
    no network access to huggingface.co to download Qwen3.5's real
    tokenizer, so dataset.py's boundary-detection LOGIC (tokenize each
    successive message prefix, diff the lengths) is validated against a
    fake that's fully predictable instead. Each message contributes
    exactly `len(content)` synthetic token ids plus one "role marker"
    token — deterministic and easy to hand-verify in assertions.
    """

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for msg in messages:
            ids.append(hash(msg["role"]) % 1000)          # 1 role-marker token
            ids.extend(range(len(msg["content"])))          # len(content) content tokens
        if add_generation_prompt:
            ids.append(999)  # a trailing generation-prompt token
        return ids


def test_tokenize_conversation_boundaries_and_mask():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "abc"},        # 1 + 3 = 4 tokens
        {"role": "user", "content": "hello"},          # 1 + 5 = 6 tokens
        {"role": "assistant", "content": "hiii"},      # 1 + 4 = 5 tokens
    ]
    ids, mask = tokenize_conversation_with_mask(tokenizer, messages)

    assert len(ids) == 4 + 6 + 5
    assert len(mask) == len(ids)
    # First 10 tokens (system + user) unmasked, last 5 (assistant) masked.
    assert mask[:10] == [0] * 10
    assert mask[10:] == [1] * 5


def test_tokenize_multi_turn_masks_every_assistant_span():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "a"},          # 2 tokens
        {"role": "assistant", "content": "bb"},     # 3 tokens
        {"role": "user", "content": "c"},           # 2 tokens
        {"role": "assistant", "content": "dd"},     # 3 tokens
    ]
    ids, mask = tokenize_conversation_with_mask(tokenizer, messages)
    assert sum(mask) == 3 + 3  # both assistant spans, nothing else


def test_dataset_reads_manifest_and_applies_mask(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello there"},
                ]
            }
        )
        + "\n"
    )
    ds = PlannerSFTDataset(manifest, tokenizer=FakeTokenizer(), max_length=1000)
    assert len(ds) == 1
    example = ds[0]
    assert "input_ids" in example and "loss_mask" in example
    assert len(example["input_ids"]) == len(example["loss_mask"])
    assert sum(example["loss_mask"]) > 0  # something is trainable


def test_dataset_truncates_to_max_length(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "x" * 50},
                    {"role": "assistant", "content": "y" * 50},
                ]
            }
        )
        + "\n"
    )
    ds = PlannerSFTDataset(manifest, tokenizer=FakeTokenizer(), max_length=10)
    example = ds[0]
    assert len(example["input_ids"]) == 10
    assert len(example["loss_mask"]) == 10


def test_dataset_multiple_records(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    lines = [
        json.dumps({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}),
        json.dumps({"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]}),
    ]
    manifest.write_text("\n".join(lines) + "\n")
    ds = PlannerSFTDataset(manifest, tokenizer=FakeTokenizer(), max_length=1000)
    assert len(ds) == 2
