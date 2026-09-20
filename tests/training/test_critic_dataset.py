from __future__ import annotations

import json

import pytest
from PIL import Image

from krisna_training.critic.dataset import (
    CriticSFTDataset,
    build_critic_prompt,
    tokenize_critic_example_with_mask,
)


class FakeProcessor:
    """Deterministic stand-in for a real Gemma processor. Each call
    tokenizes deterministically from message content length, so boundary
    computation (prompt_len vs total_len) is exactly predictable — same
    testing strategy as the Planner's FakeTokenizer, adapted for
    multimodal content lists."""

    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_tensors, return_dict):
        ids = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                ids.extend(range(len(content)))
            else:
                for part in content:
                    if part["type"] == "text":
                        ids.extend(range(len(part["text"])))
                    elif part["type"] == "image":
                        ids.extend([9999] * 10)  # fixed-size image placeholder tokens
        if add_generation_prompt:
            ids.append(1)
        return {"input_ids": [ids], "pixel_values": [[0.0]]}


def test_build_critic_prompt_embeds_constraints():
    prompt = build_critic_prompt({"style": "minimalist"})
    assert "minimalist" in prompt
    assert "overall_score" in prompt


def test_build_critic_prompt_matches_inference_worker_exactly():
    """Regression/drift test: training/critic/dataset.py's
    build_critic_prompt() must stay word-for-word identical to
    inference/critic_worker.py's _build_prompt() — they can't share code
    (critic_worker.py must stay standalone, no krisna_training (formerly krisna_orchestrator) import,
    to run via subprocess), so this test is what actually catches drift
    instead of relying purely on the comments saying to keep them in sync."""
    from krisna_inference.backends.critic_worker import _build_prompt

    for constraints in [{}, {"style": "bold"}, {"style": "minimalist", "palette": ["#000"]}]:
        assert build_critic_prompt(constraints) == _build_prompt(constraints)


def test_tokenize_masks_only_assistant_span():
    processor = FakeProcessor()
    image = Image.new("RGB", (4, 4))
    constraints = {"style": "x"}
    critique_json = json.dumps({"overall_score": 0.8})

    result = tokenize_critic_example_with_mask(processor, image, constraints, critique_json)
    assert len(result["input_ids"]) == len(result["loss_mask"])
    assert sum(result["loss_mask"]) > 0
    # Everything before the assistant span is unmasked.
    first_masked = result["loss_mask"].index(1)
    assert all(m == 0 for m in result["loss_mask"][:first_masked])


def test_tokenize_respects_max_length():
    processor = FakeProcessor()
    image = Image.new("RGB", (4, 4))
    critique_json = json.dumps({"overall_score": 0.5, "note": "x" * 200})

    result = tokenize_critic_example_with_mask(processor, image, {}, critique_json, max_length=20)
    assert len(result["input_ids"]) == 20
    assert len(result["loss_mask"]) == 20


def test_tokenize_includes_pixel_values():
    processor = FakeProcessor()
    image = Image.new("RGB", (4, 4))
    result = tokenize_critic_example_with_mask(processor, image, {}, "{}")
    assert result["pixel_values"] is not None


def test_dataset_reads_manifest_and_loads_image(tmp_path):
    img_path = tmp_path / "render.png"
    Image.new("RGB", (4, 4), color="blue").save(img_path)

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_path": "render.png",
                "constraints": {"style": "bold"},
                "critique": {"overall_score": 0.75, "dimensions": {}, "suggested_edits": []},
            }
        )
        + "\n"
    )
    ds = CriticSFTDataset(manifest, processor=FakeProcessor())
    assert len(ds) == 1
    example = ds[0]
    assert "input_ids" in example and "loss_mask" in example
    assert sum(example["loss_mask"]) > 0
