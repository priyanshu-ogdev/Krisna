"""Critic SFT dataset. Each training example is exactly one (image,
constraints) -> target critique JSON pair — simpler than the Planner's
arbitrary-multi-turn case, so this uses a direct prompt-length /
total-length split rather than planner/dataset.py's general prefix-diffing
across many turns.

`build_critic_prompt()` is a DELIBERATE, comment-linked duplicate of
inference/critic_worker.py's prompt construction — not imported, because
critic_worker.py must stay a fully standalone script runnable via
`[worker_python, path]` subprocess invocation with no krisna_training (formerly krisna_orchestrator)
import at all (see that module's docstring). Keeping the wording identical
here is what makes training match inference-time prompting; if you change
the prompt in one place, change it in the other.
"""

from __future__ import annotations

import json
from pathlib import Path

from krisna_training.critic.masking import build_loss_mask_from_boundaries


def build_critic_prompt(constraints: dict) -> str:
    """MUST stay word-for-word identical to inference/critic_worker.py's
    `_run()` prompt construction."""
    return (
        "You are a senior UI/graphic design critic. Judge the ATTACHED "
        "finished design render against these constraints: "
        f"{json.dumps(constraints)}. Respond with ONLY a JSON object matching "
        "this exact shape: {\"overall_score\": <0-1 float>, \"dimensions\": "
        "{\"visual_hierarchy\": {\"score\": <0-1>, \"note\": \"<=2 sentences\"}, "
        "\"readability\": {...}, \"layout_consistency\": {...}, "
        "\"brand_alignment\": {...}}, \"suggested_edits\": [{\"region\": "
        "[x,y,w,h], \"instruction\": \"...\"}]}"
    )


def _unwrap_batch(ids) -> list[int]:
    """Some processors return input_ids as [[...]] (batch dim of 1) even
    for a single unbatched call; others return [...] directly. Normalize
    to a flat list either way."""
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        return list(ids[0])
    return list(ids)


def tokenize_critic_example_with_mask(
    processor, image, constraints: dict, critique_json: str, max_length: int = 4096
) -> dict:
    """processor: anything exposing `.apply_chat_template(messages,
    add_generation_prompt, tokenize=True, return_tensors=None,
    return_dict=True) -> {"input_ids": [...], "pixel_values": ..., ...}` —
    a real Gemma processor, or (for tests) a deterministic fake.

    Returns {"input_ids": [...], "loss_mask": [...], "pixel_values": ...}.
    """
    prompt_text = build_critic_prompt(constraints)
    user_message = {
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}],
    }

    prompt_out = processor.apply_chat_template(
        [user_message], add_generation_prompt=True, tokenize=True, return_tensors=None, return_dict=True
    )
    prompt_len = len(_unwrap_batch(prompt_out["input_ids"]))

    full_messages = [user_message, {"role": "assistant", "content": critique_json}]
    full_out = processor.apply_chat_template(
        full_messages, add_generation_prompt=False, tokenize=True, return_tensors=None, return_dict=True
    )
    full_ids = _unwrap_batch(full_out["input_ids"])

    ids = full_ids[:max_length]
    total_len = len(ids)
    clipped_prompt_len = min(prompt_len, total_len)

    mask = build_loss_mask_from_boundaries(
        roles=["user", "assistant"],
        boundaries=[0, clipped_prompt_len, total_len],
        total_len=total_len,
    )
    return {"input_ids": ids, "loss_mask": mask, "pixel_values": full_out.get("pixel_values")}


class CriticSFTDataset:
    """manifest: JSONL, one record per example —
    {"image_path": "...", "constraints": {...}, "critique": {...}}
    `critique` should match the CritiqueResult shape (§5.2) minus
    `critique_source`/`raw_model_output_ref` — those are inference-time
    provenance fields, not something the model should be trained to
    predict about itself."""

    def __init__(self, manifest_path: str | Path, processor, max_length: int = 4096) -> None:
        self.manifest_path = Path(manifest_path)
        self.processor = processor
        self.max_length = max_length
        self.records: list[dict] = []
        with self.manifest_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        record = self.records[idx]
        image_path = self.manifest_path.parent / record["image_path"]
        image = Image.open(image_path)
        critique_json = json.dumps(record["critique"])
        return tokenize_critic_example_with_mask(
            self.processor, image, record.get("constraints", {}), critique_json, self.max_length
        )
