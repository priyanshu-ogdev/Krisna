"""Chat SFT dataset for planner training. Reads a JSONL manifest, one
conversation per line: `{"messages": [{"role": "system"|"user"|
"assistant", "content": "..."}, ...]}`.

Turn boundaries (needed for assistant-only loss masking — see masking.py)
are found via the standard template-agnostic trick: tokenize the full
conversation once, then tokenize each successive PREFIX of the message
list and take the length after each one. This works regardless of the
specific chat template's exact special-token wrapping, which is why it's
used here rather than hand-parsing Qwen3.5's template format (a moving
target this codebase has already had to correct once — see
inference/planner_backend.py's notes on needing transformers built from
git main).
"""

from __future__ import annotations

import json
from pathlib import Path

from krisna_training.planner.masking import build_loss_mask_from_boundaries


class PlannerSFTDataset:
    def __init__(self, manifest_path: str | Path, tokenizer, max_length: int = 4096) -> None:
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
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
        messages = self.records[idx]["messages"]
        input_ids, loss_mask = tokenize_conversation_with_mask(self.tokenizer, messages)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            loss_mask = loss_mask[: self.max_length]

        return {"input_ids": input_ids, "loss_mask": loss_mask}


def tokenize_conversation_with_mask(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    boundaries = [0]
    for i in range(len(messages)):
        prefix_ids = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True, add_generation_prompt=False
        )
        boundaries.append(len(prefix_ids))
    # A chat template can append trailing tokens after the last message's
    # content (e.g. a closing turn marker) that aren't part of any single
    # prefix tokenization — make sure the final boundary covers the whole
    # sequence rather than leaving a gap that build_loss_mask_from_boundaries
    # would silently zero out.
    boundaries[-1] = max(boundaries[-1], len(full_ids))

    roles = [m["role"] for m in messages]
    mask = build_loss_mask_from_boundaries(roles, boundaries, len(full_ids))
    return full_ids, mask
