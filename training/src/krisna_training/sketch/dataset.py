"""Dataset for sketch-tier training. Reads a JSONL manifest — one record
per image — produced by prepare_dataset.py. Each record is either
`{"tokens": [...], "caption": "..."}` (inlined, fine for smaller datasets)
or `{"tokens_path": "shard/0001.npy", "caption": "..."}` (for larger runs
where inlining every token grid into one JSONL file gets unwieldy).

Records may also carry `"source_caption"`: the original, short, human/
source-dataset caption (e.g. Screen2Words' ~6.6-word-average summaries),
distinct from `"caption"` (the dense, VLM-recaptioned description used
by default). See `caption_mix_ratio` below for why this exists.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


class SketchTokenDataset:
    def __init__(
        self, manifest_path: str | Path, grid_h: int, grid_w: int,
        caption_mix_ratio: float = 0.95,
    ) -> None:
        """caption_mix_ratio: probability of using the dense, VLM-
        recaptioned `caption` on any given access; the complement uses
        the original short `source_caption` when one exists for that
        record (silently falls back to `caption` when it doesn't, e.g.
        records with no source-dataset label at all).

        KEPT at 0.95, matching the PRD (§6.2, §8.6), all four sketch
        YAML configs, and TrainConfig's dataclass default — this
        constructor default was the one place still disagreeing (a
        stale P2a "fix" to 0.85 that was never applied anywhere else).

        The P2a rationale's underlying worry is real (train/inference
        caption-length mismatch: dense VLM captions vs. short imperative
        user prompts like "make me a dark dashboard"), but its proposed
        fix was checked against the actual source rather than trusted at
        face value. Betker et al. 2023 (DALL-E 3) explicitly swept the
        synthetic/short caption blend ratio and found that a very high
        percentage of synthetic (dense) captions maximized model
        performance — and this held even when evaluated with
        ground-truth (short, human-style) prompts, i.e. the exact P2a
        scenario. So more dense-caption exposure during training, not
        less, is what the cited paper's own sweep supports. 0.95 is the
        right ratio; cfg_dropout_prob (see make_collate_fn in train.py)
        is the mechanism that actually protects the unconditional/
        short-prompt path at inference.
        """
        self.manifest_path = Path(manifest_path)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.caption_mix_ratio = caption_mix_ratio
        self.records: list[dict] = []
        with self.manifest_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        if "tokens" in record:
            tokens = record["tokens"]
        else:
            import numpy as np

            tokens_path = self.manifest_path.parent / record["tokens_path"]
            tokens = np.load(tokens_path).reshape(-1).tolist()

        expected_len = self.grid_h * self.grid_w
        if len(tokens) != expected_len:
            raise ValueError(
                f"Record {idx} has {len(tokens)} tokens, expected "
                f"{expected_len} ({self.grid_h}x{self.grid_w}). Was this "
                "manifest built for a different grid resolution?"
            )

        dense_caption = record.get("caption", "")
        source_caption = record.get("source_caption")
        if source_caption and random.random() >= self.caption_mix_ratio:
            caption = source_caption
        else:
            caption = dense_caption
        return {"tokens": tokens, "caption": caption}
