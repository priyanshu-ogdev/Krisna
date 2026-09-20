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
        caption_mix_ratio: float = 0.85,
    ) -> None:
        """caption_mix_ratio: probability of using the dense, VLM-
        recaptioned `caption` on any given access; the complement uses
        the original short `source_caption` when one exists for that
        record (silently falls back to `caption` when it doesn't, e.g.
        records with no source-dataset label at all).

        Default CHANGED from 0.95 to 0.85 (semantic audit, P2a fix):
        Betker et al. 2023's 95/5 split was validated for photo-domain
        models where human captions are long and descriptive. Krisna's
        Sketch tier receives raw user intent messages at inference time
        ("make me a dark dashboard" — 5-8 words, imperative framing)
        while training on dense VLM captions of finished screenshots
        ("A dark settings screen with toggle switches" — descriptive,
        pixel-level). The 15% short-caption share better regularizes
        the model against this training/inference distribution gap
        without fully abandoning the density that makes VLM captions
        useful for structural grounding. See
        docs/review/09_synthetic_data_audit.md and the semantic audit
        docs for context.
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
