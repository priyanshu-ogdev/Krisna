"""Dataset for sketch-tier training. Reads a JSONL manifest — one record
per image — produced by prepare_dataset.py or sync_sketch_tier.py. Each
record is one of:
  - `{"tokens": [...], "caption": "..."}` — inlined, fine for small/test
    datasets, kept for backward compatibility with any existing manifest.
  - `{"tokens_path": "tokens/0001234.npy", "caption": "..."}` — one .npy
    file per image. LEGACY format, also kept for backward compatibility,
    but no longer written by either producer — see `tokens_shard` below
    for why.
  - `{"tokens_shard": "tokens/shard_00003.npy", "tokens_index": 42,
    "caption": "..."}` — UPGRADE (docs/review/
    33_model_review_2_data_training_connection.md): the current, default
    format both producers write. At PRD §8.3's target corpus scale
    (100K-500K images), one `.npy` file per image means hundreds of
    thousands of individual small files — a real, well-known I/O
    bottleneck (per-sample open()/read()/close() syscall overhead
    dominates at this scale, worse on network-mounted storage or with
    many parallel DataLoader workers). Consolidated shards
    (training/sketch/token_shard_writer.py) are opened via
    `np.load(path, mmap_mode="r")` ONCE per shard in __init__ — safe and
    standard across forked DataLoader worker processes (each gets its
    own OS-level mapping to the same file via the page cache) — and read
    per-sample as a cheap slice against the memory-mapped array, with no
    per-sample file-open cost at all.

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
        # Opened lazily, one mmap per distinct shard file referenced by
        # any record — not eagerly for every possible shard on disk, in
        # case a manifest only actually touches a subset (e.g. a
        # held-out split carved from the same shard pool).
        self._shard_arrays: dict[str, "object"] = {}
        with self.manifest_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def _get_shard_array(self, shard_relative_path: str):
        if shard_relative_path not in self._shard_arrays:
            import numpy as np

            shard_path = self.manifest_path.parent / shard_relative_path
            # mmap_mode="r": pages faulted in lazily on access, not one
            # eager read of the whole shard — keeps memory footprint low
            # even for a large shard, while still avoiding per-sample
            # file-open overhead (the file is opened once here, not once
            # per __getitem__ call).
            self._shard_arrays[shard_relative_path] = np.load(shard_path, mmap_mode="r")
        return self._shard_arrays[shard_relative_path]

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        if "tokens" in record:
            tokens = record["tokens"]
        elif "tokens_shard" in record:
            arr = self._get_shard_array(record["tokens_shard"])
            tokens = arr[record["tokens_index"]].tolist()
        else:
            if "_cached_tokens" not in record:
                import numpy as np

                tokens_path = self.manifest_path.parent / record["tokens_path"]
                record["_cached_tokens"] = np.load(tokens_path).reshape(-1).tolist()
            tokens = record["_cached_tokens"]

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
