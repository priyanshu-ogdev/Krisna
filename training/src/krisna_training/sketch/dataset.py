"""Dataset for sketch-tier training. Reads a JSONL manifest — one record
per image — produced by prepare_dataset.py. Each record is either
`{"tokens": [...], "caption": "..."}` (inlined, fine for smaller datasets)
or `{"tokens_path": "shard/0001.npy", "caption": "..."}` (for larger runs
where inlining every token grid into one JSONL file gets unwieldy).
"""

from __future__ import annotations

import json
from pathlib import Path


class SketchTokenDataset:
    def __init__(self, manifest_path: str | Path, grid_h: int, grid_w: int) -> None:
        self.manifest_path = Path(manifest_path)
        self.grid_h = grid_h
        self.grid_w = grid_w
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
        return {"tokens": tokens, "caption": record.get("caption", "")}
