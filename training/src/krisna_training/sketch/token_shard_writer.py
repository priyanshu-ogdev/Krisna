"""Consolidated, memory-mappable token shard writer.

UPGRADE (docs/review/33_model_review_2_data_training_connection.md):
both `sync_sketch_tier.py` and `prepare_dataset.py` used to write ONE
`.npy` file PER IMAGE (`tokens/{i:07d}.npy`, ~1-4KB each depending on
grid size). At PRD §8.3's target corpus scale (100K-500K images), that's
hundreds of thousands of individual files — every epoch, every worker
process independently pays a real `open()`/`read()`/`close()` syscall
per sample, a well-known I/O bottleneck at this scale (small-file
overhead dominates over actual bytes transferred, and gets worse on
network-mounted storage or with many parallel DataLoader workers hitting
the filesystem simultaneously).

Fixed by writing ONE consolidated array per shard
(`tokens/shard_00000.npy`, shape `(shard_size, grid_h*grid_w)`, dtype
int32) instead of one file per image. `SketchTokenDataset` opens each
shard via `np.load(path, mmap_mode="r")` ONCE in `__init__` — this is
safe and standard practice across forked DataLoader worker processes
(each worker gets its own OS-level mapping to the same underlying file
via the page cache, no explicit copying or IPC needed) — and reads
individual samples as cheap slice operations against the memory-mapped
array, with no per-sample file-open overhead at all.

Sharded (not one array for the WHOLE corpus) so a single shard write
failure doesn't require re-tokenizing everything, and so shard files stay
a reasonable, streamable size regardless of total corpus size —
`DEFAULT_SHARD_SIZE` chosen so a shard at the 32x32 (512px, Stage 2) grid
size stays well under 100MB (10,000 * 1024 tokens * 4 bytes ≈ 41MB),
comfortable for filesystems and cloud storage alike.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_SHARD_SIZE = 10_000


class TokenShardWriter:
    """Buffers token grids in memory up to `shard_size`, then flushes them
    as one consolidated `.npy` array file. Call `add(tokens)` for each
    image in corpus order, then `close()` once at the end to flush any
    remaining partial shard — mirrors a standard buffered-writer
    lifecycle (open → add* → close), not a context manager, so it slots
    into both callers' existing per-image `for` loops with minimal
    disruption.
    """

    def __init__(self, tokens_dir: str | Path, expected_token_len: int, shard_size: int = DEFAULT_SHARD_SIZE):
        self.tokens_dir = Path(tokens_dir)
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        self.expected_token_len = expected_token_len
        self.shard_size = shard_size
        self._buffer: list[list[int]] = []
        self._shard_index = 0
        self._total_written = 0

    @property
    def total_written(self) -> int:
        """Running count of tokens added so far (flushed or still
        buffered) — exposed for caller progress logging, not used
        internally for shard/index bookkeeping (that's derived purely
        from `_shard_index`/buffer length, see `add()`)."""
        return self._total_written

    def add(self, tokens: list[int]) -> tuple[str, int]:
        """Returns (shard_relative_path, index_within_shard) for THIS
        token grid — the exact pair a manifest record needs to look it
        back up later (`tokens_shard`, `tokens_index`)."""
        if len(tokens) != self.expected_token_len:
            raise ValueError(
                f"Token grid has {len(tokens)} tokens, expected {self.expected_token_len} "
                "— mismatched grid resolution passed to TokenShardWriter."
            )
        self._buffer.append(tokens)
        idx_in_shard = len(self._buffer) - 1
        shard_path = self._shard_relative_path(self._shard_index)
        self._total_written += 1
        if len(self._buffer) >= self.shard_size:
            self._flush()
        return shard_path, idx_in_shard

    def _shard_relative_path(self, shard_index: int) -> str:
        return f"tokens/shard_{shard_index:05d}.npy"

    def _flush(self) -> None:
        if not self._buffer:
            return
        arr = np.array(self._buffer, dtype="int32")
        out_path = self.tokens_dir.parent / self._shard_relative_path(self._shard_index)
        np.save(out_path, arr)
        self._buffer = []
        self._shard_index += 1

    def close(self) -> None:
        """Flush any remaining partial shard. Must be called once after
        the last add() — a forgotten close() would silently drop the
        final (possibly full) shard's worth of images, so callers must
        not skip this."""
        self._flush()
