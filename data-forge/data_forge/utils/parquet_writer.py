"""Append-friendly Parquet writer for the pipeline's tabular outputs.

pipeline.yaml, DATAFORGE_ARCHITECTURE.md, and the PRD all describe two
directories written in Parquet: `/preference_pairs/*.parquet` and
`/ui_critique/*.parquet`. Neither had any producing code anywhere in this
repo prior to this revision — `pandas`/`pyarrow` weren't even project
dependencies (see pyproject.toml). This module is the shared utility both
current (s10_5_critic_preference.py) and future stages should use so the
write pattern (sharding, schema stability, atomic writes) is consistent
rather than reinvented per-stage.

Design notes:
- One shard per pipeline run/chunk, not one giant growing file — matches
  the sharded convention already used everywhere else in this pipeline
  (`train-*.tar`, `shard-*.tar` in the storage layout).
- Atomic write (write to a temp path, then rename) so a crash mid-write
  never leaves a truncated/corrupt shard for a downstream training job to
  silently ingest.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from data_forge.logging_setup import get_logger

log = get_logger("utils.parquet_writer")


def write_records_parquet(
    records: list[dict[str, Any]],
    directory: Path,
    shard_prefix: str,
) -> Path | None:
    """Write a list of flat dict records to a new Parquet shard.

    Args:
        records: List of flat (non-nested-object) dicts. Nested dicts/lists
            are JSON-stringified per-field before writing, since Parquet
            columns need a consistent type and arbitrary nested Python
            structures don't reliably round-trip through pyarrow's type
            inference across shards written at different times.
        directory: Target directory (e.g. resolved_paths["ui_critique"]).
        shard_prefix: Filename prefix, e.g. "critique" -> critique_<uuid>.parquet

    Returns:
        Path to the written shard, or None if `records` was empty (no
        empty shard is written — an empty file would still show up in
        directory listings and confuse "is this dataset populated" checks).
    """
    if not records:
        return None

    import json

    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)

    # Flatten: any dict/list value is JSON-stringified so every row in the
    # shard has a stable, single-typed column set. Callers that need the
    # structured value back should json.loads() it on read.
    flat_records = []
    for rec in records:
        flat = {}
        for k, v in rec.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = v
        flat_records.append(flat)

    df = pd.DataFrame(flat_records)

    shard_name = f"{shard_prefix}_{uuid.uuid4().hex[:12]}.parquet"
    final_path = directory / shard_name
    tmp_path = directory / f".{shard_name}.tmp"

    df.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
    os.replace(tmp_path, final_path)  # atomic on POSIX and Windows (same volume)

    log.info(
        "parquet_shard_written",
        path=str(final_path),
        rows=len(flat_records),
        bytes=final_path.stat().st_size,
    )
    return final_path
