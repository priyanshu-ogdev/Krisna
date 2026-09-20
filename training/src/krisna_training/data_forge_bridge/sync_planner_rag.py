"""Synchronizes Data-Forge exported planner_rag_corpus/uicrit_critiques.jsonl

into the training / inference directory.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger("krisna_training.data_forge_bridge.sync_planner_rag")


def sync(data_forge_model_data_dir: str | Path, output_dir: str | Path) -> Path:
    """Copies uicrit_critiques.jsonl from model_data/planner_rag_corpus into output_dir.

    Returns the path to the synchronized JSONL file.
    """
    model_data_dir = Path(data_forge_model_data_dir)
    source_file = model_data_dir / "planner_rag_corpus" / "uicrit_critiques.jsonl"
    if not source_file.exists():
        # Fall back to direct path if caller pointed directly at planner_rag_corpus
        alt_source = model_data_dir / "uicrit_critiques.jsonl"
        if alt_source.exists():
            source_file = alt_source
        else:
            raise FileNotFoundError(
                f"UICrit critiques file not found at {source_file}. "
                "Ensure data-forge stage s12_model_data_export has been run."
            )

    out_path = Path(output_dir)
    if out_path.name != "planner_rag_corpus":
        target_dir = out_path / "planner_rag_corpus"
    else:
        target_dir = out_path

    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "uicrit_critiques.jsonl"

    shutil.copy2(source_file, target_file)

    count = count_critiques(target_file)
    log.info("sync_planner_rag_complete", extra={"records": count, "dest": str(target_file)})
    return target_file


def count_critiques(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Sync Data-Forge RAG corpus to destination directory.")
    parser.add_argument("--source", required=True, help="Path to data-forge model_data directory")
    parser.add_argument("--dest", required=True, help="Target destination directory")
    args = parser.parse_args()
    sync(args.source, args.dest)


if __name__ == "__main__":
    main()
