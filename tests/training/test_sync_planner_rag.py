from __future__ import annotations

import json
from pathlib import Path
import pytest

from krisna_training.data_forge_bridge.sync_planner_rag import count_critiques, sync


def test_sync_planner_rag_copies_and_counts(tmp_path):
    model_data = tmp_path / "model_data"
    rag_source = model_data / "planner_rag_corpus"
    rag_source.mkdir(parents=True)

    jsonl_file = rag_source / "uicrit_critiques.jsonl"
    entries = [
        {"record_id": "r1", "critique_output": {"summary": "good"}},
        {"record_id": "r2", "critique_output": {"summary": "needs spacing"}},
    ]
    jsonl_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    out_dir = tmp_path / "data"
    synced = sync(model_data, out_dir)

    assert synced.exists()
    assert synced.name == "uicrit_critiques.jsonl"
    assert count_critiques(synced) == 2


def test_sync_planner_rag_missing_source_raises(tmp_path):
    empty_model_data = tmp_path / "empty_model_data"
    empty_model_data.mkdir()
    with pytest.raises(FileNotFoundError, match="UICrit critiques file not found"):
        sync(empty_model_data, tmp_path / "out")
