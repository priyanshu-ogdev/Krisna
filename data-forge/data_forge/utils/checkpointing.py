"""Pipeline checkpoint/resume — per-record and per-stage state persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path

from data_forge.logging_setup import get_logger

log = get_logger("utils.checkpointing")


class CheckpointManager:
    """Manages stage completion markers for pipeline resume."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self._dir = checkpoint_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def mark_complete(
        self, stage: str, chunk_id: str, metadata: dict | None = None
    ) -> None:
        path = self._dir / f"{stage}_{chunk_id}.done"
        data = {
            "stage": stage,
            "chunk_id": chunk_id,
            "completed_at": time.time(),
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        log.debug("checkpoint_marked", stage=stage, chunk=chunk_id)

    def is_complete(self, stage: str, chunk_id: str) -> bool:
        path = self._dir / f"{stage}_{chunk_id}.done"
        return path.exists()

    def clear_stage(self, stage: str) -> int:
        """Clear all checkpoints for a stage. Returns count cleared."""
        count = 0
        for f in self._dir.glob(f"{stage}_*.done"):
            f.unlink()
            count += 1
        log.info("checkpoints_cleared", stage=stage, count=count)
        return count

    def clear_all(self) -> int:
        """Clear all checkpoints. Returns count cleared."""
        count = 0
        for f in self._dir.glob("*.done"):
            f.unlink()
            count += 1
        log.info("all_checkpoints_cleared", count=count)
        return count

    def list_completed(self) -> list[dict]:
        """List all completed checkpoints."""
        results = []
        for f in sorted(self._dir.glob("*.done")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                results.append(data)
            except json.JSONDecodeError:
                results.append({"file": f.name, "error": "invalid json"})
        return results
