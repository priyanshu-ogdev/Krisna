"""JSONL dataset for the canonical `krisna.layout.v1` representation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NODE_TYPES = ("container", "text", "image", "button", "input", "list", "card", "icon", "divider", "other")
TYPE_TO_ID = {name: index for index, name in enumerate(NODE_TYPES)}


def _bbox(raw: dict[str, Any]) -> list[float]:
    if all(k in raw for k in ("x1", "y1", "x2", "y2")):
        return [float(raw[k]) for k in ("x1", "y1", "x2", "y2")]
    if all(k in raw for k in ("x", "y", "width", "height")):
        return [float(raw["x"]), float(raw["y"]), float(raw["x"] + raw["width"]), float(raw["y"] + raw["height"])]
    raise ValueError("bbox must use x1/y1/x2/y2 or x/y/width/height")


def canonicalize_layout_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy boxes/types without generating synthetic labels."""
    record = {**record, "version": "krisna.layout.v1"}
    nodes, seen = [], set()
    for index, node in enumerate(record.get("nodes", [])):
        node_id = str(node.get("id") or f"node_{index}")
        if node_id in seen:
            raise ValueError(f"duplicate node id: {node_id}")
        seen.add(node_id)
        node_type = str(node.get("type", "other"))
        raw_box = node["bbox"] if isinstance(node.get("bbox"), dict) else node
        nodes.append({**node, "id": node_id, "type": node_type if node_type in TYPE_TO_ID else "other", "bbox": _bbox(raw_box)})
    return {**record, "nodes": nodes, "relations": list(record.get("relations", []))}


class LayoutJSONLDataset:
    """Lazy JSONL reader; torch is imported only by the optional trainer."""

    def __init__(self, path: str | Path, max_nodes: int = 64):
        self.path = Path(path)
        self.max_nodes = max_nodes
        self._offsets = []
        with self.path.open() as handle:
            position = 0
            for line in handle:
                if line.strip():
                    self._offsets.append(position)
                position += len(line.encode())

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open() as handle:
            handle.seek(self._offsets[index])
            record = canonicalize_layout_record(json.loads(handle.readline()))
        if len(record["nodes"]) > self.max_nodes:
            raise ValueError(f"layout has {len(record['nodes'])} nodes, max_nodes={self.max_nodes}")
        return record
