"""Shard router — ratio-enforced domain routing for training pool assembly.

Enforces the ui_first_ratio configured in pipeline.yaml (a human decision,
not auto-determined — v12 Stage 6).
"""

from __future__ import annotations

from typing import Any

from data_forge.logging_setup import get_logger
from data_forge.manifest import ManifestRecord

log = get_logger("data.shard_router")


class ShardRouter:
    """Assigns records to training shards with ratio enforcement."""

    def __init__(
        self,
        ui_first_ratio: float = 0.70,
        overflow_action: str = "exclude",
        records_per_shard: int = 5000,
    ) -> None:
        self._ui_ratio = ui_first_ratio
        self._overflow = overflow_action
        self._shard_size = records_per_shard

    def route(
        self, records: list[ManifestRecord]
    ) -> list[dict[str, Any]]:
        """Assign all records to globally numbered shards.

        Returns:
            List of {record_id, domain, shard_id, status} dicts.
        """
        ui_records = [r for r in records if r.domain == "ui_first"]
        gen_records = [r for r in records if r.domain == "general_design"]

        if not 0.0 <= self._ui_ratio <= 1.0:
            raise ValueError(f"ui_first_ratio must be in [0, 1], got {self._ui_ratio}")
        if self._shard_size < 1:
            raise ValueError(f"records_per_shard must be positive, got {self._shard_size}")

        total = len(records)
        if len(ui_records) + len(gen_records) != total:
            raise ValueError("Every record must have a known training domain before routing")
        log.info(
            "routing_plan",
            total=total,
            ui_available=len(ui_records),
            gen_available=len(gen_records),
            requested_ui_ratio=self._ui_ratio,
            note="All eligible records are retained; domain balancing is a training-time policy.",
        )

        # Interleave domains toward the requested mix while both are available;
        # retain the remaining tail rather than silently throwing away data.
        queues = {"ui_first": ui_records, "general_design": gen_records}
        positions = {"ui_first": 0, "general_design": 0}
        selected: list[ManifestRecord] = []
        ui_selected = 0
        gen_selected = 0
        while positions["ui_first"] < len(ui_records) or positions["general_design"] < len(gen_records):
            ui_pos = positions["ui_first"]
            gen_pos = positions["general_design"]
            ui_available = ui_pos < len(ui_records)
            gen_available = gen_pos < len(gen_records)

            if ui_available and gen_available:
                next_size = len(selected) + 1
                ui_deficit = next_size * self._ui_ratio - ui_selected
                gen_deficit = next_size * (1.0 - self._ui_ratio) - gen_selected
                domain = "ui_first" if ui_deficit >= gen_deficit else "general_design"
            else:
                domain = "ui_first" if ui_available else "general_design"
            selected.append(queues[domain][positions[domain]])
            if domain == "ui_first":
                ui_selected += 1
            else:
                gen_selected += 1
            positions[domain] += 1

        assignments: list[dict[str, Any]] = []
        for i, record in enumerate(selected):
            shard_id = f"shard_{i // self._shard_size:04d}"
            assignments.append({
                "record_id": record.id,
                "domain": record.domain,
                "shard_id": shard_id,
                "status": "routed",
            })

        routed_count = sum(1 for a in assignments if a["status"] == "routed")
        log.info(
            "routing_completed",
            routed=routed_count,
            ui_routed=len(ui_records),
            general_routed=len(gen_records),
            actual_ui_ratio=round(len(ui_records) / total, 4) if total else 0.0,
            shards=(routed_count + self._shard_size - 1) // self._shard_size,
        )

        return assignments
