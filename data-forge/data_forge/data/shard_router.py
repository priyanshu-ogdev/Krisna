"""Shard router — ratio-enforced domain routing for training pool assembly.

Enforces the ui_first_ratio configured in pipeline.yaml (a human decision,
not auto-determined — v12 Stage 6).
"""

from __future__ import annotations

import math
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
        """Assign domains and shards to records, enforcing ratio.

        Returns:
            List of {record_id, domain, shard_id, status} dicts.
        """
        ui_records = [r for r in records if r.domain == "ui_first"]
        gen_records = [r for r in records if r.domain == "general_design"]

        total = len(records)
        target_ui = math.floor(total * self._ui_ratio)
        target_gen = total - target_ui

        log.info(
            "routing_plan",
            total=total,
            ui_available=len(ui_records),
            gen_available=len(gen_records),
            target_ui=target_ui,
            target_gen=target_gen,
        )

        # Cap each domain to its target
        selected_ui = ui_records[:target_ui]
        selected_gen = gen_records[:target_gen]
        overflow_ui = ui_records[target_ui:]
        overflow_gen = gen_records[target_gen:]

        # Assign shards
        assignments: list[dict[str, Any]] = []

        all_selected = selected_ui + selected_gen
        shard_idx = 0
        for i, record in enumerate(all_selected):
            if i > 0 and i % self._shard_size == 0:
                shard_idx += 1
            shard_id = f"shard_{shard_idx:04d}"
            assignments.append({
                "record_id": record.id,
                "domain": record.domain,
                "shard_id": shard_id,
                "status": "routed",
            })

        # Handle overflow
        for record in overflow_ui + overflow_gen:
            if self._overflow == "exclude":
                assignments.append({
                    "record_id": record.id,
                    "domain": record.domain,
                    "shard_id": None,
                    "status": "overflow_excluded",
                })
            else:
                # keep_unrouted — leave in pipeline but don't assign shard
                assignments.append({
                    "record_id": record.id,
                    "domain": record.domain,
                    "shard_id": None,
                    "status": "routed",
                })

        routed_count = sum(1 for a in assignments if a["status"] == "routed")
        overflow_count = sum(1 for a in assignments if a["status"] == "overflow_excluded")
        log.info(
            "routing_completed",
            routed=routed_count,
            overflow_excluded=overflow_count,
            shards=shard_idx + 1,
        )

        return assignments
