"""Pipeline metrics collection and summary statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageMetrics:
    """Metrics for a single stage execution."""

    stage_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    records_in: int = 0
    records_out: int = 0
    records_excluded: int = 0
    records_failed: int = 0
    inference_calls: int = 0
    bytes_written: int = 0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def throughput_rps(self) -> float:
        d = self.duration_seconds
        return self.records_in / d if d > 0 else 0.0


@dataclass
class PipelineMetrics:
    """Aggregated metrics across all stages."""

    start_time: float = 0.0
    end_time: float = 0.0
    stages: list[StageMetrics] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return self.end_time - self.start_time

    def add_stage(self, metrics: StageMetrics) -> None:
        self.stages.append(metrics)

    def summary(self) -> dict[str, Any]:
        return {
            "total_duration_s": round(self.total_duration, 2),
            "stages": [
                {
                    "name": s.stage_name,
                    "duration_s": round(s.duration_seconds, 2),
                    "records_in": s.records_in,
                    "records_out": s.records_out,
                    "excluded": s.records_excluded,
                    "failed": s.records_failed,
                    "throughput_rps": round(s.throughput_rps, 1),
                }
                for s in self.stages
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.summary(), indent=2), encoding="utf-8"
        )
