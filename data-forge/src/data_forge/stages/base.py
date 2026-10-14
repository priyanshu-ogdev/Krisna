"""Abstract base class for pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.manifest import Manifest


@dataclass
class StageResult:
    """Result from running a pipeline stage."""

    stage_name: str
    records_processed: int = 0
    records_failed: int = 0
    records_excluded: int = 0
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.records_failed == 0


class MissingVLMEngineError(RuntimeError):
    """A strict VLM stage was invoked without its required engine."""


class Stage(ABC):
    """Abstract base for all pipeline stages.

    Subclasses must implement `run()` and set `name` and `requires`.
    """

    name: str = ""
    requires: ClassVar[tuple[str, ...]] = ()

    def require_vlm_engine(self, config: PipelineConfig, engine: Any | None) -> None:
        """Fail before any fallback or mutation when strict inference is requested.

        Only VLM stages call this guard; deterministic stages remain engine-free.
        Legacy stage-specific missing-engine behavior is preserved when opted out.
        """
        if config.strict_vlm_no_fallback and engine is None:
            raise MissingVLMEngineError(
                f"{self.name} requires an active VLM engine "
                "(strict_vlm_no_fallback=True); skipping inference or using "
                "source/empty/freeform fallback output is forbidden"
            )

    @abstractmethod
    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        """Execute the stage on the given record IDs.

        Args:
            manifest: The pipeline manifest database.
            config: Pipeline configuration.
            record_ids: Record IDs to process (empty for global stages).
            engine: GPU model engine (if this stage needs inference).

        Returns:
            StageResult with processing counts.
        """
        ...
