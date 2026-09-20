"""Data-Forge: Zero-touch automated data pipeline for the Krisna project (v14 spec — no-RLHF revision)."""

from __future__ import annotations

__version__ = "0.14.0"

from data_forge.config import PipelineConfig, load_config
from data_forge.logging_setup import get_logger, setup_logging
from data_forge.manifest import Manifest
from data_forge.orchestrator import (
    ChunkResult,
    Orchestrator,
    get_registered_stages,
    register_stage,
    validate_stage_ordering,
)
from data_forge.stages import (
    STAGE_MODULES,
    Stage,
    StageResult,
    register_all_stages,
)

__all__ = [
    "__version__",
    "Manifest",
    "PipelineConfig",
    "load_config",
    "Orchestrator",
    "ChunkResult",
    "validate_stage_ordering",
    "get_registered_stages",
    "register_stage",
    "Stage",
    "StageResult",
    "STAGE_MODULES",
    "register_all_stages",
    "setup_logging",
    "get_logger",
]
