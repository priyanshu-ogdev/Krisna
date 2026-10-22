"""Krivex Inference: Swap Orchestrator, Design State Manager, and Verifiers.

Subsystem for executing multi-tier agentic UI generation flows across
Planner, Sketch, and Polish models with resident review within declared VRAM envelopes.
"""

from __future__ import annotations

__version__ = "0.2.0"
__version_info__ = (0, 2, 0)

from krivex_inference.orchestrator.swap_orchestrator import SwapOrchestrator
from krivex_inference.orchestrator.design_state import DesignState, SessionStage
from krivex_inference.orchestrator.store import DesignStateStore
from krivex_inference.orchestrator.model_registry import Tier, ModelSpec, ModelBackend, MockBackend
from krivex_inference.orchestrator.state_machine import ResidencyState, StateMachine
from krivex_inference.backends.factory import real_backend_factory

__all__ = [
    "__version__",
    "__version_info__",
    "SwapOrchestrator",
    "DesignState",
    "SessionStage",
    "DesignStateStore",
    "Tier",
    "ModelSpec",
    "ModelBackend",
    "MockBackend",
    "ResidencyState",
    "StateMachine",
    "real_backend_factory",
]
