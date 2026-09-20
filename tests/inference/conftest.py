from __future__ import annotations

import pytest

from krisna_inference.orchestrator.design_state import DesignState
from krisna_inference.orchestrator.model_registry import MockBackend, ModelSpec
from krisna_inference.orchestrator.store import DesignStateStore
from krisna_inference.orchestrator.swap_orchestrator import SwapOrchestrator


def fast_mock_factory(spec: ModelSpec) -> MockBackend:
    return MockBackend(spec=spec, load_latency_s=0.0, unload_latency_s=0.0)


@pytest.fixture
def orchestrator() -> SwapOrchestrator:
    return SwapOrchestrator(backend_factory=fast_mock_factory, envelope_gb=24.0)


@pytest.fixture
async def started_orchestrator(orchestrator: SwapOrchestrator) -> SwapOrchestrator:
    await orchestrator.start()
    yield orchestrator
    await orchestrator.shutdown()


@pytest.fixture
def store(tmp_path) -> DesignStateStore:
    s = DesignStateStore(db_path=tmp_path / "test_sessions.db")
    yield s
    s.close()


@pytest.fixture
def sketch_ready_state(store: DesignStateStore) -> DesignState:
    state = store.create()
    state.sketch_tokens.vq_tokens = "vq_grid_ref_abc123"
    state.touch()
    return store.save(state, expected_revision=0)


@pytest.fixture
def finalized_state(store: DesignStateStore, sketch_ready_state: DesignState) -> DesignState:
    from krisna_inference.orchestrator.design_state import SessionStage

    state = sketch_ready_state
    state.stage = SessionStage.FINALIZED
    state.finalize_output.renderer_used = "z_image_turbo"
    state.finalize_output.image_ref = "render://test/final.png"
    state.touch()
    return store.save(state, expected_revision=state.revision - 1)
