from __future__ import annotations

import pytest

from krisna_inference.orchestrator.design_state import DesignState
from krisna_inference.orchestrator.exceptions import SessionNotFoundError, StaleDesignStateError
from krisna_inference.orchestrator.store import DesignStateStore


def test_create_and_get_roundtrip(store: DesignStateStore):
    state = store.create()
    fetched = store.get(state.session_id)
    assert fetched.session_id == state.session_id
    assert fetched.revision == 0


def test_get_missing_session_raises(store: DesignStateStore):
    with pytest.raises(SessionNotFoundError):
        store.get("does-not-exist")


def test_save_bumps_and_persists(store: DesignStateStore):
    state = store.create()
    state.append_turn("user", "hello")  # bumps revision to 1
    saved = store.save(state, expected_revision=0)
    assert saved.revision == 1

    fetched = store.get(state.session_id)
    assert fetched.conversation_history[0].content == "hello"
    assert fetched.revision == 1


def test_stale_write_rejected(store: DesignStateStore):
    state = store.create()
    state.append_turn("user", "first writer")
    store.save(state, expected_revision=0)

    # A second writer who read the OLD revision (0) tries to save.
    stale_copy = DesignState(**state.model_dump())
    stale_copy.session_id = state.session_id
    stale_copy.append_turn("user", "stale writer")
    with pytest.raises(StaleDesignStateError):
        store.save(stale_copy, expected_revision=0)


def test_list_sessions_by_stage(store: DesignStateStore):
    from krisna_inference.orchestrator.design_state import SessionStage

    a = store.create()
    b = store.create()
    b.stage = SessionStage.FINALIZED
    b.touch()
    store.save(b, expected_revision=0)

    finalized = store.list_sessions(stage="finalized")
    assert finalized == [b.session_id]
    all_sessions = set(store.list_sessions())
    assert all_sessions == {a.session_id, b.session_id}


def test_delete_session(store: DesignStateStore):
    state = store.create()
    assert store.get(state.session_id) is not None
    assert store.delete(state.session_id) is True
    with pytest.raises(SessionNotFoundError):
        store.get(state.session_id)
    assert store.delete(state.session_id) is False
