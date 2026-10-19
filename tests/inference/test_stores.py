from __future__ import annotations

import pytest
from PIL import Image

from krisna_inference.backends.common import BlobStore
from krisna_inference.orchestrator.design_state import DesignState, SessionStage
from krisna_inference.orchestrator.exceptions import SessionNotFoundError, StaleDesignStateError
from krisna_inference.orchestrator.store import DesignStateStore


# ---------------------------------------------------------------------------
# BlobStore Tests
# ---------------------------------------------------------------------------

def test_blob_store_save_and_load_roundtrip(tmp_path):
    store = BlobStore(root=tmp_path / "blobs")
    img = Image.new("RGB", (4, 4), color="red")
    ref = store.save_image(img, prefix="test")

    assert ref.startswith("blob://test_")
    loaded = store.load_image(ref)
    assert loaded.size == (4, 4)


def test_blob_store_load_missing_blob_raises(tmp_path):
    store = BlobStore(root=tmp_path / "blobs")
    with pytest.raises(FileNotFoundError):
        store.load_image("blob://does-not-exist.png")


def test_blob_store_load_non_blob_ref_raises(tmp_path):
    store = BlobStore(root=tmp_path / "blobs")
    with pytest.raises(ValueError):
        store.load_image("not-a-blob-ref")


def test_blob_store_path_for(tmp_path):
    store = BlobStore(root=tmp_path / "blobs")
    p = store.path_for("blob://foo.png")
    assert p == (tmp_path / "blobs" / "foo.png")


# ---------------------------------------------------------------------------
# DesignStateStore Tests
# ---------------------------------------------------------------------------

def test_design_state_store_create_and_get_roundtrip(store: DesignStateStore):
    state = store.create()
    fetched = store.get(state.session_id)
    assert fetched.session_id == state.session_id
    assert fetched.revision == 0


def test_design_state_store_get_missing_session_raises(store: DesignStateStore):
    with pytest.raises(SessionNotFoundError):
        store.get("does-not-exist")


def test_design_state_store_save_bumps_and_persists(store: DesignStateStore):
    state = store.create()
    state.append_turn("user", "hello")  # bumps revision to 1
    saved = store.save(state, expected_revision=0)
    assert saved.revision == 1

    fetched = store.get(state.session_id)
    assert fetched.conversation_history[0].content == "hello"
    assert fetched.revision == 1


def test_design_state_store_stale_write_rejected(store: DesignStateStore):
    state = store.create()
    state.append_turn("user", "first writer")
    store.save(state, expected_revision=0)

    # A second writer who read the OLD revision (0) tries to save.
    stale_copy = DesignState(**state.model_dump())
    stale_copy.session_id = state.session_id
    stale_copy.append_turn("user", "stale writer")
    with pytest.raises(StaleDesignStateError):
        store.save(stale_copy, expected_revision=0)


def test_design_state_store_list_sessions_by_stage(store: DesignStateStore):
    a = store.create()
    b = store.create()
    b.stage = SessionStage.FINALIZED
    b.touch()
    store.save(b, expected_revision=0)

    finalized = store.list_sessions(stage="finalized")
    assert finalized == [b.session_id]
    all_sessions = set(store.list_sessions())
    assert all_sessions == {a.session_id, b.session_id}
