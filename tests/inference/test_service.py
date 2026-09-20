from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Run the service against an isolated DB file in a temp cwd so tests
    # never touch (or depend on) a real krisna_sessions.db in the repo.
    monkeypatch.chdir(tmp_path)
    import krisna_inference.orchestrator.service as service_module

    importlib.reload(service_module)
    with TestClient(service_module.app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["residency_state"] == "idle_resident"


def test_create_and_get_session(client: TestClient):
    r = client.post("/session", json={"style": "minimalist", "palette": ["#111111"]})
    assert r.status_code == 200
    body = r.json()
    session_id = body["session_id"]
    assert body["constraints"]["style"] == "minimalist"

    r2 = client.get(f"/session/{session_id}")
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id


def test_get_missing_session_404(client: TestClient):
    r = client.get("/session/does-not-exist")
    assert r.status_code == 404


def test_message_finalize_critique_full_cycle(client: TestClient):
    session_id = client.post("/session", json={}).json()["session_id"]

    r = client.post(f"/session/{session_id}/message", json={"message": "make a signup form"})
    assert r.status_code == 200
    assert r.json()["stage"] == "sketching"

    r = client.post(f"/session/{session_id}/finalize", json={"quality": False})
    assert r.status_code == 200
    assert r.json()["stage"] == "finalized"
    assert r.json()["finalize_output"]["renderer_used"] == "z_image_turbo"

    r = client.post(f"/session/{session_id}/critique")
    assert r.status_code == 200
    assert r.json()["critique"]["requested"] is True


def test_finalize_without_sketch_returns_409(client: TestClient):
    session_id = client.post("/session", json={}).json()["session_id"]
    r = client.post(f"/session/{session_id}/finalize", json={})
    assert r.status_code == 409


def test_preference_pairs_stats_endpoint(client: TestClient):
    r = client.get("/preference-pairs/stats")
    assert r.status_code == 200
    body = r.json()
    assert body == {"total": 0, "verifier_stack": 0, "gemma_critique": 0, "uicrit_seed": 0}


def test_critique_with_comparison_builds_preference_pair(client: TestClient):
    session_id = client.post("/session", json={}).json()["session_id"]
    client.post(f"/session/{session_id}/message", json={"message": "make a signup form"})
    client.post(f"/session/{session_id}/finalize", json={})

    r = client.post(
        f"/session/{session_id}/critique",
        json={"compare_against_image_ref": "blob://previous.png", "compare_against_score": 0.1},
    )
    assert r.status_code == 200
    assert len(r.json()["preference_pair_refs"]) == 1

    stats = client.get("/preference-pairs/stats").json()
    assert stats["gemma_critique"] == 1


def test_preference_pairs_export_endpoint(client: TestClient, tmp_path, monkeypatch):
    session_id = client.post("/session", json={}).json()["session_id"]
    client.post(f"/session/{session_id}/message", json={"message": "make a signup form"})
    client.post(f"/session/{session_id}/finalize", json={})
    client.post(
        f"/session/{session_id}/critique",
        json={"compare_against_image_ref": "blob://previous.png", "compare_against_score": 0.1},
    )

    r = client.post("/preference-pairs/export")
    assert r.status_code == 200
    assert r.json()["written"] == 1


def test_orchestrator_status_endpoint(client: TestClient):
    r = client.get("/orchestrator/status")
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_available"] is True
    # New: ram ledger snapshot always present (no-op in full-VRAM mode,
    # since MockBackend's default REGISTRY specs all have ram_gb=0.0) —
    # a caller shouldn't have to know low_vram_mode is off first.
    assert "ram" in body
    assert body["ram"]["used_gb"] == 0.0
    assert body["low_vram_mode"] is False


def test_orchestrator_status_includes_real_hardware_probes(client: TestClient):
    """real_vram/real_ram wire probe_real_vram()/probe_real_ram() (already
    existed in vram_budget.py for diagnostics, confirmed unwired into any
    API response before this) into the status endpoint. Both are allowed
    to be None (no CUDA / no /proc on this host) — the only real
    assertion is that the endpoint never crashes and always includes the
    keys, so a client doesn't have to guess whether they're present."""
    r = client.get("/orchestrator/status")
    body = r.json()
    assert "real_vram" in body
    assert "real_ram" in body
    assert body["real_vram"] is None or {"free_gb", "total_gb"} <= body["real_vram"].keys()
    assert body["real_ram"] is None or {"free_gb", "total_gb"} <= body["real_ram"].keys()


def test_session_render_returns_404_before_finalize(client: TestClient):
    """New alongside the frontend's pixel-forming visualization — this
    route didn't exist before; finalize_output.image_ref was never
    fetchable as real bytes by any client. 404, not a crash, when no
    image exists yet."""
    r = client.post("/session", json={})
    sid = r.json()["session_id"]
    r = client.get(f"/session/{sid}/render")
    assert r.status_code == 404


def test_session_render_returns_404_for_mock_refs(client: TestClient):
    """MockBackend's own image_ref (render://mock/...) is never a real
    blob — the render route must not attempt to serve it as one (and
    must not crash trying)."""
    r = client.post("/session", json={})
    sid = r.json()["session_id"]
    client.post(f"/session/{sid}/message", json={"message": "make a signup form"})
    r = client.post(f"/session/{sid}/finalize", json={})
    assert r.status_code == 200
    r = client.get(f"/session/{sid}/render")
    assert r.status_code == 404
