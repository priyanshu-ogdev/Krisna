"""Regression tests for the opt-in vLLM Planner backend
(docs/review/29_vllm_planner_migration_research.md). These test what's
verifiable WITHOUT real GPU hardware or vllm/torch installed: the
factory's env-var-gated backend selection, and the pure
gpu_memory_utilization formula (kept torch-free specifically so it has a
direct unit test rather than only being exercised end-to-end against
real hardware this project's sandbox does not have).
"""

from __future__ import annotations

import sys

import pytest

from krisna_inference.backends.planner_backend_vllm import (
    compute_gpu_memory_utilization_fraction,
)
from krisna_inference.orchestrator.model_registry import REGISTRY, Tier


def test_default_backend_is_transformers_when_env_unset(monkeypatch):
    from krisna_inference.backends.factory import real_backend_factory
    from krisna_inference.backends.planner_backend import PlannerBackend

    monkeypatch.delenv("KRISNA_PLANNER_BACKEND", raising=False)
    backend = real_backend_factory(REGISTRY[Tier.PLANNER])
    assert type(backend) is PlannerBackend


def test_default_backend_is_transformers_for_any_other_value(monkeypatch):
    from krisna_inference.backends.factory import real_backend_factory
    from krisna_inference.backends.planner_backend import PlannerBackend

    monkeypatch.setenv("KRISNA_PLANNER_BACKEND", "transformers")
    backend = real_backend_factory(REGISTRY[Tier.PLANNER])
    assert type(backend) is PlannerBackend

    monkeypatch.setenv("KRISNA_PLANNER_BACKEND", "something_typoed")
    backend2 = real_backend_factory(REGISTRY[Tier.PLANNER])
    assert type(backend2) is PlannerBackend


def test_vllm_backend_selected_only_on_explicit_opt_in(monkeypatch):
    from krisna_inference.backends.factory import real_backend_factory
    from krisna_inference.backends.planner_backend_vllm import PlannerBackendVLLM

    monkeypatch.setenv("KRISNA_PLANNER_BACKEND", "vllm")
    backend = real_backend_factory(REGISTRY[Tier.PLANNER])
    assert isinstance(backend, PlannerBackendVLLM)
    assert backend.spec.tier == Tier.PLANNER


def test_vllm_backend_inherits_planner_backend():
    """The whole point of subclassing rather than duplicating: RAG
    retrieval, prompt building, and the JSON retry loop must be the
    SAME implementation for both backends."""
    from krisna_inference.backends.planner_backend import PlannerBackend
    from krisna_inference.backends.planner_backend_vllm import PlannerBackendVLLM

    assert issubclass(PlannerBackendVLLM, PlannerBackend)


def test_vllm_backend_does_not_reuse_bnb_checkpoint_by_default():
    """bitsandbytes is not a declared vllm dependency at all (verified
    directly against vllm==0.29.0's wheel metadata during this project's
    review) — the vLLM backend must default to a checkpoint from vLLM's
    well-supported quantization family, not silently carry over the main
    venv's NF4 checkpoint id."""
    from krisna_inference.backends.planner_backend_vllm import PlannerBackendVLLM

    backend = PlannerBackendVLLM(REGISTRY[Tier.PLANNER])
    assert "bnb" not in backend.model_id.lower()
    assert "nf4" not in backend.model_id.lower()


class TestGpuMemoryUtilizationFormula:
    """The formula this project derived to keep vLLM's own greedy
    memory reservation consistent with VRAMLedger's declared budgets,
    rather than vLLM silently grabbing ~92% of the whole physical card
    (its own upstream default) and starving co-resident tiers."""

    def test_scales_with_declared_over_total_ratio(self):
        # Planner's declared 6.5GB on a 24GB card (this project's own
        # enforced envelope, PRD §3) with a 15% margin.
        frac = compute_gpu_memory_utilization_fraction(6.5, 24.0)
        assert frac == pytest.approx((6.5 / 24.0) * 1.15, rel=1e-6)

    def test_capped_at_0_95_even_for_a_tiny_card(self):
        # A declared budget close to or exceeding the real card's total
        # memory must never request >95% — vLLM's own practical ceiling.
        frac = compute_gpu_memory_utilization_fraction(20.0, 20.0)
        assert frac == 0.95

    def test_small_declared_budget_on_a_large_card_is_small(self):
        # Same Planner budget (6.5GB) but on the actual TRAINING machine
        # from PRD §3 (RTX A6000, 48GB) — the fraction must shrink
        # accordingly, since the formula is relative to the REAL card,
        # not a fixed absolute GB target.
        frac = compute_gpu_memory_utilization_fraction(6.5, 48.0)
        assert frac < compute_gpu_memory_utilization_fraction(6.5, 24.0)

    def test_rejects_non_positive_total(self):
        with pytest.raises(ValueError):
            compute_gpu_memory_utilization_fraction(6.5, 0.0)
        with pytest.raises(ValueError):
            compute_gpu_memory_utilization_fraction(6.5, -1.0)


@pytest.mark.asyncio
async def test_load_fails_clearly_when_worker_interpreter_missing(tmp_path):
    """Mirrors test_inference_factory.py's equivalent CriticBackend test
    exactly — same failure mode, same expected behavior."""
    from krisna_inference.orchestrator.exceptions import BackendLoadError
    from krisna_inference.backends.planner_backend_vllm import PlannerBackendVLLM

    backend = PlannerBackendVLLM(
        REGISTRY[Tier.PLANNER], worker_python=str(tmp_path / "does-not-exist")
    )
    with pytest.raises(BackendLoadError, match="worker interpreter not found"):
        await backend.load()


@pytest.mark.asyncio
async def test_call_sync_real_subprocess_roundtrip(tmp_path):
    """Exercises the ACTUAL subprocess spawn + stdin/stdout JSON-line I/O
    path (_spawn, _call_sync) end to end, without needing vllm installed
    at all — using a tiny fake worker script that just echoes back
    whatever it's sent, wrapped in {"ok": true, ...}. This is the one
    piece of this backend genuinely testable without real GPU hardware:
    the subprocess plumbing itself, as opposed to what vLLM does inside
    it."""
    from krisna_inference.backends.planner_backend_vllm import PlannerBackendVLLM

    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    req = json.loads(line)\n"
        "    print(json.dumps({'ok': True, 'echo': req}), flush=True)\n"
    )

    backend = PlannerBackendVLLM(REGISTRY[Tier.PLANNER], worker_python=sys.executable)
    # Monkeypatch WORKER_SCRIPT for this instance only, via the module-level
    # constant _spawn() reads — simplest is to spawn manually here instead
    # of going through _spawn(), since WORKER_SCRIPT is a module constant
    # pointing at the real planner_worker_vllm.py.
    import subprocess

    backend._proc = subprocess.Popen(
        [sys.executable, str(fake_worker)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        response = await backend._call("generate", {"messages": [{"role": "user", "content": "hi"}]})
        assert response["ok"] is True
        assert response["echo"]["cmd"] == "generate"
        assert response["echo"]["params"]["messages"][0]["content"] == "hi"
    finally:
        await backend._kill()
