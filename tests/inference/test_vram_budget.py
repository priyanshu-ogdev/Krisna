from __future__ import annotations

import pytest

from krisna_inference.orchestrator.exceptions import VRAMBudgetExceededError
from krisna_inference.orchestrator.model_registry import Tier
from krisna_inference.orchestrator.vram_budget import VRAMLedger


def test_admits_within_envelope():
    ledger = VRAMLedger(envelope_gb=24.0)
    ledger.admit(Tier.PLANNER)   # 6.5
    ledger.admit(Tier.SKETCH)    # 0.2
    assert ledger.used_gb == pytest.approx(6.7)
    assert ledger.would_fit(Tier.POLISH_DEFAULT)  # 12.0 more = 18.7, fits


def test_rejects_over_envelope():
    ledger = VRAMLedger(envelope_gb=20.0)
    ledger.admit(Tier.PLANNER)
    ledger.admit(Tier.SKETCH)
    ledger.admit(Tier.POLISH_DEFAULT)
    assert ledger.used_gb == pytest.approx(18.7)
    assert not ledger.would_fit(Tier.POLISH_QUALITY)  # +12.0 = 30.7, doesn't fit
    with pytest.raises(VRAMBudgetExceededError):
        ledger.admit(Tier.POLISH_QUALITY)


def test_release_frees_budget():
    ledger = VRAMLedger(envelope_gb=18.6)
    ledger.admit(Tier.PLANNER)
    ledger.admit(Tier.SKETCH)
    assert not ledger.would_fit(Tier.POLISH_DEFAULT)  # 6.7 + 12.0 = 18.7 > 18.6
    ledger.release(Tier.SKETCH)
    assert ledger.would_fit(Tier.POLISH_DEFAULT)  # 6.5 + 12.0 = 18.5 <= 18.6


def test_double_admit_is_idempotent():
    ledger = VRAMLedger(envelope_gb=24.0)
    ledger.admit(Tier.PLANNER)
    used_after_first = ledger.used_gb
    ledger.admit(Tier.PLANNER)
    assert ledger.used_gb == used_after_first


def test_snapshot_shape():
    ledger = VRAMLedger(envelope_gb=24.0)
    ledger.admit(Tier.PLANNER)
    snap = ledger.snapshot()
    assert snap["resident"] == ["planner"]
    assert snap["used_gb"] == pytest.approx(6.5)
