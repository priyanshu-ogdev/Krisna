from __future__ import annotations

import pytest

from krisna_inference.orchestrator.exceptions import VRAMBudgetExceededError
from krisna_inference.orchestrator.model_registry import Tier
from krisna_inference.orchestrator.vram_budget import VRAMLedger


def test_admits_within_envelope():
    ledger = VRAMLedger(envelope_gb=24.0)
    ledger.admit(Tier.PLANNER)   # 6.5
    ledger.admit(Tier.SKETCH)    # 3.0
    assert ledger.used_gb == pytest.approx(9.5)
    assert ledger.would_fit(Tier.POLISH_DEFAULT)  # 8.0 more = 17.5, fits


def test_rejects_over_envelope():
    ledger = VRAMLedger(envelope_gb=24.0)
    ledger.admit(Tier.PLANNER)
    ledger.admit(Tier.SKETCH)
    ledger.admit(Tier.POLISH_DEFAULT)
    assert ledger.used_gb == pytest.approx(17.5)
    assert not ledger.would_fit(Tier.POLISH_QUALITY)  # +16.0 = 33.5, doesn't fit
    with pytest.raises(VRAMBudgetExceededError):
        ledger.admit(Tier.POLISH_QUALITY)


def test_release_frees_budget():
    ledger = VRAMLedger(envelope_gb=15.0)
    ledger.admit(Tier.PLANNER)
    ledger.admit(Tier.SKETCH)
    assert not ledger.would_fit(Tier.POLISH_DEFAULT)  # 9.5 + 8.0 = 17.5 > 15
    ledger.release(Tier.SKETCH)
    assert ledger.would_fit(Tier.POLISH_DEFAULT)  # 6.5 + 8.0 = 14.5 <= 15


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
