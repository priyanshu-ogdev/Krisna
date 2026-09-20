"""Regression test: every stage's declared `requires` must be consistent
with the orchestrator's actual hardcoded execution order.

This exists because `requires` was previously pure decoration — declared
on every Stage subclass, read by nothing. The PII-scrub ordering bug
(s03_5_pii_scrub declared requires=("s03_quality",) but actually ran
before it) shipped silently because nothing checked the two against each
other. This test is that check, locked in so it can't regress again.
"""

from __future__ import annotations

from data_forge.cli import _register_all_stages  # triggers stage registration
from data_forge.orchestrator import validate_stage_ordering


def test_stage_ordering_is_internally_consistent() -> None:
    _register_all_stages()
    violations = validate_stage_ordering()
    assert violations == [], (
        "Stage requires/execution-order mismatch found:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
