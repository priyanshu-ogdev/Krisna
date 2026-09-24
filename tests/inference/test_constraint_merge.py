"""Regression tests for the mid-turn constraint-staleness bug found on
inference-layer review: swap_orchestrator.run_conversational_turn used to
call Sketch with the SAME `constraints` dict the caller passed in before
the Planner ran, so a Planner-produced constraint_updates delta (e.g.
"switch to dark mode") wasn't reflected in that same turn's sketch —
only the NEXT turn's, since flows.py only applied it to persisted
DesignState after run_conversational_turn had already returned.

apply_constraint_updates (constraint_merge.py) is the single shared fix
for this: both swap_orchestrator.py (in-flight kwargs for this turn) and
flows.py (persisted DesignState) now call the SAME function so they
can't drift out of sync with each other again.
"""

from __future__ import annotations

from krisna_inference.orchestrator.constraint_merge import apply_constraint_updates


def test_style_update_is_applied():
    constraints = {"style": "light minimal", "palette": [], "layout_hints": "", "locked_regions": []}
    merged = apply_constraint_updates(constraints, {"style": "dark mode"})
    assert merged["style"] == "dark mode"


def test_original_dict_is_not_mutated():
    constraints = {"style": "light minimal", "palette": [], "layout_hints": "", "locked_regions": []}
    apply_constraint_updates(constraints, {"style": "dark mode"})
    assert constraints["style"] == "light minimal"


def test_unrecognized_or_missing_keys_leave_field_unchanged():
    constraints = {"style": "light minimal", "palette": ["#fff"], "layout_hints": "sidebar left", "locked_regions": []}
    merged = apply_constraint_updates(constraints, {"style": None, "some_other_field": "x"})
    assert merged["style"] == "light minimal"
    assert merged["palette"] == ["#fff"]
    assert merged["layout_hints"] == "sidebar left"


def test_non_dict_constraint_updates_is_a_no_op():
    constraints = {"style": "light minimal", "palette": [], "layout_hints": "", "locked_regions": []}
    merged = apply_constraint_updates(constraints, None)
    assert merged == constraints
    merged2 = apply_constraint_updates(constraints, "not a dict")
    assert merged2 == constraints


def test_palette_only_updates_when_list():
    constraints = {"style": "x", "palette": ["#000"], "layout_hints": "", "locked_regions": []}
    merged = apply_constraint_updates(constraints, {"palette": "not-a-list"})
    assert merged["palette"] == ["#000"]  # unchanged — invalid type ignored
    merged2 = apply_constraint_updates(constraints, {"palette": ["#111", "#222"]})
    assert merged2["palette"] == ["#111", "#222"]


def test_locked_regions_accepts_plain_dicts():
    constraints = {"style": "x", "palette": [], "layout_hints": "", "locked_regions": []}
    update = {"locked_regions": [{"bbox": [0, 0, 1, 1], "reason": "logo"}]}
    merged = apply_constraint_updates(constraints, update)
    assert merged["locked_regions"] == [{"bbox": [0, 0, 1, 1], "reason": "logo"}]


def test_locked_regions_malformed_entries_are_dropped():
    constraints = {"style": "x", "palette": [], "layout_hints": "", "locked_regions": [{"bbox": [0, 0, 1, 1], "reason": "old"}]}
    update = {"locked_regions": [{"missing_bbox": True}]}
    merged = apply_constraint_updates(constraints, update)
    # No valid entries in the update -> original locked_regions preserved,
    # not silently wiped to an empty list.
    assert merged["locked_regions"] == [{"bbox": [0, 0, 1, 1], "reason": "old"}]
