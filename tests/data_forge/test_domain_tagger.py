"""Tests for data_forge/data/domain_tagger.py.

Covers the gap traced during the v10 PRD completeness review: pd12m and
cc12m (the real, configured general_visual dataset keys — see
configs/datasets.yaml) were absent from both `_UI_FIRST_SOURCES` and
`_GENERAL_DESIGN_SOURCES`, so they were only correctly tagged
`general_design` via the structural heuristic's final default branch —
fragile, since a false-positive structural detection could silently
misroute a stock-photo record into `ui_first`, contaminating the sketch
tier's domain-gated training data.
"""

from __future__ import annotations

from data_forge.data.domain_tagger import tag_domain
from data_forge.manifest import ManifestRecord


def _record(source_dataset: str, structure_output=None) -> ManifestRecord:
    return ManifestRecord(
        id="test",
        source_dataset=source_dataset,
        structure_output=structure_output,
    )


class TestExplicitUIFirstSources:
    def test_all_seven_ui_first_sources_tag_correctly(self):
        """Per DATA_SOURCES.md: all seven listed sources must resolve to
        ui_first by exact source-key match, independent of the
        structural heuristic."""
        for source in (
            "rico_core", "rico_semantic", "clay", "enrico",
            "webui", "screen2words", "uicrit",
        ):
            rec = _record(source, structure_output=None)
            assert tag_domain(rec) == "ui_first", f"{source} should tag as ui_first"


class TestExplicitGeneralDesignSources:
    def test_pd12m_tags_general_design_by_exact_match(self):
        """BUG FIX regression: pd12m must resolve to general_design via
        an explicit source-key match, not by falling through to the
        heuristic's default branch."""
        rec = _record("pd12m", structure_output=None)
        assert tag_domain(rec) == "general_design"

    def test_cc12m_tags_general_design_by_exact_match(self):
        rec = _record("cc12m", structure_output=None)
        assert tag_domain(rec) == "general_design"

    def test_pd12m_is_not_misrouted_by_a_false_positive_structural_match(self):
        """The core regression case: before the fix, pd12m/cc12m records
        relied entirely on the heuristic's final default branch. If a
        stock photo happened to have >=2 spurious UI-element-type
        detections, it would have been misrouted into ui_first. With an
        explicit source-key match, the structural heuristic is never
        consulted for these sources at all."""
        spurious_structure = {
            "elements": [
                {"type": "button", "bbox": [0, 0, 1, 1]},
                {"type": "toolbar", "bbox": [0, 0, 1, 1]},
            ]
        }
        rec = _record("pd12m", structure_output=spurious_structure)
        assert tag_domain(rec) == "general_design", (
            "pd12m must stay general_design even with a spurious "
            "structural UI-element match — the explicit source match "
            "must take priority over the heuristic"
        )


class TestStructuralHeuristicFallback:
    def test_unknown_source_with_ui_elements_falls_back_to_ui_first(self):
        """Sources not in either explicit set still use the structural
        heuristic as a fallback — this behavior is intentional and
        should keep working for genuinely unrecognized sources."""
        structure = {
            "elements": [
                {"type": "button", "bbox": [0, 0, 1, 1]},
                {"type": "dropdown", "bbox": [0, 0, 1, 1]},
            ]
        }
        rec = _record("some_new_unlisted_source", structure_output=structure)
        assert tag_domain(rec) == "ui_first"

    def test_unknown_source_with_no_ui_elements_defaults_general_design(self):
        rec = _record("some_new_unlisted_source", structure_output=None)
        assert tag_domain(rec) == "general_design"

    def test_unknown_source_with_only_one_ui_element_type_defaults_general_design(self):
        """A single overlapping type isn't enough to trip the heuristic
        (requires >=2)."""
        structure = {"elements": [{"type": "button", "bbox": [0, 0, 1, 1]}]}
        rec = _record("some_new_unlisted_source", structure_output=structure)
        assert tag_domain(rec) == "general_design"
