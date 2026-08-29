"""Domain tagging — classifies records as ui_first vs general_design."""

from __future__ import annotations

from data_forge.logging_setup import get_logger
from data_forge.manifest import ManifestRecord

log = get_logger("data.domain_tagger")

# Datasets known to be UI-first (mobile/web app screenshots)
# BUG FIX: enrico, screen2words, and uicrit are real, configured dataset
# keys (see configs/datasets.yaml and the PRD's §8.1 data sources table —
# Enrico is explicitly "cleanest subset" of UI layout data, Screen2Words is
# UI caption pairs, UICrit is UI critique screens) but were missing here.
# Without an exact source-key match, tag_domain() falls through to the
# structure-based heuristic below, which only fires on >=2 overlapping UI
# element *types* extracted by s06_structure — a real, silent
# misclassification risk for datasets whose structural extraction doesn't
# happen to surface two matching types, undermining the ui_first_ratio
# enforcement in s07_routing.py before it even runs. figma_ui and
# dribbble_ui are left in place as forward-declared source keys for
# datasets not yet added to datasets.yaml — harmless if unused.
_UI_FIRST_SOURCES = frozenset([
    "rico_core",
    "rico_semantic",
    "clay",
    "enrico",
    "webui",
    "screen2words",
    "uicrit",
    "figma_ui",
    "dribbble_ui",
])

# Datasets known to be general design
# BUG FIX: pd12m and cc12m are real, configured dataset keys (see
# configs/datasets.yaml — both category: "general_visual", PRD §8.1's
# "General visual backbone" / "Supplementary" rows) but were missing
# here, same class of gap as the ui_first fix above. Without an exact
# match, both fell through to the structure-based heuristic and were
# only correctly tagged `general_design` via that heuristic's final
# default branch — which works in practice (stock/photo content rarely
# trips >=2 UI-element-type overlaps) but is fragile: a false-positive
# structural detection on a PD12M/CC12M image would silently misroute it
# into `ui_first`, contaminating the sketch tier's domain-gated training
# data (s08_encoding.py Branch 3) with non-UI content — exactly what the
# domain gate exists to prevent. Explicit source-key matching removes
# that dependency on the heuristic's default behavior. design_inspiration,
# behance_general, and poster_design are left in place as forward-
# declared source keys for datasets not yet added to datasets.yaml —
# harmless if unused, same pattern as figma_ui/dribbble_ui above.
_GENERAL_DESIGN_SOURCES = frozenset([
    "pd12m",
    "cc12m",
    "design_inspiration",
    "behance_general",
    "poster_design",
])


def tag_domain(record: ManifestRecord) -> str:
    """Classify a record as 'ui_first' or 'general_design'.

    Uses a combination of:
    1. Source dataset category (from datasets.yaml)
    2. Structural extraction hints (presence of UI components)
    """
    # Source-based classification
    if record.source_dataset in _UI_FIRST_SOURCES:
        return "ui_first"
    if record.source_dataset in _GENERAL_DESIGN_SOURCES:
        return "general_design"

    # Structure-based heuristic: if structural extraction found UI elements
    if record.structure_output:
        elements = record.structure_output.get("elements", [])
        ui_types = {
            "button", "text_input", "navigation_bar", "tab_bar",
            "toggle", "checkbox", "dropdown", "slider", "toolbar",
        }
        found_types = {e.get("type", "") for e in elements}
        ui_overlap = found_types & ui_types
        if len(ui_overlap) >= 2:
            return "ui_first"

    # Default to general_design for unknown sources
    return "general_design"
