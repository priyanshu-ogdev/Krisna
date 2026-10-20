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
    "screenspot",
    "seeclick",
    "android_control",
    "mind2web",
    "figma_ui",
    "dribbble_ui",
    "figma2code",
    "mobileviews",
    "wave_ui",
    "showui_desktop",
    "websight",
    "aria_ui",
    "atomblock_webui",
    "ui_vision",
    "crello",
    "pku_poster",
    "publaynet",
    "docbank",
    "doclaynet",
    "abo_ecommerce",
    "magicbrush",
])

# Datasets known to be general visual (unstructured photo / web crawl)
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
