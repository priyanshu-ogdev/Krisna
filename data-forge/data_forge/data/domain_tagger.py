"""Domain tagging — classifies records as ui_first vs general_design."""

from __future__ import annotations

from data_forge.logging_setup import get_logger
from data_forge.manifest import ManifestRecord

log = get_logger("data.domain_tagger")

# Datasets known to be UI-first (mobile/web app screenshots)
_UI_FIRST_SOURCES = frozenset([
    "rico_core",
    "rico_semantic",
    "clay",
    "webui",
    "figma_ui",
    "dribbble_ui",
])

# Datasets known to be general design
_GENERAL_DESIGN_SOURCES = frozenset([
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
