"""Pipeline stages package — register and manage all pipeline stages."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from data_forge.stages.base import Stage, StageResult

if TYPE_CHECKING:
    from typing import Any

log = logging.getLogger("data_forge.stages")

STAGE_MODULES: tuple[str, ...] = (
    "s00_manifest_planning",
    "s01_fetch",
    "s01_5_uicrit_join",
    "s01_6_preference_pairs",
    "s02_dedup",
    "s03_quality",
    "s03_5_pii_scrub",
    "s04_safety",
    "s04_5_escalation",
    "s05_recaption",
    "s05_ocr_enrichment",
    "s05_5_pii_text_redact",
    "s06_structure",
    "s07_routing",
    "s08_encoding",
    "s08_5_dpo_encoding",
    "s09_heldout",
    "s10_audit",
    "s11_registry_watcher",
    "s12_model_data_export",
)


def register_all_stages() -> dict[str, Any]:
    """Import all stage modules to trigger @register_stage decorators.

    Returns a dictionary of registered stage names to their classes.
    """
    for mod_name in STAGE_MODULES:
        full_name = f"data_forge.stages.{mod_name}"
        try:
            importlib.import_module(full_name)
        except Exception as e:
            log.warning("could_not_load_stage: module=%s, error=%s", full_name, e)

    from data_forge.orchestrator import get_registered_stages

    return get_registered_stages()


__all__ = [
    "Stage",
    "StageResult",
    "STAGE_MODULES",
    "register_all_stages",
]
