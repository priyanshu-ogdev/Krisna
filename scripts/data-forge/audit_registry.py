"""Produce a deterministic, read-only audit receipt for the Data-Forge registry.

This command never downloads data or changes the workspace. It is intended to
be run before a fetch and committed alongside a research run receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="data-forge/configs/datasets.yaml")
    parser.add_argument("--pipeline", default="data-forge/configs/pipeline.yaml")
    parser.add_argument("--models", default="data-forge/configs/models.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(Path("data-forge/src")))
    from data_forge.config import load_config
    from data_forge.stages.s00_manifest_planning import _production_preflight

    config = load_config(args.pipeline, args.models, args.datasets)
    enabled = {key: spec for key, spec in config.datasets.items() if spec.enabled}
    disabled = {key: spec for key, spec in config.datasets.items() if not spec.enabled}
    receipt = {
        "datasets_config": str(Path(args.datasets)),
        "enabled_sources": sorted(enabled),
        "disabled_sources": sorted(disabled),
        "training_eligible_sources": sorted(
            key for key, spec in enabled.items() if spec.training_eligible
        ),
        "policy_violations": config.training_policy_violations(),
        "effective_image_records": sum(
            spec.storage_relevant_record_count() for spec in enabled.values()
        ),
        "effective_preference_pairs": sum(
            spec.preference_pair_relevant_count() for spec in enabled.values()
        ),
        "preflight_blockers": _production_preflight(config),
        "network_fetch_allowed": bool(enabled),
        "training_export_ready": not config.training_policy_violations()
        and any(spec.training_eligible for spec in enabled.values()),
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
