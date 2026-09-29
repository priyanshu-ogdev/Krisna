#!/usr/bin/env python3
"""Resolve every unpinned `revision: "main"` in datasets.yaml/models.yaml to
a real commit SHA, via the HuggingFace Hub API — closing the gap flagged
throughout this codebase's comments ("run scripts/pin_revisions.py...")
that was never actually built.

API verified directly against two independent, authoritative sources
before writing this (not assumed): HuggingFace's own Hub API docs
(`GET /api/{models,datasets}/{repo_id}/revision/{revision}`) and the
`huggingface_hub` library's own `HfApi.model_info()`/`dataset_info()`
source, both confirming the response includes a `sha` field — this
script uses the library wrappers directly rather than hand-rolling HTTP,
since they already handle auth headers/retries correctly.

Uses `ruamel.yaml` (round-trip mode), not plain PyYAML — these config
files carry extensive, load-bearing comments (dataset verification notes,
license findings, bug-fix explanations accumulated over this project's
whole review history). A naive `yaml.safe_load()` + `yaml.dump()` round
trip would silently destroy every one of them. `ruamel.yaml` preserves
comments and formatting when you mutate a scalar value in place rather
than rebuilding the document from scratch.

Usage:
    python scripts/pin_revisions.py                  # dry run — prints
                                                        what would change
    python scripts/pin_revisions.py --apply           # writes the real
                                                        SHAs in place
    python scripts/pin_revisions.py --only pd12m,rico_core   # scope to
                                                        specific keys
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ruamel.yaml import YAML


def _load(path: Path):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # avoid re-wrapping long lines, which would touch
                        # unrelated content and make diffs noisy
    # CONFIRMED via an actual load->dump->diff round-trip test against both
    # real config files, not assumed: ruamel's default sequence indent
    # doesn't match this project's YAML style (4-space parent key, 6-space
    # list items — i.e. sequence=4, offset=2), so without this every
    # --apply run would have silently reformatted every list in the file
    # even when nothing semantically changed. Verified this exact setting
    # produces a byte-for-byte no-op round-trip on an unmodified file.
    yaml.indent(mapping=2, sequence=4, offset=2)

    # CONFIRMED separately: ruamel's default NoneType representer drops
    # explicit `null` values to an empty scalar on dump (a real, known
    # ruamel quirk, not this project's bug) — `quantization: null` would
    # silently become `quantization:` on any --apply run that touched
    # that file at all, even for an entry this script never looked at.
    # Both fields are semantically None either way, but an unrelated,
    # unexplained formatting change in a --apply run's diff is exactly
    # the kind of thing that erodes trust in a "safe" automation script.
    def _represent_none(representer, data):
        return representer.represent_scalar("tag:yaml.org,2002:null", "null")
    yaml.representer.add_representer(type(None), _represent_none)

    with path.open() as f:
        return yaml, yaml.load(f)


def _save(yaml: YAML, data, path: Path) -> None:
    with path.open("w") as f:
        yaml.dump(data, f)


def resolve_dataset_revisions(datasets_yaml: Path, only: set[str] | None, apply: bool) -> list[dict]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    yaml, data = _load(datasets_yaml)
    api = HfApi()
    results = []

    for key, spec in data.get("datasets", {}).items():
        if only and key not in only:
            continue
        if not isinstance(spec, dict):
            continue
        if spec.get("source_type") != "huggingface":
            continue
        repo_id = spec.get("repo_id")
        if not repo_id:
            # designsense_10k / designpref — repo_id: null, nothing to
            # resolve. Not an error; these are documented as genuinely
            # not-yet-released elsewhere in this file.
            continue
        current_revision = spec.get("revision")
        if current_revision not in ("main", None):
            results.append({"key": key, "repo_id": repo_id, "status": "already_pinned", "revision": current_revision})
            continue

        try:
            info = api.dataset_info(repo_id, revision="main")
            sha = info.sha
        except HfHubHTTPError as e:
            results.append({"key": key, "repo_id": repo_id, "status": "error", "error": str(e)})
            continue
        except Exception as e:
            results.append({"key": key, "repo_id": repo_id, "status": "error", "error": f"{type(e).__name__}: {e}"})
            continue

        results.append({"key": key, "repo_id": repo_id, "status": "resolved", "sha": sha})
        if apply:
            spec["revision"] = sha  # in-place mutation — ruamel keeps the
                                      # line's trailing comment attached

    if apply:
        _save(yaml, data, datasets_yaml)

    return results


def resolve_model_revisions(models_yaml: Path, only: set[str] | None, apply: bool) -> list[dict]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    yaml, data = _load(models_yaml)
    api = HfApi()
    results = []

    def _walk(node, path_prefix=""):
        """models.yaml nests model specs under top-level sections
        (tier1/tier2/ocr/embeddings/encoders/...) — walk any dict that
        has both model_id and revision keys, regardless of nesting depth,
        rather than hardcoding each section name (which would silently
        stop working the next time a section is renamed or added)."""
        if not isinstance(node, dict):
            return
        if "model_id" in node and "revision" in node:
            key = path_prefix or "?"
            if only and key not in only:
                return
            model_id = node["model_id"]
            current_revision = node.get("revision")
            if current_revision not in ("main", None):
                results.append({"key": key, "repo_id": model_id, "status": "already_pinned", "revision": current_revision})
                return
            try:
                info = api.model_info(model_id, revision="main")
                sha = info.sha
            except HfHubHTTPError as e:
                results.append({"key": key, "repo_id": model_id, "status": "error", "error": str(e)})
                return
            except Exception as e:
                results.append({"key": key, "repo_id": model_id, "status": "error", "error": f"{type(e).__name__}: {e}"})
                return
            results.append({"key": key, "repo_id": model_id, "status": "resolved", "sha": sha})
            if apply:
                node["revision"] = sha
        else:
            for k, v in node.items():
                _walk(v, path_prefix=f"{path_prefix}.{k}" if path_prefix else str(k))

    _walk(data)
    if apply:
        _save(yaml, data, models_yaml)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs-dir", default="data-forge/configs", help="Directory containing datasets.yaml/models.yaml")
    parser.add_argument("--apply", action="store_true", help="Write real SHAs in place. Default: dry run, print only.")
    parser.add_argument("--only", default=None, help="Comma-separated dataset/model keys to scope to.")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    configs_dir = Path(args.configs_dir)
    datasets_yaml = configs_dir / "datasets.yaml"
    models_yaml = configs_dir / "models.yaml"

    if not datasets_yaml.exists() or not models_yaml.exists():
        print(f"Could not find datasets.yaml/models.yaml under {configs_dir}", file=sys.stderr)
        return 1

    print(f"{'APPLYING' if args.apply else 'DRY RUN'} — resolving unpinned revisions...\n")

    dataset_results = resolve_dataset_revisions(datasets_yaml, only, args.apply)
    model_results = resolve_model_revisions(models_yaml, only, args.apply)

    had_errors = False
    for label, results in (("datasets.yaml", dataset_results), ("models.yaml", model_results)):
        print(f"--- {label} ---")
        for r in results:
            if r["status"] == "resolved":
                print(f"  {r['key']:30s} {r['repo_id']:45s} -> {r['sha']}")
            elif r["status"] == "already_pinned":
                print(f"  {r['key']:30s} {r['repo_id']:45s} already pinned to {r['revision']}")
            elif r["status"] == "error":
                had_errors = True
                print(f"  {r['key']:30s} {r['repo_id']:45s} ERROR: {r['error']}")
        print()

    if not args.apply:
        print("Dry run only — no files were changed. Re-run with --apply to write real SHAs.")

    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())
