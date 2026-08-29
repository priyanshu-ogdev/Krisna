"""Resolve every model in configs/models.yaml to its current real commit SHA
and write it back — replacing 'main' (which drifts over time) with an
immutable pin. Run this deliberately, when you're ready to lock a dataset
or training version, not automatically on every pipeline run.

Requires network access to huggingface.co, which the development sandbox
this pipeline was built in does NOT have — this must be run on a machine
that does (your actual training box, or any machine with normal internet).

Usage:
    python scripts/pin_revisions.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing the file.",
    )
    parser.add_argument(
        "--config", default="configs/models.yaml", type=Path,
        help="Path to models.yaml (default: configs/models.yaml)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "huggingface_hub is required: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    with open(args.config, encoding="utf-8") as f:
        models = yaml.safe_load(f)

    api = HfApi()
    changed = []
    failed = []

    def walk(node, path=""):
        """Find every dict with a model_id key, anywhere in the nested config."""
        if isinstance(node, dict):
            if "model_id" in node:
                yield path, node
            for k, v in node.items():
                yield from walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from walk(v, f"{path}[{i}]")

    for path, spec in walk(models):
        repo_id = spec["model_id"]
        old_revision = spec.get("revision", "main")
        repo_type = spec.get("repo_type", "model")
        try:
            refs = api.list_repo_refs(repo_id, repo_type=repo_type)
            if not refs.branches:
                failed.append((path, repo_id, "no branches returned"))
                continue
            main_branch = next(
                (b for b in refs.branches if b.name == "main"), refs.branches[0]
            )
            new_revision = main_branch.target_commit
        except Exception as e:  # noqa: BLE001 — report and continue, don't abort the whole run
            failed.append((path, repo_id, str(e)))
            continue

        if new_revision != old_revision:
            changed.append((path, repo_id, old_revision, new_revision))
            if not args.dry_run:
                spec["revision"] = new_revision

    print(f"\n{'Would pin' if args.dry_run else 'Pinned'} {len(changed)} model(s):")
    for path, repo_id, old, new in changed:
        print(f"  {path} ({repo_id}): {old!r} -> {new}")

    if failed:
        print(f"\n{len(failed)} model(s) could not be resolved — left unchanged:")
        for path, repo_id, reason in failed:
            print(f"  {path} ({repo_id}): {reason}")

    if not args.dry_run and changed:
        with open(args.config, "w", encoding="utf-8") as f:
            yaml.safe_dump(models, f, default_flow_style=False, sort_keys=False)
        print(f"\nWrote {args.config}. Review the diff before committing.")
    elif args.dry_run:
        print("\nDry run — no file written.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
