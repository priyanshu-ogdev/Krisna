"""CPU-only preflight and launcher for the native Qwen Image 2.1 trainer."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .qwen_trainer_contract import (
    audit_prepared_dataset,
    build_accelerate_command,
    format_command,
    prepare_trainer_config,
    verify_diffusers_checkout,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("the launcher requires PyYAML to read its checked-in config") from exc
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("training config must be a YAML object")
    return value


def _coerce_value(v: str) -> Any:
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in ("null", "none"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_cli_overrides(extra_args: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--"):
            kv = arg[2:]
            if "=" in kv:
                k, v = kv.split("=", 1)
                overrides[k.replace("-", "_")] = _coerce_value(v)
                i += 1
            else:
                k = kv.replace("-", "_")
                if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                    overrides[k] = _coerce_value(extra_args[i + 1])
                    i += 2
                else:
                    overrides[k] = True
                    i += 1
        else:
            i += 1
    return overrides


def preflight(config_path: str | Path, cli_overrides: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[str], bool]:
    raw = load_yaml(config_path)
    if cli_overrides:
        raw.update(cli_overrides)
    prepared = prepare_trainer_config(raw)
    diffusers_dir = raw["diffusers_dir"]
    script, dirty = verify_diffusers_checkout(diffusers_dir)
    dataset = prepared.get("dataset_name")
    if not dataset:
        raise ValueError("dataset_name is required")
    receipt = audit_prepared_dataset(dataset)
    if receipt["split_counts"].get("train", 0) <= 0:
        raise ValueError("refusing launch with no train records")
    command = build_accelerate_command(raw, diffusers_dir)
    print(json.dumps({
        "status": "preflight_ok",
        "trainer": str(script),
        "diffusers_revision_pinned": True,
        "diffusers_checkout_dirty": dirty,
        "split_counts": receipt["split_counts"],
        "columns": receipt["columns"],
        "command": format_command(command),
    }, sort_keys=True))
    return prepared, command, dirty


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML training config")
    parser.add_argument("--dry-run", action="store_true",
                        help="run CPU-only checks and print the redacted command")
    args, extra = parser.parse_known_args(argv)
    cli_overrides = _parse_cli_overrides(extra) if extra else None
    try:
        _, command, _ = preflight(args.config, cli_overrides=cli_overrides)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0
    # This is the only subprocess in this module that can start training, and
    # it is unreachable until every receipt, split, path, and git check passes.
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
