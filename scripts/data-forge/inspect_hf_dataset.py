#!/usr/bin/env python3
"""Utility script to query HuggingFace datasets server for dataset schema and info.

Queries the datasets-server.huggingface.co API for dataset metadata, features,
and configuration info without downloading full datasets.

Usage:
    python scripts/data-forge/inspect_hf_dataset.py --datasets conceptual_12m bevaya/RICO-Screen2Words
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Sequence


def fetch_dataset_info(dataset_name: str, timeout: float = 10.0) -> dict | None:
    url = f"https://datasets-server.huggingface.co/info?dataset={dataset_name}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "krisna-data-forge/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"[-] HTTP Error {exc.code} fetching {dataset_name}: {exc.reason}", file=sys.stderr)
    except urllib.error.URLError as exc:
        print(f"[-] Network Error fetching {dataset_name}: {exc.reason}", file=sys.stderr)
    except Exception as exc:
        print(f"[-] Error fetching {dataset_name}: {exc}", file=sys.stderr)
    return None


def print_dataset_info(dataset_name: str, data: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    dataset_info = data.get("dataset_info", {})
    if not dataset_info:
        print("  No configuration info available.")
        return

    for config_name, info in dataset_info.items():
        print(f"\n  Configuration: {config_name}")
        features = info.get("features", {})
        if features:
            print("  Features:")
            for feat_name, feat_spec in features.items():
                feat_type = feat_spec.get("_type", feat_spec.get("dtype", str(feat_spec)))
                print(f"    - {feat_name}: {feat_type}")
        splits = info.get("splits", {})
        if splits:
            print("  Splits:")
            for split_name, split_spec in splits.items():
                num_examples = split_spec.get("num_examples", "unknown")
                print(f"    - {split_name}: {num_examples} examples")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect HuggingFace dataset schemas and info.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["conceptual_12m", "bevaya/RICO-Screen2Words"],
        help="One or more HuggingFace dataset identifiers",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds (default: 10.0)",
    )
    args = parser.parse_args(argv)

    for ds in args.datasets:
        info = fetch_dataset_info(ds, timeout=args.timeout)
        if info:
            print_dataset_info(ds, info)
        else:
            print(f"Could not retrieve metadata for {ds}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
