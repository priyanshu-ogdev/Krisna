#!/usr/bin/env python3
"""Download verified Planner artifacts into the repository's root models/.

The downloader never overwrites an existing artifact without --force and writes
a manifest containing the exact source revision and SHA256.  GGUF downloads
are intentionally explicit because model publishers may offer several quality
levels and the selected file is a product decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=os.environ.get("KRIVEX_PLANNER_GGUF_REPO", "unsloth/Qwen3.5-9B-GGUF"))
    p.add_argument("--filename", required=True, help="Exact GGUF filename from the publisher")
    p.add_argument("--mmproj-filename")
    p.add_argument("--revision", default="main")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub before downloading Planner models") from exc

    root = Path(args.models_dir) / "planner"
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for filename in [args.filename, args.mmproj_filename]:
        if not filename:
            continue
        target = root / Path(filename).name
        if target.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite {target}; pass --force")
        source = hf_hub_download(args.repo_id, filename=filename, revision=args.revision, local_dir=root)
        source_path = Path(source)
        if source_path != target:
            target.write_bytes(source_path.read_bytes())
        entries.append({"filename": target.name, "sha256": sha256(target)})

    manifest = {"repo_id": args.repo_id, "revision": args.revision, "artifacts": entries}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
