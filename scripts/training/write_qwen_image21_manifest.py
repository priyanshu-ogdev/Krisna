"""Write the required manifest beside an official Qwen Image 2.1 LoRA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
from krivex_training.polish.qwen_manifest import write_manifest  # noqa: I001

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_path")
    parser.add_argument("--dataset-dir", required=True,
                        help="prepared split-isolated dataset containing dataset_receipt.json")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--inference-steps", type=int, default=40)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--dataset-manifest-sha256")
    parser.add_argument("--split-counts-json")
    parser.add_argument("--lora-target-modules", default="")
    args = parser.parse_args()
    print(write_manifest(args.adapter_path, dataset_dir=args.dataset_dir, resolution=args.resolution,
                         inference_steps=args.inference_steps, rank=args.rank,
                         lora_alpha=args.lora_alpha, base_revision=args.base_revision,
                         dataset_manifest_sha256=args.dataset_manifest_sha256,
                         split_counts=json.loads(args.split_counts_json) if args.split_counts_json else None,
                         lora_target_modules=[item for item in args.lora_target_modules.split(",") if item]))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
