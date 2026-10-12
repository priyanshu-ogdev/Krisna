"""Write the required manifest beside an official Qwen Image 2.1 LoRA."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
from krisna_training.polish.qwen_manifest import write_manifest  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_path")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--inference-steps", type=int, default=40)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    args = parser.parse_args()
    print(write_manifest(args.adapter_path, resolution=args.resolution,
                         inference_steps=args.inference_steps, rank=args.rank,
                         lora_alpha=args.lora_alpha))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
