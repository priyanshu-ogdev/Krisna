#!/usr/bin/env python3
"""Synchronize Data-Forge exported model_data into training-ready datasets in ./data/

Bridges the data-forge output to the training tier configurations:
1. Polish default LoRA:
   DATA_ROOT/model_data/polish_zimage_turbo -> ./data/polish_default_train (images + metadata.jsonl)
2. Diffusion-DPO preference pairs:
   DATA_ROOT/preference_pairs -> ./krisna_preference_pairs.db + ./krisna_blobs/
3. Sketch tier (MaskGIT):
   DATA_ROOT/model_data/sketch_tier_maskgit -> ./data/sketch_train_256 (256px VQ tokens + manifest.jsonl)
   DATA_ROOT/model_data/sketch_tier_maskgit -> ./data/sketch_train_512 (512px VQ tokens + manifest.jsonl)
4. Planner RAG Corpus:
   DATA_ROOT/model_data/planner_rag_corpus/uicrit_critiques.jsonl -> ./data/planner_rag_corpus/uicrit_critiques.jsonl

Usage:
    python scripts/data-forge/sync_to_training.py
    python scripts/data-forge/sync_to_training.py --data-root D:\\data_krisna
    python scripts/data-forge/sync_to_training.py --check-only
    python scripts/data-forge/sync_to_training.py --no-sketch-512
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repository packages are in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in [REPO_ROOT / "data-forge" / "src", REPO_ROOT / "training" / "src", REPO_ROOT / "inference" / "src", REPO_ROOT]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


def _load_env_file() -> None:
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
        except Exception:
            pass


_load_env_file()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync Data-Forge model_data into training-ready ./data/ datasets."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", str(REPO_ROOT / "data_krisna")),
        help="Root storage directory where data-forge outputs land (default: $DATA_ROOT or ./data_krisna)",
    )
    parser.add_argument(
        "--target-dir",
        default=str(REPO_ROOT / "data"),
        help="Target directory for training data (default: ./data)",
    )
    parser.add_argument(
        "--pref-db",
        default=str(REPO_ROOT / "krisna_preference_pairs.db"),
        help="Path to preference pairs SQLite DB (default: ./krisna_preference_pairs.db)",
    )
    parser.add_argument(
        "--blob-root",
        default=str(REPO_ROOT / "krisna_blobs"),
        help="Path to shared BlobStore root (default: ./krisna_blobs)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only inspect and report readiness of data-forge exports without modifying ./data",
    )
    parser.add_argument(
        "--sketch-256",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sync sketch tier at 256px resolution (default: True)",
    )
    parser.add_argument(
        "--sketch-512",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sync sketch tier at 512px resolution (default: True)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_data_dir = data_root / "model_data"

    print("=" * 70)
    print("  Krisna Data-Forge -> Model Training Bridge")
    print("=" * 70)
    print(f"  DATA_ROOT:          {data_root}")
    print(f"  Model Data Export:  {model_data_dir}")
    print(f"  Training Data Dir:  {target_dir}")
    print(f"  Preference DB:      {args.pref_db}")
    print(f"  Blob Store:         {args.blob_root}")
    print(f"  Sketch 256px:       {args.sketch_256}")
    print(f"  Sketch 512px:       {args.sketch_512}")
    print("=" * 70)

    if not model_data_dir.exists() and not (data_root / "preference_pairs").exists():
        print(f"\n[!] Notice: No data-forge exports found at {data_root}.")
        print("    Run the pipeline first to generate data:")
        print("        ./run_data_forge.sh run")
        print("        or: .\\run_data_forge.ps1 run")
        return 0

    results = {}

    # 1. Polish Tier (Z-Image-Turbo LoRA)
    zimage_export = model_data_dir / "polish_zimage_turbo"
    polish_target = target_dir / "polish_default_train"
    if zimage_export.exists() and (zimage_export / "images").exists():
        count = len(list((zimage_export / "images").glob("*.*")))
        if args.check_only:
            print(f"\n[OK] Polish Tier: Found {count} exported images in {zimage_export}")
            results["polish"] = f"Ready ({count} images in export)"
        else:
            print(f"\n[*] Syncing Polish Tier: {zimage_export} -> {polish_target}...")
            try:
                from krisna_training.data_forge_bridge.sync_polish_default import sync as sync_polish
                out = sync_polish(model_data_dir, polish_target)
                synced_count = len([p for p in polish_target.iterdir() if p.is_file() and p.name != "metadata.jsonl"])
                print(f"    [SUCCESS] Polish dataset synced to {out} ({synced_count} images with metadata.jsonl)")
                results["polish"] = f"Synced ({synced_count} images) -> ready for './train.sh polish-default'"
            except Exception as e:
                print(f"    [ERROR] Polish sync failed: {e}")
                results["polish"] = f"Failed: {e}"
    else:
        print(f"\n[-] Polish Tier: {zimage_export}/images not yet populated (run stage s12).")
        results["polish"] = "Not exported yet"

    # 2. Diffusion-DPO Preference Pairs
    pref_source = data_root / "preference_pairs"
    if pref_source.exists():
        if args.check_only:
            count = len(list(pref_source.rglob("*.json")))
            print(f"\n[OK] Preference Pairs: Found {count} JSON pairs in {pref_source}")
            results["dpo"] = f"Ready ({count} pairs)"
        else:
            print(f"\n[*] Syncing Preference Pairs: {pref_source} -> {args.pref_db}...")
            try:
                from krisna_training.data_forge_bridge.sync_dpo_pairs import sync as sync_dpo
                from krisna_training.dpo.preference_store import PreferenceStore
                from krisna_inference.backends.common import BlobStore

                blob_store = BlobStore(root=Path(args.blob_root))
                pref_store = PreferenceStore(db_path=Path(args.pref_db))
                counts = sync_dpo(data_root, pref_store, blob_store)
                pref_store.close()
                total_imported = sum(v for k, v in counts.items() if not k.startswith("skipped"))
                print(f"    [SUCCESS] Preference pairs imported: {total_imported} pairs across sources: {counts}")
                results["dpo"] = f"Synced ({total_imported} pairs) -> ready for './train.sh polish-dpo'"
            except Exception as e:
                print(f"    [ERROR] DPO sync failed: {e}")
                results["dpo"] = f"Failed: {e}"
    else:
        print(f"\n[-] Preference Pairs: {pref_source} not yet populated (run stage s01_6).")
        results["dpo"] = "Not exported yet"

    # 3. Sketch Tier (MaskGIT) — Stage 1 (256px) and Stage 2 (512px)
    sketch_export = model_data_dir / "sketch_tier_maskgit"
    sketch_target_256 = target_dir / "sketch_train_256"
    sketch_target_512 = target_dir / "sketch_train_512"
    vqgan_ckpt = REPO_ROOT / "checkpoints" / "vqgan" / "last.ckpt"
    vqgan_cfg = REPO_ROOT / "checkpoints" / "vqgan" / "model.yaml"

    if sketch_export.exists() and (sketch_export / "images").exists():
        count = len(list((sketch_export / "images").glob("*.*")))
        if not (vqgan_ckpt.exists() and vqgan_cfg.exists()):
            print(f"\n[!] Sketch Tier: Found {count} images, but VQGAN weights are missing.")
            print(f"    Expected: {vqgan_ckpt} and {vqgan_cfg}")
            print("    To download VQGAN weights:")
            print("        ./scripts/training/download_vqgan.sh")
            print("        or: python scripts/inference/download_weights.py")
            if args.sketch_256:
                results["sketch_256"] = f"Images ready ({count}), awaiting VQGAN weights"
            if args.sketch_512:
                results["sketch_512"] = f"Images ready ({count}), awaiting VQGAN weights"
        elif args.check_only:
            print(f"\n[OK] Sketch Tier: Found {count} exported images + VQGAN checkpoint verified.")
            if args.sketch_256:
                m256 = (sketch_target_256 / "manifest.jsonl").exists()
                results["sketch_256"] = "Ready (manifest exists)" if m256 else f"Ready to sync ({count} images)"
            if args.sketch_512:
                m512 = (sketch_target_512 / "manifest.jsonl").exists()
                results["sketch_512"] = "Ready (manifest exists)" if m512 else f"Ready to sync ({count} images)"
        else:
            try:
                from krisna_training.data_forge_bridge.sync_sketch_tier import sync as sync_sketch
                from krisna_training.sketch.vq_tokenizer import VQTokenizer

                tokenizer = VQTokenizer(checkpoint_path=vqgan_ckpt, config_path=vqgan_cfg)

                if args.sketch_256:
                    print(f"\n[*] Syncing Sketch Tier Stage 1 (256px): {sketch_export} -> {sketch_target_256}...")
                    m256 = sync_sketch(model_data_dir, sketch_target_256, tokenizer, image_size=256)
                    print(f"    [SUCCESS] Sketch Stage 1 (256px) manifest written to {m256}")
                    results["sketch_256"] = "Synced -> ready for './train.sh sketch-stage1'"

                if args.sketch_512:
                    print(f"\n[*] Syncing Sketch Tier Stage 2 (512px): {sketch_export} -> {sketch_target_512}...")
                    m512 = sync_sketch(model_data_dir, sketch_target_512, tokenizer, image_size=512)
                    print(f"    [SUCCESS] Sketch Stage 2 (512px) manifest written to {m512}")
                    results["sketch_512"] = "Synced -> ready for './train.sh sketch-stage2'"
            except Exception as e:
                print(f"    [ERROR] Sketch sync failed: {e}")
                if args.sketch_256:
                    results["sketch_256"] = f"Failed: {e}"
                if args.sketch_512:
                    results["sketch_512"] = f"Failed: {e}"
    else:
        print(f"\n[-] Sketch Tier: {sketch_export}/images not yet populated (run stage s12).")
        if args.sketch_256:
            results["sketch_256"] = "Not exported yet"
        if args.sketch_512:
            results["sketch_512"] = "Not exported yet"

    # 4. Planner RAG Corpus
    rag_export = model_data_dir / "planner_rag_corpus" / "uicrit_critiques.jsonl"
    rag_target_dir = target_dir / "planner_rag_corpus"
    rag_target_file = rag_target_dir / "uicrit_critiques.jsonl"

    if rag_export.exists():
        critique_count = sum(1 for line in rag_export.read_text(encoding="utf-8").splitlines() if line.strip())
        if args.check_only:
            print(f"\n[OK] Planner RAG Corpus: Found {critique_count} human critique records in {rag_export}")
            results["rag_corpus"] = f"Ready ({critique_count} critique records)"
        else:
            print(f"\n[*] Syncing Planner RAG Corpus: {rag_export} -> {rag_target_file}...")
            try:
                from krisna_training.data_forge_bridge.sync_planner_rag import count_critiques, sync as sync_rag
                synced_file = sync_rag(model_data_dir, target_dir)
                rec_count = count_critiques(synced_file)
                print(f"    [SUCCESS] Planner RAG corpus synced ({rec_count} records) -> {synced_file}")
                results["rag_corpus"] = f"Synced ({rec_count} records) -> ready for inference Planner RAG"
            except Exception as e:
                print(f"    [ERROR] Planner RAG sync failed: {e}")
                results["rag_corpus"] = f"Failed: {e}"
    else:
        print(f"\n[-] Planner RAG Corpus: {rag_export} not yet populated (run stage s12).")
        results["rag_corpus"] = "Not exported yet"

    # 5. Final Readiness Summary
    print("\n" + "=" * 70)
    print("  Model Training Readiness Summary")
    print("=" * 70)
    for tier, status in results.items():
        print(f"  * {tier:<15}: {status}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
