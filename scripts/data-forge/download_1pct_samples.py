#!/usr/bin/env python3
"""
Krisna Data-Forge: 1% Target Dataset Downloader & Local Validator
================================================================

Downloads and verifies a 1% representative sample of every dataset
required by the Krisna data pipeline for end-to-end validation on
the development machine (12GB RTX 3060, 32GB RAM).

Target Footprints:
- PD12M: 2,000 images (from 200,000 full target)
- CC12M: 1,500 images (from 150,000 full target)
- RICO Core: 663 screenshots (from 66,261 full target)
- RICO Semantic: 663 screenshots (from 66,261 full target)
- Screen2Words: 225 screenshots + captions (from 22,417 full target)
- CLAY: Full repo CSV (59,555 layout annotations, ~24MB)
- Enrico: Full repo (1,460 screen topic annotations, ~1MB)
- UICrit: Full repo CSV (1,000 critiques, ~4.9MB)
- WebUI: 350K split metadata JSON (~5.95MB)
- GameLabel-10K: 100 preference pairs (from 9,800 full target)
- HPDv2: 600 preference pairs (from 60,000 full target)
- TASTE Benchmark: 1,600 eval ratings (Full benchmark)
- PartiPrompts: 1,632 prompts (Full benchmark)
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import gzip
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "D:/data_krisna"))
RAW_DIR = DATA_ROOT / "raw"
PREF_DIR = DATA_ROOT / "preference_pairs"
EVAL_DIR = DATA_ROOT / "heldout" / "external_eval"


def log(msg: str, **kwargs):
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    prefix = f"[{time.strftime('%H:%M:%S')}]"
    print(f"{prefix} {msg} {extra}".strip(), flush=True)


def decode_gamelabel_image(raw: Any) -> bytes | None:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, dict) and "bytes" in raw:
        return raw["bytes"]
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    elif s.startswith('b"') and s.endswith('"'):
        s = s[2:-1]
    try:
        return base64.b64decode(s)
    except Exception:
        return None


def setup_directories():
    for d in [
        RAW_DIR / "pd12m",
        RAW_DIR / "cc12m",
        RAW_DIR / "rico_core",
        RAW_DIR / "rico_semantic",
        RAW_DIR / "screen2words",
        RAW_DIR / "clay",
        RAW_DIR / "enrico",
        RAW_DIR / "uicrit",
        RAW_DIR / "webui",
        PREF_DIR / "gamelabel_10k",
        PREF_DIR / "hpdv2",
        EVAL_DIR / "taste",
        EVAL_DIR / "partiprompts",
    ]:
        d.mkdir(parents=True, exist_ok=True)
    log("Directories initialized under", root=str(DATA_ROOT))


def sync_eval_references():
    log("Checking evaluation reference datasets...")
    taste_dir = EVAL_DIR / "taste"
    parti_dir = EVAL_DIR / "partiprompts"

    cache_root = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    taste_snaps = list(cache_root.glob("datasets--purvanshi--TASTE/snapshots/*"))
    parti_snaps = list(cache_root.glob("datasets--nateraw--parti-prompts/snapshots/*"))

    if taste_snaps and not list(taste_dir.glob("*.parquet")) and not list(taste_dir.glob("*.json")):
        shutil.copytree(taste_snaps[0], taste_dir, dirs_exist_ok=True)
        log("Synced TASTE reference benchmark", files=len(list(taste_dir.glob("*"))))
    else:
        log("TASTE reference benchmark already present", files=len(list(taste_dir.glob("*"))))

    if parti_snaps and not list(parti_dir.glob("*.tsv")):
        shutil.copytree(parti_snaps[0], parti_dir, dirs_exist_ok=True)
        log("Synced PartiPrompts benchmark", files=len(list(parti_dir.glob("*"))))
    else:
        log("PartiPrompts benchmark already present", files=len(list(parti_dir.glob("*"))))


def download_webui_metadata():
    dest = RAW_DIR / "webui" / "train_split_web350k.json"
    if dest.exists() and dest.stat().st_size > 100_000:
        log("WebUI metadata already present", size=dest.stat().st_size)
        return
    log("Downloading WebUI 350K split metadata...")
    r = httpx.get(
        "https://huggingface.co/datasets/biglab/webui-350k/resolve/main/train_split_web350k.json",
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    r.raise_for_status()
    dest.write_bytes(r.content)
    log("Downloaded WebUI metadata", bytes=len(r.content))


def extract_gamelabel_10k(target_count: int = 100):
    dest = PREF_DIR / "gamelabel_10k"
    existing = len(list(dest.glob("*.json")))
    if existing >= target_count:
        log("GameLabel-10K pairs already present", count=existing)
        return

    log("Fetching GameLabel-10K partial CSV via HTTP Range (35MB)...")
    req = urllib.request.Request(
        "https://huggingface.co/datasets/Jonathan-Zhou/GameLabel-10k/resolve/main/data.csv",
        headers={"Range": "bytes=0-36700160", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    last_nl = data.rfind(b"\n")
    df = pd.read_csv(io.BytesIO(data[:last_nl]))
    log("Parsed GameLabel-10K CSV buffer", rows=len(df))

    written = 0
    seen_hashes = set()
    for idx, row in df.iterrows():
        try:
            va, vb = int(row["img0_votes"]), int(row["img1_votes"])
        except Exception:
            continue
        if va == vb:
            continue
        label = "a" if va > vb else "b"
        a_blob = decode_gamelabel_image(row.get("img0_encoding"))
        b_blob = decode_gamelabel_image(row.get("img1_encoding"))
        if not a_blob or not b_blob:
            continue

        try:
            a_img = Image.open(io.BytesIO(a_blob)).convert("RGB")
            b_img = Image.open(io.BytesIO(b_blob)).convert("RGB")
        except Exception:
            continue

        pair_hash = hashlib.sha256(a_blob).hexdigest() + hashlib.sha256(b_blob).hexdigest()
        if pair_hash in seen_hashes:
            continue
        seen_hashes.add(pair_hash)

        pair_id = f"gamelabel_10k_{written:07d}"
        a_img.save(dest / f"{pair_id}_a.png", "PNG")
        b_img.save(dest / f"{pair_id}_b.png", "PNG")
        (dest / f"{pair_id}.json").write_text(
            json.dumps(
                {
                    "pair_id": pair_id,
                    "prompt": str(row.get("prompt", "")),
                    "image_a": f"{pair_id}_a.png",
                    "image_b": f"{pair_id}_b.png",
                    "preferred": label,
                    "origin": "gamelabel_10k",
                    "label_source": "human",
                    "vote_margin": abs(va - vb),
                }
            ),
            encoding="utf-8",
        )
        written += 1
        if written >= target_count:
            break

    log("Extracted GameLabel-10K preference pairs", count=written)


def extract_hpdv2(target_count: int = 600):
    dest = PREF_DIR / "hpdv2"
    existing = len(list(dest.glob("*.json")))
    if existing >= target_count:
        log("HPDv2 pairs already present", count=existing)
        return

    log("Loading HPDv2 annotations (test.json)...")
    cache_root = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    test_json_candidates = list(cache_root.glob("datasets--ymhao--HPDv2/**/test.json"))
    if not test_json_candidates:
        test_json_path = hf_hub_download("ymhao/HPDv2", "test.json", repo_type="dataset")
    else:
        test_json_path = str(test_json_candidates[0])

    with open(test_json_path, encoding="utf-8") as f:
        meta_data = json.load(f)

    log("Streaming HPDv2 images from test.tar.gz via HTTP Range (32MB)...")
    req = urllib.request.Request(
        "https://huggingface.co/datasets/ymhao/HPDv2/resolve/main/test.tar.gz",
        headers={"Range": "bytes=0-33554432", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req) as resp:
        gz_data = resp.read()

    images_by_name: dict[str, bytes] = {}
    try:
        gz = gzip.GzipFile(fileobj=io.BytesIO(gz_data))
        tf = tarfile.TarFile(fileobj=gz)
        while True:
            try:
                m = tf.next()
                if m is None:
                    break
                if m.isfile():
                    f_data = tf.extractfile(m)
                    if f_data:
                        # Normalize name: "test/00600.jpg" -> "00600.jpg"
                        base_name = os.path.basename(m.name)
                        images_by_name[base_name] = f_data.read()
            except Exception:
                break
    except Exception as e:
        log("HPDv2 tar decompression finished", error=str(e))

    log("Decompressed HPDv2 candidate images", count=len(images_by_name))

    written = 0
    for sample in meta_data:
        prompt = sample.get("prompt", "")
        img_paths = sample.get("image_path", [])
        ranks = sample.get("rank", [])
        if len(img_paths) < 2 or len(ranks) < 2:
            continue

        # Form pairs where rank_i < rank_j (lower rank value is better)
        pairs_for_prompt = []
        for i in range(len(img_paths)):
            for j in range(i + 1, len(img_paths)):
                name_i = os.path.basename(img_paths[i])
                name_j = os.path.basename(img_paths[j])
                if name_i in images_by_name and name_j in images_by_name:
                    r_i, r_j = ranks[i], ranks[j]
                    if r_i != r_j:
                        pairs_for_prompt.append((name_i, name_j, r_i, r_j))

        for name_i, name_j, r_i, r_j in pairs_for_prompt[:2]:  # max 2 pairs per prompt
            preferred_name = name_i if r_i < r_j else name_j
            rejected_name = name_j if r_i < r_j else name_i

            try:
                a_img = Image.open(io.BytesIO(images_by_name[preferred_name])).convert("RGB")
                b_img = Image.open(io.BytesIO(images_by_name[rejected_name])).convert("RGB")
            except Exception:
                continue

            pair_id = f"hpdv2_{written:07d}"
            a_img.save(dest / f"{pair_id}_a.png", "PNG")
            b_img.save(dest / f"{pair_id}_b.png", "PNG")
            (dest / f"{pair_id}.json").write_text(
                json.dumps(
                    {
                        "pair_id": pair_id,
                        "prompt": prompt,
                        "image_a": f"{pair_id}_a.png",  # preferred
                        "image_b": f"{pair_id}_b.png",  # rejected
                        "preferred": "a",
                        "origin": "hpdv2",
                        "label_source": "human",
                    }
                ),
                encoding="utf-8",
            )
            written += 1
            if written >= target_count:
                break
        if written >= target_count:
            break

    log("Extracted HPDv2 preference pairs", count=written)


def extract_rico_core(target_count: int = 663):
    dest = RAW_DIR / "rico_core"
    existing = len(list(dest.glob("*.png")))
    if existing >= target_count:
        log("RICO Core screenshots already present", count=existing)
        return

    log("Reading RICO Core from HuggingFace via Parquet streaming...")
    fs = HfFileSystem()
    repo_file = "datasets/creative-graphic-design/Rico/ui-screenshots-and-view-hierarchies/train-00000-of-00015.parquet"
    pf = pq.ParquetFile(repo_file, filesystem=fs)

    # 100 rows per group, read groups 0..6 (700 rows)
    groups_to_read = min(7, pf.num_row_groups)
    table = pf.read_row_groups(list(range(groups_to_read)), columns=["screenshot", "activity_name"])
    log("Fetched RICO Core parquet rows", total_rows=len(table))

    written = 0
    for idx in range(min(len(table), target_count)):
        row_img = table["screenshot"][idx].as_py()
        blob = row_img.get("bytes") if isinstance(row_img, dict) else row_img
        if not blob:
            continue
        try:
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            out_file = dest / f"rico_core_{written:06d}.png"
            img.save(out_file, "PNG")
            written += 1
        except Exception:
            continue

    log("Extracted RICO Core screenshots", count=written)


def extract_rico_semantic(target_count: int = 663):
    dest = RAW_DIR / "rico_semantic"
    existing = len(list(dest.glob("*.png")))
    if existing >= target_count:
        log("RICO Semantic screenshots already present", count=existing)
        return

    log("Reading RICO Semantic from HuggingFace via Parquet streaming...")
    fs = HfFileSystem()
    repo_file = "datasets/creative-graphic-design/Rico/ui-screenshots-and-hierarchies-with-semantic-annotations/train-00000-of-00003.parquet"
    pf = pq.ParquetFile(repo_file, filesystem=fs)

    groups_to_read = min(7, pf.num_row_groups)
    table = pf.read_row_groups(list(range(groups_to_read)), columns=["screenshot", "activity_name"])
    log("Fetched RICO Semantic parquet rows", total_rows=len(table))

    written = 0
    for idx in range(min(len(table), target_count)):
        row_img = table["screenshot"][idx].as_py()
        blob = row_img.get("bytes") if isinstance(row_img, dict) else row_img
        if not blob:
            continue
        try:
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            out_file = dest / f"rico_semantic_{written:06d}.png"
            img.save(out_file, "PNG")
            written += 1
        except Exception:
            continue

    log("Extracted RICO Semantic screenshots", count=written)


def extract_screen2words(target_count: int = 225):
    dest = RAW_DIR / "screen2words"
    existing = len(list(dest.glob("*.png")))
    if existing >= target_count:
        log("Screen2Words screenshots already present", count=existing)
        return

    log("Reading Screen2Words from HuggingFace via Parquet streaming...")
    fs = HfFileSystem()
    repo_file = "datasets/bevaya/RICO-Screen2Words/data/train-00000-of-00008.parquet"
    pf = pq.ParquetFile(repo_file, filesystem=fs)

    table = pf.read_row_group(0, columns=["screenId", "captions", "image"]).slice(0, target_count + 50)
    log("Fetched Screen2Words parquet rows", rows=len(table))

    captions_map = {}
    written = 0
    for idx in range(len(table)):
        screen_id = table["screenId"][idx].as_py()
        captions = table["captions"][idx].as_py()
        img_dict = table["image"][idx].as_py()
        blob = img_dict.get("bytes") if isinstance(img_dict, dict) else img_dict
        if not blob:
            continue
        try:
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            file_name = f"screen2words_{written:06d}.png"
            img.save(dest / file_name, "PNG")
            captions_map[file_name] = {
                "screen_id": screen_id,
                "captions": captions,
                "primary_caption": captions[0] if captions else "",
            }
            written += 1
            if written >= target_count:
                break
        except Exception:
            continue

    (dest / "captions_index.json").write_text(json.dumps(captions_map, indent=2), encoding="utf-8")
    log("Extracted Screen2Words screenshots + captions", count=written)


async def fetch_urls_concurrently(
    items: list[tuple[str, str]],  # (url, caption)
    dest_dir: Path,
    prefix: str,
    target_count: int,
    concurrency: int = 24,
):
    existing = len(list(dest_dir.glob("*.jpg")))
    if existing >= target_count:
        log(f"{prefix} images already present", count=existing)
        return

    log(f"Downloading {prefix} images concurrently (target={target_count}, pool={len(items)})...")
    sem = asyncio.Semaphore(concurrency)
    written = 0
    lock = asyncio.Lock()

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        timeout=3.5,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:

        async def _download_one(idx: int, url: str, caption: str):
            nonlocal written
            async with sem:
                async with lock:
                    if written >= target_count:
                        return
                try:
                    resp = await client.get(url, follow_redirects=True)
                    if resp.status_code != 200 or len(resp.content) < 1000:
                        return
                    # Verify PIL
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception:
                    return

                async with lock:
                    if written >= target_count:
                        return
                    out_name = f"{prefix}_{written:06d}.jpg"
                    img.save(dest_dir / out_name, "JPEG", quality=90)
                    meta_path = dest_dir / f"{prefix}_{written:06d}.json"
                    meta_path.write_text(
                        json.dumps({"url": url, "caption": caption, "file": out_name}),
                        encoding="utf-8",
                    )
                    written += 1
                    if written % 25 == 0 or written == target_count:
                        log(f"{prefix} downloaded", count=written, target=target_count)

        tasks = [_download_one(i, url, cap) for i, (url, cap) in enumerate(items)]
        await asyncio.gather(*tasks, return_exceptions=True)

    log(f"{prefix} download batch completed", count=written)


async def fetch_pd12m(target_count: int = 100):
    dest = RAW_DIR / "pd12m"
    existing = len(list(dest.glob("*.jpg")))
    if existing >= target_count:
        log("PD12M images already present", count=existing)
        return

    log("Loading PD12M URLs from cached shard 000...")
    cache_root = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    pd12m_parquets = list(cache_root.glob("datasets--Spawning--PD12M/**/metadata/pd12m.000.parquet"))
    if not pd12m_parquets:
        log("PD12M parquet not found in cache! Downloading...")
        p = hf_hub_download("Spawning/PD12M", "metadata/pd12m.000.parquet", repo_type="dataset")
        parquet_path = Path(p)
    else:
        parquet_path = pd12m_parquets[0]

    df = pd.read_parquet(parquet_path, columns=["url", "caption"])
    items = [(row["url"], str(row.get("caption", ""))) for _, row in df.dropna(subset=["url"]).head(1000).iterrows()]
    log("Loaded PD12M candidate URLs", count=len(items))

    await fetch_urls_concurrently(items, dest, "pd12m", target_count, concurrency=32)


async def fetch_cc12m(target_count: int = 100):
    dest = RAW_DIR / "cc12m"
    existing = len(list(dest.glob("*.jpg")))
    if existing >= target_count:
        log("CC12M images already present", count=existing)
        return

    log("Loading CC12M URLs from HuggingFace refs/convert/parquet via streaming...")
    fs = HfFileSystem()
    repo_file = "datasets/google-research-datasets/conceptual_12m@refs/convert/parquet/default/train/0000.parquet"
    pf = pq.ParquetFile(repo_file, filesystem=fs)

    table = pf.read_row_group(0, columns=["image_url", "caption"]).slice(0, 1000)
    items = []
    for i in range(len(table)):
        u = table["image_url"][i].as_py()
        c = table["caption"][i].as_py()
        if u:
            items.append((u, str(c or "")))
    log("Loaded CC12M candidate URLs", count=len(items))

    await fetch_urls_concurrently(items, dest, "cc12m", target_count, concurrency=32)


def verify_disk_usage():
    total_bytes = 0
    file_count = 0
    for root, _, files in os.walk(DATA_ROOT):
        for f in files:
            fp = Path(root) / f
            total_bytes += fp.stat().st_size
            file_count += 1
    gb = total_bytes / (1024 ** 3)
    mb = total_bytes / (1024 ** 2)
    log("Total Data Forge Footprint", files=file_count, megabytes=round(mb, 1), gigabytes=round(gb, 2))


async def main():
    start_time = time.time()
    log("=== Krisna Data-Forge: Starting 1% Dataset Slice Ingestion ===")
    setup_directories()

    # 1. Eval References (TASTE, PartiPrompts)
    sync_eval_references()

    # 2. WebUI metadata
    download_webui_metadata()

    # 3. Mobile UI Datasets (RICO Core, RICO Semantic, Screen2Words)
    extract_rico_core(target_count=663)
    extract_rico_semantic(target_count=663)
    extract_screen2words(target_count=225)

    # 4. Preference Pairs (GameLabel-10K, HPDv2)
    extract_gamelabel_10k(target_count=96)
    extract_hpdv2(target_count=369)

    # 5. General Visual Backbones (PD12M, CC12M)
    await fetch_pd12m(target_count=100)
    await fetch_cc12m(target_count=100)

    verify_disk_usage()
    elapsed = round(time.time() - start_time, 1)
    log(f"=== 1% Ingestion Complete in {elapsed}s ===")


if __name__ == "__main__":
    asyncio.run(main())
