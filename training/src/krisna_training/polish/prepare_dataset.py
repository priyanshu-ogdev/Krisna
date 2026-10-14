"""Builds a dataset directory in the standard format diffusers' dreambooth-
LoRA scripts (and DiffSynth-Studio) expect: a flat folder of images plus a
`metadata.jsonl` with `file_name`/`text` columns.

Accepts captions either as a pre-loaded dictionary {filename: caption}
or as a path to a .json / .jsonl file, parsing formats automatically.

UPGRADE (docs/review/34_model_review_3_polish_default.md): this pipeline
never used `source_caption` at all, even though data-forge's own
`_export_zimage` exports it specifically (its own comment: "a future
Z-Image fine-tune with caption conditioning would need source_caption to
apply DALL-E 3-style caption mixing the same way the sketch tier already
does"). `caption_mix_ratio` closes that gap. The SAME train/inference
distribution-matching concern that motivated Sketch's
caption_mix_ratio applies equally here: this tier is a diffusion image
generator that receives SHORT, casual user prompts at inference (via
polish_default_backend.py's effective_prompt), while every image here
was, until this fix, trained on ONLY its dense, VLM-recaptioned
description — zero exposure to short-caption style at all.

ARCHITECTURAL DIFFERENCE from Sketch's version, stated plainly: Sketch's
SketchTokenDataset re-rolls the dense-vs-source choice on every
`__getitem__` call (so a different training epoch can see a different
caption style for the same image). This tier trains through the
OFFICIAL, third-party `train_dreambooth_lora_z_image.py` script, which
has no hook for per-epoch/per-access randomization — `metadata.jsonl` is
written once, and whatever caption lands there is what every epoch of
that training run sees for that image. The mixing decision is therefore
made ONCE per image, at prepare() time, not per-epoch. This is a real,
weaker form of the same idea (less caption-style diversity across a
training run than Sketch gets), not a full equivalent — documented here
rather than silently presented as identical to Sketch's mechanism.
"""

from __future__ import annotations

import json
import hashlib
import logging
import random
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

log = logging.getLogger("krisna_training.polish.prepare_dataset")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_qwen_pairs_from_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    source_root: str | Path | None = None,
) -> Path:
    """Materialize a strict Qwen dataset from a Data-Forge pair manifest.

    The function deliberately refuses an unassigned split, missing provenance,
    duplicate pair IDs, missing files, or hash mismatches. It is the training
    side of the Data-Forge ``polish_qwen_image21/manifest.jsonl`` contract and
    keeps the source receipt in ``metadata.jsonl`` rather than reducing the
    data to filename/caption pairs.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    root = Path(source_root) if source_root is not None else manifest_path.parent
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    source_files: dict[str, tuple[Path, Path]] = {}
    seen: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Qwen manifest JSON at line {line_number}") from exc
        pair_id = str(row.get("record_id") or row.get("pair_id") or "").strip()
        if not pair_id or not re.fullmatch(r"[A-Za-z0-9._-]+", pair_id) or pair_id in seen:
            raise ValueError(f"Missing or duplicate pair ID at line {line_number}")
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"Pair {pair_id} has no source-disjoint split assignment")
        if not row.get("source_dataset") or not row.get("source_provenance"):
            raise ValueError(f"Pair {pair_id} is missing source provenance")
        condition = (root / str(row["condition_image"])).resolve()
        target = (root / str(row["target_image"])).resolve()
        try:
            condition.relative_to(root.resolve())
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Pair {pair_id} image path escapes source root") from exc
        if not condition.is_file() or not target.is_file():
            raise ValueError(f"Pair {pair_id} references a missing image")
        condition_sha = _sha256_file(condition)
        target_sha = _sha256_file(target)
        if condition_sha == target_sha:
            raise ValueError(f"Pair {pair_id} condition and target are identical")
        if row.get("condition_sha256") != condition_sha or row.get("target_sha256") != target_sha:
            raise ValueError(f"Pair {pair_id} image hash receipt does not match bytes")
        condition_name = f"condition_{pair_id}{condition.suffix.lower() or '.png'}"
        target_name = f"target_{pair_id}{target.suffix.lower() or '.png'}"
        source_group = row.get("source_group") or row.get("source_group_id") or row.get("group")
        rows.append({
            "pair_id": pair_id,
            "image": target_name,
            "condition_image": condition_name,
            "caption": row.get("caption") or row.get("prompt") or "",
            "split": row["split"],
            "source_dataset": row["source_dataset"],
            "source_group": source_group,
            "source_revision": row.get("source_revision"),
            "source_provenance": row["source_provenance"],
            "condition_sha256": condition_sha,
            "target_sha256": target_sha,
        })
        source_files[pair_id] = (condition, target)
        seen.add(pair_id)
    if not rows:
        raise ValueError(f"No Qwen pairs found in {manifest_path}")
    # A group is the strongest available split identity.  If the upstream
    # manifest has no group, dataset identity is the conservative fallback:
    # one source dataset may not straddle train/validation/test.
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group = str(row.get("source_group") or f"dataset:{row['source_dataset']}")
        group_splits.setdefault(group, set()).add(row["split"])
    overlap = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    if overlap:
        raise ValueError(f"source groups cross dataset splits: {list(overlap)[:3]}")

    # datasets is intentionally required here: JSON metadata alone is not a
    # native typed two-image dataset and would let the trainer guess a schema.
    try:
        from datasets import Dataset, Features, Image, Value
    except ImportError as exc:
        raise RuntimeError(
            "Qwen preparation requires the 'datasets' and 'pyarrow' packages; "
            "install the pinned training environment before preparing data"
        ) from exc
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Qwen preparation requires pyarrow for split-isolated Parquet output"
        ) from exc

    split_counts = {split: sum(row["split"] == split for row in rows)
                    for split in ("train", "validation", "test")}
    if split_counts["train"] <= 0:
        raise ValueError("Qwen dataset must contain at least one train pair")
    columns = ["image", "condition_image", "caption", "pair_id", "split",
               "source_dataset", "source_revision", "source_provenance", "source_group",
               "condition_sha256", "target_sha256"]
    features = Features({
        "image": Image(), "condition_image": Image(), "caption": Value("string"),
        "pair_id": Value("string"), "split": Value("string"),
        "source_dataset": Value("string"), "source_revision": Value("string"),
        "source_provenance": Value("string"), "source_group": Value("string"),
        "condition_sha256": Value("string"), "target_sha256": Value("string"),
    })
    with TemporaryDirectory(prefix="qwen-pairs-") as staging_name:
        staging = Path(staging_name)
        for split in ("train", "validation", "test"):
            split_rows = [row for row in rows if row["split"] == split]
            if not split_rows:
                stale = output_dir / split
                if stale.exists():
                    shutil.rmtree(stale)
                continue
            split_dir = staging / split
            split_dir.mkdir(parents=True)
            serializable = []
            for row in split_rows:
                condition, target = source_files[row["pair_id"]]
                serializable.append({
                    "image": {"bytes": target.read_bytes(), "path": None},
                    "condition_image": {"bytes": condition.read_bytes(), "path": None},
                    "caption": row["caption"], "pair_id": row["pair_id"],
                    "split": row["split"], "source_dataset": row["source_dataset"],
                    "source_revision": str(row.get("source_revision") or ""),
                    "source_provenance": json.dumps(row["source_provenance"], sort_keys=True),
                    "source_group": str(row.get("source_group") or ""),
                    "condition_sha256": row["condition_sha256"],
                    "target_sha256": row["target_sha256"],
                })
            Dataset.from_list(serializable, features=features).to_parquet(str(split_dir / "data.parquet"))

        # Keep the historical root file as a train-only compatibility index;
        # no validation/test row is ever mixed into it.
        train_rows = [row for row in rows if row["split"] == "train"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in train_rows),
            encoding="utf-8",
        )
        for split in ("train", "validation", "test"):
            split_rows = [row for row in rows if row["split"] == split]
            if split_rows:
                split_dir = output_dir / split
                split_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staging / split / "data.parquet", split_dir / "data.parquet")
                (split_dir / "metadata.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in split_rows),
                    encoding="utf-8",
                )
                for row in split_rows:
                    condition, target = source_files[row["pair_id"]]
                    shutil.copy2(condition, split_dir / row["condition_image"])
                    shutil.copy2(target, split_dir / row["image"])

        def sha(path: Path) -> str:
            return _sha256_file(path)
        metadata_hashes = {
            split: sha(output_dir / split / "metadata.jsonl")
            for split in ("train", "validation", "test")
            if (output_dir / split / "metadata.jsonl").exists()
        }
        parquet_hashes = {
            split: sha(output_dir / split / "data.parquet")
            for split in ("train", "validation", "test")
            if (output_dir / split / "data.parquet").exists()
        }
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
    (output_dir / "dataset_receipt.json").write_text(
        json.dumps({
            "schema_version": 2,
            "manifest_path": str(manifest_path),
            "pair_count": len(rows),
            "split_counts": counts,
            "manifest_sha256": _sha256_file(manifest_path),
            "columns": columns,
            "metadata_sha256": metadata_hashes,
            "parquet_sha256": parquet_hashes,
            "image_hashes": {row["pair_id"]: {
                "condition": row["condition_sha256"], "target": row["target_sha256"]
            } for row in rows},
            "source_split_policy": "group disjoint; source_dataset fallback when group is absent",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def prepare_image_pairs(condition_dir: str | Path, target_dir: str | Path,
                        output_dir: str | Path, captions: dict | str | Path | None = None) -> Path:
    """Prepare the official Qwen img2img schema.

    Pairing is filename-based and strict: silently using a target as its own
    condition would train identity copying, so missing or ambiguous pairs fail.
    """
    condition_dir, target_dir, output_dir = map(Path, (condition_dir, target_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    dense, _ = load_captions(captions)
    conditions = {p.name: p for p in condition_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    targets = {p.name: p for p in target_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    if not targets:
        raise ValueError(f"No target images found under {target_dir}")
    missing = sorted(set(targets) - set(conditions))
    if missing:
        raise ValueError(f"Missing condition images for {len(missing)} targets (first: {missing[:3]})")
    metadata = output_dir / "metadata.jsonl"
    with metadata.open("w", encoding="utf-8") as handle:
        for name in sorted(targets):
            target_name = f"target_{name}"
            condition_name = f"condition_{name}"
            shutil.copy2(targets[name], output_dir / target_name)
            shutil.copy2(conditions[name], output_dir / condition_name)
            caption = dense.get(name, "Refine this design into a polished production render.")
            handle.write(json.dumps({"image": target_name, "condition_image": condition_name,
                                     "caption": caption}) + "\n")
    return output_dir


def load_captions(
    captions_input: dict[str, str] | dict[str, dict] | str | Path | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Parse captions from dict, JSON file, or JSONL file.

    Returns (dense_captions, source_captions) — both {filename: caption},
    source_captions only populated when the input actually carries a
    `source_caption` field.

    The dict-input form now accepts TWO shapes per value, auto-detected:
    a plain string (`{"a.png": "a caption"}` — the original, simplest
    calling convention, no source_caption possible) or a nested dict
    (`{"a.png": {"caption": "...", "source_caption": "..."}}` — BUG FIX,
    docs/review/34_model_review_3_polish_default.md: sync_polish_default.py
    used to build a plain `{filename: caption_string}` dict itself,
    silently discarding source_caption before ever reaching this
    function — meaning this whole caption-mixing feature was completely
    inert for the real data-forge pipeline, the ONLY real production
    caller of this module, even though it worked correctly for the
    JSONL-file-path calling convention tested directly against this
    function. sync_polish_default.py was fixed to pass the richer nested
    shape instead of pre-flattening to strings.
    """
    if captions_input is None:
        return {}, {}
    if isinstance(captions_input, dict):
        dense: dict[str, str] = {}
        source: dict[str, str] = {}
        for fname, value in captions_input.items():
            if isinstance(value, dict):
                dense[fname] = value.get("caption", "")
                if value.get("source_caption"):
                    source[fname] = value["source_caption"]
            else:
                dense[fname] = value
        return dense, source

    cpath = Path(captions_input)
    if not cpath.exists():
        log.warning("captions_file_not_found", extra={"path": str(cpath)})
        return {}, {}

    raw = cpath.read_text(encoding="utf-8").strip()
    if not raw:
        return {}, {}

    captions: dict[str, str] = {}
    source_captions: dict[str, str] = {}
    if cpath.suffix.lower() == ".jsonl" or ("\n" in raw and not raw.startswith("[")):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            fname = rec.get("image_filename") or rec.get("filename") or rec.get("source_image") or rec.get("record_id")
            if fname:
                name = Path(fname).name
                captions[name] = rec.get("caption", "")
                if rec.get("source_caption"):
                    source_captions[name] = rec["source_caption"]
    else:
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                captions = loaded
            elif isinstance(loaded, list):
                for item in loaded:
                    if isinstance(item, dict):
                        fname = item.get("image_filename") or item.get("filename") or item.get("source_image")
                        if fname:
                            name = Path(fname).name
                            captions[name] = item.get("caption", "")
                            if item.get("source_caption"):
                                source_captions[name] = item["source_caption"]
        except Exception as e:
            log.warning("captions_json_parse_failed", extra={"path": str(cpath), "error": str(e)})

    return captions, source_captions


def prepare(
    image_dir: str | Path,
    output_dir: str | Path,
    captions: dict[str, str] | str | Path | None = None,
    use_shared_instance_prompt: str | None = None,
    # UPGRADE (docs/review/34_model_review_3_polish_default.md): matches
    # Sketch tier's default exactly (SketchTokenDataset, TrainConfig,
    # all four sketch YAML configs) and the same Betker et al. 2023
    # (DALL-E 3) sweep finding that motivates it — see this module's
    # docstring for the full reasoning and the one real architectural
    # difference (resolved once per image here, not per-epoch).
    caption_mix_ratio: float = 0.95,
    seed: int | None = None,
    image_allowlist: set[str] | None = None,
) -> Path:
    """image_dir: source images. output_dir: where the prepared dataset is
    written (a flat copy of the images + metadata.jsonl). captions: either
    maps source filename -> caption text, or points to a JSON / JSONL file.
    Images without an entry fall back to `use_shared_instance_prompt` if given.

    Returns output_dir.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    captions_dict, source_captions_dict = load_captions(captions)
    rng = random.Random(seed)  # seedable for reproducible dataset builds, unseeded (system entropy) by default

    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if image_allowlist is not None:
        if any(Path(name).name != name or name in {".", ".."} for name in image_allowlist):
            raise ValueError("Image allowlist entries must be plain filenames")
        by_name: dict[str, list[Path]] = {}
        for path in image_paths:
            by_name.setdefault(path.name, []).append(path)
        duplicate_names = [
            name for name, matches in by_name.items()
            if len(matches) > 1 and name in image_allowlist
        ]
        if duplicate_names:
            raise ValueError(f"Ambiguous image filenames in allowlisted export: {duplicate_names[:5]}")
        missing = image_allowlist - by_name.keys()
        if missing:
            raise ValueError(f"Allowlisted training images are missing: {sorted(missing)[:5]}")
        image_paths = sorted(by_name[name][0] for name in image_allowlist)
    if not image_paths:
        raise ValueError(f"No images found under {image_dir}")

    metadata_path = output_dir / "metadata.jsonl"
    written = 0
    mixed_to_source = 0
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        for image_path in image_paths:
            dest = output_dir / image_path.name
            if not dest.exists() or dest.resolve() != image_path.resolve():
                shutil.copy2(image_path, dest)

            dense_caption = captions_dict.get(image_path.name)
            source_caption = source_captions_dict.get(image_path.name)
            if dense_caption is None:
                caption = use_shared_instance_prompt or ""
            elif source_caption and rng.random() >= caption_mix_ratio:
                caption = source_caption
                mixed_to_source += 1
            else:
                caption = dense_caption

            meta_file.write(json.dumps({"file_name": image_path.name, "text": caption}) + "\n")
            written += 1

    log.info(
        "polish_prepare_dataset_complete",
        extra={"written": written, "mixed_to_source_caption": mixed_to_source,
               "caption_mix_ratio": caption_mix_ratio},
    )
    return output_dir


def count_prepared(output_dir: str | Path) -> int:
    metadata_path = Path(output_dir) / "metadata.jsonl"
    if not metadata_path.exists():
        return 0
    with metadata_path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# Alias for backward compatibility
prepare_dataset = prepare
