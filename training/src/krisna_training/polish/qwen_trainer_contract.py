"""Contract and launch-preflight helpers for the pinned native Qwen trainer.

The repository config contains provenance and research metadata in addition to
the arguments understood by Diffusers.  This module is the single boundary
that validates the metadata and converts the pinned revision field to the
official trainer's ``--revision`` argument.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
PIPELINE_CLASS = "QwenImage21Pipeline"
CONDITIONING_CONTRACT = "image_conditioned_edit"
DIFFUSERS_REVISION = "efe46a5b177af67c64b3666da8ddddac7ee12163"
TRAINER_FILENAME = "train_dreambooth_lora_qwenimage21_img2img.py"

SUPPORTED_OPTIONS = {
    "pretrained_model_name_or_path",
    "revision",
    "dataset_name",
    "dataset_config_name",
    "instance_data_dir",
    "cache_dir",
    "image_column",
    "cond_image_column",
    "caption_column",
    "repeats",
    "class_data_dir",
    "instance_prompt",
    "class_prompt",
    "validation_image",
    "validation_num_inference_steps",
    "validation_prompt",
    "skip_final_inference",
    "final_validation_prompt",
    "num_validation_images",
    "validation_epochs",
    "rank",
    "lora_alpha",
    "lora_dropout",
    "with_prior_preservation",
    "prior_loss_weight",
    "num_class_images",
    "output_dir",
    "seed",
    "resolution",
    "aspect_ratio_buckets",
    "use_aspect_ratio_buckets",
    "center_crop",
    "random_flip",
    "caption_dropout",
    "train_batch_size",
    "sample_batch_size",
    "num_train_epochs",
    "max_train_steps",
    "checkpointing_steps",
    "checkpoints_total_limit",
    "resume_from_checkpoint",
    "gradient_accumulation_steps",
    "gradient_checkpointing",
    "learning_rate",
    "scale_lr",
    "lr_scheduler",
    "lr_warmup_steps",
    "lr_num_cycles",
    "lr_power",
    "dataloader_num_workers",
    "weighting_scheme",
    "logit_mean",
    "logit_std",
    "mode_scale",
    "optimizer",
    "use_8bit_adam",
    "adam_beta1",
    "adam_beta2",
    "prodigy_beta3",
    "prodigy_decouple",
    "adam_weight_decay",
    "lora_layers",
    "adam_epsilon",
    "prodigy_use_bias_correction",
    "prodigy_safeguard_warmup",
    "max_grad_norm",
    "push_to_hub",
    "hub_token",
    "hub_model_id",
    "logging_dir",
    "allow_tf32",
    "cache_latents",
    "report_to",
    "mixed_precision",
    "upcast_before_saving",
    "offload",
}

METADATA_ONLY_KEYS = {
    "pretrained_model_revision",
    "pipeline_class",
    "conditioning_contract",
    "base_weights_frozen",
    "val_fraction",
    "val_every",
    "use_dora",
    "load_in_4bit",
    "bnb_4bit_quant_type",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_use_double_quant",
}

SUPPORTED_SCHEDULERS = {
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
}


def prepare_trainer_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate repository config and return only official trainer options."""
    cfg = dict(raw)
    diffusers_dir = cfg.pop("diffusers_dir", None)
    if not diffusers_dir:
        raise ValueError("diffusers_dir is required by the launcher")

    if cfg.get("pretrained_model_name_or_path") != MODEL_ID:
        raise ValueError(f"expected pretrained_model_name_or_path={MODEL_ID!r}")
    if cfg.pop("pretrained_model_revision", None) != MODEL_REVISION:
        raise ValueError(f"expected pinned Qwen revision {MODEL_REVISION}")
    if cfg.pop("pipeline_class", None) != PIPELINE_CLASS:
        raise ValueError(f"expected pipeline_class={PIPELINE_CLASS!r}")
    if cfg.pop("conditioning_contract", None) != CONDITIONING_CONTRACT:
        raise ValueError(f"expected conditioning_contract={CONDITIONING_CONTRACT!r}")
    if cfg.pop("base_weights_frozen", None) is not True:
        raise ValueError("base_weights_frozen must be true")
    if "use_dora" in cfg:
        if not isinstance(cfg["use_dora"], bool):
            raise ValueError("use_dora must be a boolean")
        cfg.pop("use_dora")
    if "load_in_4bit" in cfg:
        if not isinstance(cfg["load_in_4bit"], bool):
            raise ValueError("load_in_4bit must be a boolean")
        if cfg["load_in_4bit"]:
            quant_type = cfg.get("bnb_4bit_quant_type", "nf4")
            if quant_type not in ("nf4", "fp4"):
                raise ValueError(f"unsupported bnb_4bit_quant_type {quant_type!r}; choose 'nf4' or 'fp4'")
            compute_dtype = cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            if compute_dtype not in ("bfloat16", "float16"):
                raise ValueError(f"unsupported bnb_4bit_compute_dtype {compute_dtype!r}; choose 'bfloat16' or 'float16'")
            if "bnb_4bit_use_double_quant" in cfg and not isinstance(cfg["bnb_4bit_use_double_quant"], bool):
                raise ValueError("bnb_4bit_use_double_quant must be a boolean")
        cfg.pop("load_in_4bit", None)
        cfg.pop("bnb_4bit_quant_type", None)
        cfg.pop("bnb_4bit_compute_dtype", None)
        cfg.pop("bnb_4bit_use_double_quant", None)
    for unsupported in ("val_fraction", "val_every"):
        if unsupported in cfg and cfg[unsupported] is not None:
            raise ValueError(
                f"{unsupported} is unsupported by the native Diffusers trainer; "
                "put held-out records in the prepared validation/test splits instead"
            )
        cfg.pop(unsupported, None)
    cfg["revision"] = MODEL_REVISION

    if not str(cfg.get("instance_prompt") or "").strip():
        raise ValueError("instance_prompt is required by the official trainer")
    if cfg.get("validation_prompt") and not cfg.get("validation_image"):
        raise ValueError("validation_image is required when validation_prompt is configured")
    if cfg.get("lr_scheduler") not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"unsupported lr_scheduler {cfg.get('lr_scheduler')!r}; "
            f"choose one of {sorted(SUPPORTED_SCHEDULERS)}"
        )

    if not str(cfg.get("cond_image_column") or "").strip():
        raise ValueError("cond_image_column is required for native image conditioning")
    if not str(cfg.get("image_column") or "").strip():
        raise ValueError("image_column is required for native image conditioning")
    if cfg["cond_image_column"] == cfg["image_column"]:
        raise ValueError("condition and target columns must be different")
    if cfg.get("random_flip") is not False:
        raise ValueError("random_flip must be false: flipping changes UI text")
    for key in ("rank", "lora_alpha", "max_train_steps", "resolution"):
        if key in cfg and (not isinstance(cfg[key], (int, float)) or cfg[key] <= 0):
            raise ValueError(f"{key} must be positive")
    if cfg.get("mixed_precision") != "bf16":
        raise ValueError("mixed_precision must be bf16 for the pinned Qwen trainer")
    if cfg.get("use_aspect_ratio_buckets") is not True:
        raise ValueError("use_aspect_ratio_buckets must be true and explicit")
    buckets = str(cfg.get("aspect_ratio_buckets") or "")
    if not buckets:
        raise ValueError("aspect_ratio_buckets must be explicitly configured")
    for bucket in buckets.split(";"):
        parts = bucket.split(",")
        if len(parts) != 2:
            raise ValueError("aspect_ratio_buckets must use 'height,width;...' format")
        try:
            height, width = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("aspect_ratio_buckets values must be integers") from exc
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("aspect_ratio_buckets dimensions must be positive multiples of 32")

    unknown = sorted(set(cfg) - SUPPORTED_OPTIONS)
    if unknown:
        raise ValueError(f"unsupported Diffusers trainer config keys: {', '.join(unknown)}")
    return cfg


def trainer_script(diffusers_dir: str | Path) -> Path:
    return Path(diffusers_dir) / "examples" / "dreambooth" / TRAINER_FILENAME


def verify_diffusers_checkout(diffusers_dir: str | Path) -> tuple[Path, bool]:
    """Verify the external checkout without changing it or importing GPU code.

    Dirty changes are deliberately allowed: local trainer fixes are part of the
    reproducibility receipt, but the commit underneath them must remain pinned.
    """
    root = Path(diffusers_dir)
    if not (root / ".git").exists():
        raise ValueError(f"Diffusers checkout is missing at {root}; run the pinned setup script")
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True, stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot inspect Diffusers checkout {root}: {exc}") from exc
    if head != DIFFUSERS_REVISION:
        raise ValueError(f"Diffusers HEAD {head} does not match pinned {DIFFUSERS_REVISION}")
    script = trainer_script(root)
    if not script.is_file():
        raise ValueError(f"native Qwen trainer is missing: {script}")
    source = script.read_text(encoding="utf-8", errors="replace")
    required_markers = ("QwenImage21Pipeline", "cond_image_column", "image_column")
    missing = [marker for marker in required_markers if marker not in source]
    if missing:
        raise ValueError(f"trainer is not the native typed Qwen img2img contract; missing {missing}")
    return script, bool(status.strip())


def build_accelerate_command(config: dict[str, Any], diffusers_dir: str | Path) -> list[str]:
    """Build the exact native command, with no subprocess side effects."""
    prepared = prepare_trainer_config(config)
    script = trainer_script(diffusers_dir)
    args = ["accelerate", "launch", str(script)]
    for key, value in prepared.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}={value}")
    return args


def format_command(args: list[str]) -> str:
    """Render a command safely; secrets never appear in launch logs."""
    rendered: list[str] = []
    for arg in args:
        if arg.startswith("--hub_token="):
            arg = "--hub_token=<redacted>"
        rendered.append(shlex.quote(arg))
    return " ".join(rendered)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_prepared_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Audit the split-isolated typed dataset before any expensive launch."""
    root = Path(dataset_dir)
    receipt_path = root / "dataset_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"dataset receipt is missing: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"dataset receipt cannot be read: {receipt_path}") from exc
    if receipt.get("schema_version") != 2:
        raise ValueError("dataset receipt schema_version 2 is required")
    manifest_path = Path(str(receipt.get("manifest_path") or ""))
    if not manifest_path.is_file():
        candidate = root / manifest_path
        manifest_path = candidate if candidate.is_file() else manifest_path
    if not manifest_path.is_file() or _sha256(manifest_path) != receipt.get("manifest_sha256"):
        raise ValueError("dataset manifest receipt hash does not match the manifest bytes")
    expected_columns = receipt.get("columns")
    if expected_columns != ["image", "condition_image", "caption", "pair_id", "split",
                            "source_dataset", "source_revision", "source_provenance", "source_group",
                            "condition_sha256", "target_sha256"]:
        raise ValueError("dataset receipt has unexpected native paired columns")
    counts = receipt.get("split_counts")
    if not isinstance(counts, dict) or int(counts.get("train", 0)) <= 0:
        raise ValueError("dataset receipt must contain train > 0")
    if receipt.get("pair_count") != sum(int(counts.get(split, 0)) for split in ("train", "validation", "test")):
        raise ValueError("dataset receipt pair_count does not match split counts")
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise ValueError("preflight requires pyarrow to audit typed Parquet datasets") from exc

    seen_groups: dict[str, str] = {}
    seen_pairs: set[str] = set()
    for split in ("train", "validation", "test"):
        expected = int(counts.get(split, 0))
        split_dir = root / split
        metadata_path = split_dir / "metadata.jsonl"
        parquet_path = split_dir / "data.parquet"
        if expected == 0:
            if metadata_path.exists() or parquet_path.exists():
                raise ValueError(f"empty split {split} has unexpected files")
            continue
        if not metadata_path.is_file() or not parquet_path.is_file():
            raise ValueError(f"split {split} is missing metadata.jsonl or data.parquet")
        if _sha256(metadata_path) != receipt.get("metadata_sha256", {}).get(split):
            raise ValueError(f"metadata hash mismatch for split {split}")
        if _sha256(parquet_path) != receipt.get("parquet_sha256", {}).get(split):
            raise ValueError(f"Parquet hash mismatch for split {split}")
        rows = []
        for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {split} metadata at line {line_number}") from exc
            rows.append(row)
        if len(rows) != expected:
            raise ValueError(f"{split} metadata count does not match receipt")
        parquet_file = parquet.ParquetFile(parquet_path)
        parquet_rows = parquet_file.metadata.num_rows
        parquet_schema = parquet_file.schema_arrow
        parquet_file.close()
        if parquet_rows != expected:
            raise ValueError(f"{split} Parquet row count does not match receipt")
        if set(parquet_schema.names) != set(expected_columns):
            raise ValueError(f"{split} Parquet columns are not the native paired schema")
        for image_column in ("image", "condition_image"):
            field = parquet_schema.field(image_column)
            if not getattr(field.type, "names", None) or set(field.type.names) != {"bytes", "path"}:
                raise ValueError(f"{split} {image_column} is not a typed datasets.Image feature")
        for row in rows:
            pair_id = str(row.get("pair_id") or "")
            if not pair_id or pair_id in seen_pairs or row.get("split") != split:
                raise ValueError(f"invalid or duplicate pair/split in {split}")
            seen_pairs.add(pair_id)
            group = str(row.get("source_group") or f"dataset:{row.get('source_dataset') or ''}")
            prior = seen_groups.setdefault(group, split)
            if prior != split:
                raise ValueError(f"source group {group!r} crosses splits")
            for column in ("image", "condition_image"):
                image_path = split_dir / str(row.get(column) or "")
                if not image_path.is_file() or _sha256(image_path) != (
                    row.get("target_sha256") if column == "image" else row.get("condition_sha256")
                ):
                    raise ValueError(f"{split} {column} path/hash audit failed for {pair_id}")
            hashes = receipt.get("image_hashes", {}).get(pair_id, {})
            if hashes.get("condition") != row.get("condition_sha256") or hashes.get("target") != row.get("target_sha256"):
                raise ValueError(f"image hash receipt is incomplete for {pair_id}")
        # Check the encoded bytes that the native datasets.Image columns expose,
        # not just the sidecar copies used for the audit receipt.  Batch reads
        # keep this bounded for large corpora.
        encoded_file = parquet.ParquetFile(parquet_path)
        rows_by_pair = {item["pair_id"]: item for item in rows}
        try:
            for batch in encoded_file.iter_batches(
                columns=["pair_id", "image", "condition_image"], batch_size=64
            ):
                for record in batch.to_pylist():
                    pair_id = str(record.get("pair_id") or "")
                    image = record.get("image") or {}
                    condition = record.get("condition_image") or {}
                    if not image.get("bytes") or not condition.get("bytes"):
                        raise ValueError(f"encoded Image bytes are missing for {pair_id}")
                    row = rows_by_pair.get(pair_id)
                    if row is None:
                        raise ValueError(f"encoded Parquet contains unknown pair {pair_id}")
                    if hashlib.sha256(image["bytes"]).hexdigest() != row["target_sha256"]:
                        raise ValueError(f"encoded target image hash mismatch for {pair_id}")
                    if hashlib.sha256(condition["bytes"]).hexdigest() != row["condition_sha256"]:
                        raise ValueError(f"encoded condition image hash mismatch for {pair_id}")
        finally:
            encoded_file.close()
    return {"receipt": receipt, "split_counts": counts, "columns": expected_columns}
