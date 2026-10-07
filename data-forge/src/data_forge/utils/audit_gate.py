"""Shared eligibility and provenance checks for audited training exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.manifest import Manifest, ManifestRecord
from data_forge.utils.path_safety import resolve_data_path


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _to_primitive(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(item) for item in value]
    return value


def pipeline_config_fingerprint(config: PipelineConfig) -> str:
    data = {
        "version": config.version,
        "dataset_version_prefix": config.dataset_version_prefix,
        "chunk_size": config.chunk_size,
        "max_retries_per_record": config.max_retries_per_record,
        "fail_fast": config.fail_fast,
        "stages": _to_primitive(config.stages),
        "models": _to_primitive(config.models),
        "encoders": _to_primitive(config.encoders),
        "datasets": _to_primitive(config.datasets),
        "paths": _to_primitive(config.paths),
        "storage": _to_primitive(config.storage),
        "vllm_server": _to_primitive(config.vllm_server),
    }
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )
    package_root = Path(__file__).resolve().parents[1]
    for root in (package_root, config.prompts_dir, config.schemas_dir):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py") if root == package_root else root.rglob("*")):
            if not path.is_file():
                continue
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def eligible_training_records(
    manifest: Manifest, data_root: Path
) -> list[ManifestRecord]:
    records = []
    for record in manifest.get_training_pool():
        if (
            record.license_verified is not True
            or record.pii_scrubbed is not True
            or record.safety_tier != "safe"
            or not record.caption
            or not record.scrubbed_image_path
        ):
            continue
        try:
            image_path = resolve_data_path(data_root, record.scrubbed_image_path)
        except (OSError, ValueError):
            continue
        if image_path.is_file():
            records.append(record)
    return records


def training_pool_fingerprint(
    records: list[ManifestRecord], data_root: Path
) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.id):
        image_path = resolve_data_path(data_root, record.scrubbed_image_path or "")
        stat = image_path.stat()
        fields = (
            record.id,
            record.updated_at,
            record.caption or "",
            record.source_caption or "",
            record.source_dataset,
            record.domain or "",
            str(image_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        )
        digest.update("\0".join(fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
