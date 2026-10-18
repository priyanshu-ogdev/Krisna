"""Explicit adapters for local UI-artifact sources.

The adapters in this module deliberately operate only when a dataset opts in
through an explicit ``download_mode``.  They do not change the default remote
fetch policy.  Both adapters accept the two layouts encountered in local
exports: image files with JSON sidecars and parquet files containing embedded
images.

Records emitted here retain the source metadata and structure verbatim while
also assigning a stable identity from source coordinates.  The hash is always
computed from the bytes written to the canonical destination file, rather than
from a source blob that may have been decoded or re-encoded on the way there.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
ADAPTER_MODES = frozenset(
    {
        "figma2code",
        "figma2code_local",
        "local_figma2code",
        "mobileviews",
        "mobileviews_local",
        "local_mobileviews",
    }
)


def canonical_json(value: Any) -> str:
    """Serialize source identity material deterministically."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def iter_parquet_rows(
    parquet_files: list[Path],
    *,
    columns: list[str] | None = None,
    batch_size: int = 512,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield parquet rows in bounded record batches.

    ``pyarrow`` is used directly so a large source is never materialized as a
    full pandas frame.  The small pandas fallback keeps local development
    usable when pyarrow is unavailable, but still yields rows in bounded
    slices to callers.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    emitted = 0
    for parquet_file in parquet_files:
        if limit is not None and emitted >= limit:
            return
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(parquet_file)
            names = parquet.schema_arrow.names
            selected = [column for column in (columns or names) if column in names]
            for batch in parquet.iter_batches(batch_size=batch_size, columns=selected or None):
                rows = batch.to_pylist()
                for row in rows:
                    if limit is not None and emitted >= limit:
                        return
                    emitted += 1
                    yield dict(row)
        except ImportError:
            import pandas as pd

            frame = pd.read_parquet(parquet_file, columns=columns)
            for start in range(0, len(frame), batch_size):
                for row in frame.iloc[start : start + batch_size].to_dict(orient="records"):
                    if limit is not None and emitted >= limit:
                        return
                    emitted += 1
                    yield row


def _first_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value is not None and value != "":
            return value
    return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {"value": value}


def _image_blob(value: Any) -> bytes | None:
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("data") or value.get("image")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return None


def _json_safe_metadata(value: Any) -> Any:
    """Drop binary payloads from retained metadata without changing structure."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, dict):
        return {
            key: _json_safe_metadata(item)
            for key, item in value.items()
            if not (key in {"bytes", "data"} and isinstance(item, (bytes, bytearray, memoryview)))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    return value


def _parse_json_value(value: Any) -> Any:
    """Decode JSON-string columns while keeping malformed values lossless."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


class _LocalArtifactAdapter:
    """Shared implementation for explicit local source adapters."""

    adapter_name = "local"
    default_image_columns = ("image", "screenshot", "png", "jpg", "img")

    def __init__(self, *, key: str, spec: Any, source_root: Path, output_root: Path, data_root: Path) -> None:
        self.key = key
        self.spec = spec
        self.source_root = source_root
        self.output_root = output_root
        self.data_root = data_root
        self.fetch_config = spec.fetch_config
        self.batch_size = max(1, int(self.fetch_config.get("parquet_batch_size", 512)))

    def fetch(self) -> list[dict[str, Any]]:
        if not self.source_root.exists() or not self.source_root.is_dir():
            return []
        records = self._fetch_parquet() if list(self.source_root.rglob("*.parquet")) else self._fetch_images()
        sample_size = self.fetch_config.get("sample_size")
        if sample_size is not None:
            records = records[: max(0, int(sample_size))]
        return records

    def _fetch_images(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        image_files = sorted(
            path for path in self.source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_file in image_files:
            metadata = _load_json(image_file.with_suffix(".json"))
            records.append(self._save_image(image_file.read_bytes(), image_file, metadata))
        return records

    def _fetch_parquet(self) -> list[dict[str, Any]]:
        parquet_files = sorted(self.source_root.rglob("*.parquet"))
        columns = self.fetch_config.get("parquet_columns")
        first_row = next(iter_parquet_rows(parquet_files, columns=columns,
                                           batch_size=self.batch_size, limit=1), None)
        if first_row is None:
            return []
        image_column = self.fetch_config.get("image_column")
        if not image_column:
            image_column = next(
                (column for column in self.default_image_columns if _image_blob(first_row.get(column))),
                next((column for column, value in first_row.items() if _image_blob(value)), None),
            )
        if not image_column:
            return []

        records: list[dict[str, Any]] = []
        global_index = 0
        sample_size = self.fetch_config.get("sample_size")
        for parquet_file in parquet_files:
            if sample_size is not None and len(records) >= int(sample_size):
                break
            remaining = None if sample_size is None else max(0, int(sample_size) - len(records))
            for _row_index, row in enumerate(
                iter_parquet_rows([parquet_file], columns=columns,
                                  batch_size=self.batch_size, limit=remaining)
            ):
                row = dict(row)
                parquet_rel = parquet_file.relative_to(self.source_root).as_posix()
                row.setdefault("source_split", "test" if "/data_test/" in f"/{parquet_rel}" else "rest")
                row.setdefault("source_parquet", parquet_rel)
                blob = _image_blob(row.get(image_column))
                if not blob:
                    continue
                records.append(self._save_image(blob, Path(f"row-{global_index:08d}.png"), row))
                global_index += 1
        return records

    def _save_image(self, blob: bytes, source_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        source_rel = source_path.relative_to(self.source_root).as_posix() if source_path.is_absolute() and self.source_root in source_path.parents else source_path.as_posix()
        source_record_id = str(
            _first_value(metadata, ("source_record_id", "record_id", "node_id", "image_id", "screen_id", "id", "uid"))
            or source_rel
        )
        parent_group = Path(source_rel).parent.as_posix()
        if parent_group == ".":
            parent_group = ""
        group_id = str(
            _first_value(metadata, ("source_group_id", "group_id", "design_id", "file_id", "filekey", "file_key", "app_id", "package_name", "package"))
            or parent_group
            or source_record_id
        )
        revision = self.fetch_config.get("source_revision") or getattr(self.spec, "revision", None)
        if str(self.fetch_config.get("download_mode", "")).endswith("_local"):
            revision = self.fetch_config.get("local_revision") or "local-unpinned"
        identity_material = {
            "adapter": self.adapter_name,
            "dataset": self.key,
            "source_record_id": source_record_id,
            "source_group_id": group_id,
            "source_revision": revision,
        }
        canonical_id = hashlib.sha256(canonical_json(identity_material).encode("utf-8")).hexdigest()

        suffix = source_path.suffix.lower() if source_path.suffix.lower() in IMAGE_EXTENSIONS else ".png"
        image_dir = self.output_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        destination = image_dir / f"{canonical_id}{suffix}"
        temporary = destination.with_name(f".{destination.name}.part")
        temporary.write_bytes(blob)
        os.replace(temporary, destination)

        try:
            from PIL import Image

            with Image.open(destination) as image:
                image.verify()
                width, height = image.size
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        try:
            relative = str(destination.relative_to(self.data_root))
        except ValueError:
            relative = str(destination)

        metadata = _json_safe_metadata(metadata)
        metadata = {
            key: _parse_json_value(value)
            for key, value in metadata.items()
            if not key.startswith("_")
        }
        # Figma2Code keeps the structured tree in JSON-string columns. MobileViews
        # stores the view hierarchy in json_content. Decode those fields before
        # retaining them so downstream consumers can validate or normalize them
        # without another source-specific parser.
        for key in ("raw_metadata", "processed_metadata", "json_content", "structure", "layout"):
            if key in metadata:
                metadata[key] = _parse_json_value(metadata[key])
        # Some Figma exports place the file/node identity in a nested metadata
        # object. Promote only identity fields to the canonical record; retain
        # the complete nested object in source_metadata.
        for nested_key in ("raw_metadata", "processed_metadata", "json_content"):
            nested = metadata.get(nested_key)
            if isinstance(nested, dict):
                for identity_key in ("filekey", "file_key", "file_id", "node_id", "app_id", "package_name", "package", "screen_id", "state_id"):
                    if identity_key not in metadata and identity_key in nested:
                        metadata[identity_key] = nested[identity_key]
        structure = _first_value(
            metadata,
            ("source_structure", "processed_metadata", "json_content", "structure", "layout", "ui_structure", "hierarchy", "elements"),
        )
        caption = _first_value(metadata, ("source_caption", "caption", "text", "prompt", "title"))
        return {
            "source_file": source_rel,
            "image_path": relative,
            "content_hash_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "image_width": width,
            "image_height": height,
            "file_size_bytes": destination.stat().st_size,
            "canonical_id": canonical_id,
            "source_record_id": source_record_id,
            "source_group_id": group_id,
            "group_id": group_id,
            "source_revision": revision,
            "provenance_url": getattr(self.spec, "provenance_url", None),
            "source_provenance": {
                "adapter": self.adapter_name,
                "dataset": self.key,
                "source_file": source_rel,
                "source_revision": revision,
                "provenance_url": getattr(self.spec, "provenance_url", None),
                "source_split": metadata.get("source_split"),
            },
            "source_metadata": metadata,
            "source_structure": structure,
            "source_caption": str(caption) if caption is not None else None,
        }


class Figma2CodeAdapter(_LocalArtifactAdapter):
    """Adapter for an explicitly selected local Figma2Code artifact tree."""

    adapter_name = "figma2code"

    def _fetch_parquet(self) -> list[dict[str, Any]]:
        records = super()._fetch_parquet()
        return records


class MobileViewsAdapter(_LocalArtifactAdapter):
    """Adapter for an explicitly selected local MobileViews artifact tree."""

    adapter_name = "mobileviews"

    def _fetch_parquet(self) -> list[dict[str, Any]]:
        # The core MobileViews files use image_content/json_content rather than
        # the generic image/screenshot names. The base implementation already
        # preserves both columns and decodes json_content; only the explicit
        # default is needed here.
        self.fetch_config.setdefault("image_column", "image_content")
        return super()._fetch_parquet()


def build_local_adapter(*, key: str, spec: Any, output_root: Path, data_root: Path,
                        source_root: Path | None = None) -> _LocalArtifactAdapter | None:
    """Build an adapter only for an explicit local adapter mode."""

    mode = str(spec.fetch_config.get("download_mode", "")).lower()
    if mode not in ADAPTER_MODES:
        return None
    if source_root is None:
        source_root = Path(spec.fetch_config.get("local_path", output_root))
        if not source_root.is_absolute():
            source_root = data_root / source_root
    adapter_type = MobileViewsAdapter if "mobileviews" in mode else Figma2CodeAdapter
    return adapter_type(
        key=key,
        spec=spec,
        source_root=source_root,
        output_root=output_root,
        data_root=data_root,
    )


__all__ = [
    "ADAPTER_MODES",
    "Figma2CodeAdapter",
    "MobileViewsAdapter",
    "build_local_adapter",
    "canonical_json",
    "iter_parquet_rows",
]
