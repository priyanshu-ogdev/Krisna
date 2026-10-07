"""Path confinement helpers for data read from the manifest."""

from __future__ import annotations

from pathlib import Path


def resolve_data_path(data_root: Path, value: str | Path) -> Path:
    """Resolve a manifest path and reject absolute, traversal, or symlink escapes."""
    root = Path(data_root).resolve()
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=True)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Manifest path escapes DATA_ROOT: {value}")
    return resolved
