"""Dataset download manager — HuggingFace Hub, GitHub, and direct URL.

Handles dataset fetching with resume support, checksum verification,
and progress tracking via manifest updates.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from data_forge.config import DatasetSpec, PipelineConfig
from data_forge.logging_setup import get_logger

log = get_logger("data.fetcher")


class DatasetFetcher:
    """Multi-source dataset fetcher."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._raw_dir = config.resolved_paths["raw"]
        self._hf_token = os.environ.get("HF_TOKEN")

    async def fetch_dataset(self, key: str, spec: DatasetSpec) -> list[dict[str, Any]]:
        """Fetch a dataset and return a list of record dicts for manifest insertion.

        Each dict contains: source_file, image_path, content_hash_sha256,
        image_width, image_height, file_size_bytes.
        """
        log.info("fetch_starting", dataset=key, source_type=spec.source_type)

        dataset_dir = self._raw_dir / key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if spec.source_type == "huggingface":
            return await self._fetch_huggingface(key, spec, dataset_dir)
        elif spec.source_type == "github":
            return await self._fetch_github(key, spec, dataset_dir)
        elif spec.source_type == "url":
            return await self._fetch_url(key, spec, dataset_dir)
        else:
            log.error("unknown_source_type", dataset=key, source_type=spec.source_type)
            return []

    async def _fetch_huggingface(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Download dataset from HuggingFace Hub."""
        from huggingface_hub import snapshot_download

        if not spec.repo_id:
            log.error("missing_repo_id", dataset=key)
            return []

        log.info("hf_downloading", repo=spec.repo_id, dest=str(dest))

        # Build allow_patterns from fetch_config
        allow_patterns = spec.fetch_config.get("file_patterns")

        snapshot_dir = snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=spec.revision or "main",
            local_dir=str(dest),
            allow_patterns=allow_patterns,
            token=self._hf_token,
        )

        return self._scan_downloaded_files(key, Path(snapshot_dir))

    async def _fetch_github(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Clone or download a GitHub repository."""
        import subprocess

        if not spec.repo_url:
            log.error("missing_repo_url", dataset=key)
            return []

        clone_dir = dest / "repo"
        if clone_dir.exists():
            log.info("github_repo_exists", dataset=key, path=str(clone_dir))
        else:
            log.info("github_cloning", repo=spec.repo_url)
            subprocess.run(
                [
                    "git", "clone",
                    "--depth", "1",
                    "--branch", spec.branch or "main",
                    spec.repo_url,
                    str(clone_dir),
                ],
                check=True,
                capture_output=True,
            )

        return self._scan_downloaded_files(key, clone_dir)

    async def _fetch_url(
        self, key: str, spec: DatasetSpec, dest: Path
    ) -> list[dict[str, Any]]:
        """Download from a direct URL."""
        import httpx

        url = spec.fetch_config.get("url")
        if not url:
            log.error("missing_url", dataset=key)
            return []

        filename = url.rsplit("/", 1)[-1]
        file_path = dest / filename

        if file_path.exists():
            log.info("url_file_exists", path=str(file_path))
        else:
            log.info("url_downloading", url=url)
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, timeout=3600)
                response.raise_for_status()
                file_path.write_bytes(response.content)

        return self._scan_downloaded_files(key, dest)

    def _scan_downloaded_files(
        self, dataset_key: str, directory: Path
    ) -> list[dict[str, Any]]:
        """Scan downloaded directory for image files and return record metadata."""
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        records: list[dict[str, Any]] = []

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in image_extensions:
                continue

            # Compute SHA-256
            sha256 = self._compute_sha256(file_path)

            # Get image dimensions
            width, height = self._get_image_dimensions(file_path)

            # Relative path from DATA_ROOT
            try:
                rel_path = str(file_path.relative_to(self._config.data_root))
            except ValueError:
                rel_path = str(file_path)

            records.append({
                "source_file": file_path.name,
                "image_path": rel_path,
                "content_hash_sha256": sha256,
                "image_width": width,
                "image_height": height,
                "file_size_bytes": file_path.stat().st_size,
            })

        log.info(
            "scan_completed",
            dataset=dataset_key,
            image_count=len(records),
            directory=str(directory),
        )
        return records

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _get_image_dimensions(file_path: Path) -> tuple[int | None, int | None]:
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                return img.size  # (width, height)
        except Exception:
            return None, None
