"""Offline tests for fetch commit, archive, and provenance safeguards."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import ClassVar

import pytest
from data_forge.config import DatasetSpec, PipelineConfig
from data_forge.data.fetcher import DatasetFetcher, FetchSafetyError
from data_forge.stages.s00_manifest_planning import _production_preflight


class _StreamResponse:
    def __init__(self, body: bytes, *, status_code: int, headers: dict[str, str]):
        self.status_code = status_code
        self.headers = headers
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self):
        yield self._body


class _OfflineClient:
    response: _StreamResponse
    requests: ClassVar[list[dict[str, str]]] = []

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def stream(self, method: str, url: str, *, headers: dict[str, str], **_kwargs):
        self.requests.append({"method": method, "url": url, **headers})
        return self.response


@pytest.mark.asyncio
async def test_direct_url_resumes_partial_and_commits_checksum(config: PipelineConfig, monkeypatch, tmp_path: Path):
    fetcher = DatasetFetcher(config)
    destination = tmp_path / "raw"
    destination.mkdir()
    output = destination / "asset.bin"
    output.with_name(".asset.bin.part").write_bytes(b"abc")

    body = b"def"
    _OfflineClient.requests = []
    _OfflineClient.response = _StreamResponse(
        body,
        status_code=206,
        headers={"content-length": str(len(body)), "content-range": "bytes 3-5/6"},
    )
    monkeypatch.setattr("httpx.AsyncClient", _OfflineClient)

    await fetcher._download_url_resumable(
        "https://example.test/asset.bin",
        output,
        expected_sha256=hashlib.sha256(b"abcdef").hexdigest(),
        expected_size=6,
        max_download_bytes=6,
        resume=True,
        timeout_seconds=5,
    )

    assert output.read_bytes() == b"abcdef"
    assert not output.with_name(".asset.bin.part").exists()
    assert _OfflineClient.requests[0]["Range"] == "bytes=3-"


@pytest.mark.asyncio
async def test_direct_url_checksum_failure_removes_untrusted_partial(
    config: PipelineConfig, monkeypatch, tmp_path: Path
):
    fetcher = DatasetFetcher(config)
    output = tmp_path / "asset.bin"
    _OfflineClient.response = _StreamResponse(b"bad", status_code=200, headers={"content-length": "3"})
    monkeypatch.setattr("httpx.AsyncClient", _OfflineClient)

    with pytest.raises(FetchSafetyError, match="SHA-256 mismatch"):
        await fetcher._download_url_resumable(
            "https://example.test/asset.bin",
            output,
            expected_sha256="0" * 64,
            expected_size=3,
            max_download_bytes=3,
            resume=True,
            timeout_seconds=5,
        )

    assert not output.exists()
    assert not output.with_name(".asset.bin.part").exists()


@pytest.mark.asyncio
async def test_direct_url_enforces_announced_download_quota(
    config: PipelineConfig, monkeypatch, tmp_path: Path
):
    fetcher = DatasetFetcher(config)
    output = tmp_path / "asset.bin"
    _OfflineClient.response = _StreamResponse(b"1234", status_code=200, headers={"content-length": "4"})
    monkeypatch.setattr("httpx.AsyncClient", _OfflineClient)

    with pytest.raises(FetchSafetyError, match="max_download_bytes"):
        await fetcher._download_url_resumable(
            "https://example.test/asset.bin",
            output,
            expected_sha256=None,
            expected_size=None,
            max_download_bytes=3,
            resume=True,
            timeout_seconds=5,
        )

    assert not output.exists()


def test_archive_traversal_is_rejected_without_deleting_archive(config: PipelineConfig, tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("../outside.txt", "must not escape")

    DatasetFetcher(config)._unpack_archives(tmp_path)

    assert archive.exists()
    assert not (tmp_path.parent / "outside.txt").exists()


def test_archive_member_and_expansion_limits_are_enforced(config: PipelineConfig, tmp_path: Path):
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("one.txt", "one")
        writer.writestr("two.txt", "two")

    DatasetFetcher(config)._unpack_archives(tmp_path, max_members=1)

    assert archive.exists()
    assert not (tmp_path / "one.txt").exists()


def test_safe_archive_extracts_regular_members(config: PipelineConfig, tmp_path: Path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("nested/ok.txt", "safe")

    DatasetFetcher(config)._unpack_archives(tmp_path)

    assert (tmp_path / "nested" / "ok.txt").read_text(encoding="utf-8") == "safe"
    assert not archive.exists()


def test_production_preflight_requires_provenance_url():
    config = PipelineConfig(
        datasets={
            "candidate": DatasetSpec(
                display_name="Candidate",
                source_type="url",
                category="web_ui",
                training_eligible=True,
                license_status="verified",
                provenance_url=None,
            )
        }
    )

    blockers = _production_preflight(config)

    assert "dataset candidate: provenance_url is missing" in blockers
