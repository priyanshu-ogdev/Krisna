"""Tests for fetcher.py's _fetch_gamelabel_csv and _decode_gamelabel_image.

GameLabel-10K (Jonathan-Zhou/GameLabel-10k, arXiv:2409.19830) ships a
schema the generic `preference_pair` mode can't handle: `img0_votes`/
`img1_votes` are vote COUNTS (0-5), not a fixed 0/1 label column, and
`img0_encoding`/`img1_encoding` are base64-encoded JPEG bytes wrapped in
a stray Python `b'...'` bytes-repr string — confirmed against the live
dataset's own HF dataset-viewer output, not guessed — rather than the
standard HF `{"bytes": ...}` image-feature dict
`_fetch_huggingface_preference_pairs` detects.

These tests build a real local parquet file (mirroring the auto-converted
`refs/convert/parquet` revision this adapter reads from) with images
encoded exactly the way the live dataset does, and monkeypatch
huggingface_hub.snapshot_download to point at it — no network — so the
actual decode/tie-dropping/labeling path is exercised end-to-end against
the real method, not a reimplementation of its logic.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from PIL import Image

from data_forge.config import DatasetSpec, PipelineConfig
from data_forge.data.fetcher import DatasetFetcher, _decode_gamelabel_image


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _wrap(raw: bytes) -> str:
    """Reproduce the live dataset's exact encoding: base64, then wrapped
    in Python's b'...' bytes-repr as a string."""
    return "b'" + base64.b64encode(raw).decode("ascii") + "'"


class TestDecodeGamelabelImage:
    def test_decodes_the_real_wrapped_format(self):
        raw = _jpeg_bytes((255, 0, 0))
        assert _decode_gamelabel_image(_wrap(raw)) == raw

    def test_handles_double_quote_variant(self):
        raw = _jpeg_bytes((0, 255, 0))
        wrapped_dq = 'b"' + base64.b64encode(raw).decode("ascii") + '"'
        assert _decode_gamelabel_image(wrapped_dq) == raw

    def test_handles_unwrapped_raw_base64_defensively(self):
        raw = _jpeg_bytes((0, 0, 255))
        assert _decode_gamelabel_image(base64.b64encode(raw).decode("ascii")) == raw

    @pytest.mark.parametrize("bad", [None, 12345, "", "b''", "not valid base64 at all!!!"])
    def test_returns_none_rather_than_raising_on_bad_input(self, bad):
        assert _decode_gamelabel_image(bad) is None


@pytest.fixture
def fetcher_and_paths(tmp_path: Path):
    cfg = PipelineConfig()
    cfg.resolved_paths["preference_pairs"] = tmp_path / "preference_pairs"
    cfg.resolved_paths["raw"] = tmp_path / "raw"
    cfg.resolved_paths["preference_pairs"].mkdir()
    return DatasetFetcher(cfg), tmp_path


def _make_parquet_snapshot(tmp_path: Path, df: pd.DataFrame) -> Path:
    snap_dir = tmp_path / "snapshot"
    snap_dir.mkdir()
    df.to_parquet(snap_dir / "train.parquet")
    return snap_dir


class TestFetchGamelabelCsv:
    @pytest.mark.asyncio
    async def test_ties_dropped_malformed_rows_skipped_labels_correct(self, fetcher_and_paths):
        """The actual behavior that matters: a clear vote-count win on
        either side produces a correctly-labeled pair; an equal vote
        split (including the row that also happens to have unrelated
        encoding text) is dropped as a tie; a row with genuinely
        undecodable image data is skipped without crashing the whole
        fetch over one bad row.
        """
        fetcher, tmp_path = fetcher_and_paths

        row0_a, row0_b = _jpeg_bytes((255, 0, 0)), _jpeg_bytes((0, 255, 0))
        row1_a, row1_b = _jpeg_bytes((0, 0, 255)), _jpeg_bytes((255, 255, 0))
        row2_a, row2_b = _jpeg_bytes((255, 0, 255)), _jpeg_bytes((0, 255, 255))
        row3_b = _jpeg_bytes((40, 50, 60))

        df = pd.DataFrame([
            {"prompt": "a red square", "img0_votes": 3, "img1_votes": 1,
             "img0_encoding": _wrap(row0_a), "img1_encoding": _wrap(row0_b)},
            {"prompt": "a tie case", "img0_votes": 2, "img1_votes": 2,
             "img0_encoding": _wrap(row1_a), "img1_encoding": _wrap(row1_b)},
            {"prompt": "b wins here", "img0_votes": 0, "img1_votes": 4,
             "img0_encoding": _wrap(row2_a), "img1_encoding": _wrap(row2_b)},
            {"prompt": "malformed row", "img0_votes": 5, "img1_votes": 0,
             "img0_encoding": "not base64 at all!!", "img1_encoding": _wrap(row3_b)},
        ])
        snap_dir = _make_parquet_snapshot(tmp_path, df)

        spec = DatasetSpec(
            display_name="GameLabel-10K", source_type="huggingface", category="dpo_preference_general",
            repo_id="Jonathan-Zhou/GameLabel-10k", fetch_config={"download_mode": "gamelabel_csv"},
        )

        with patch("huggingface_hub.snapshot_download", return_value=str(snap_dir)):
            await fetcher._fetch_gamelabel_csv("gamelabel_10k", spec, tmp_path / "dest")

        out_dir = fetcher._config.resolved_paths["preference_pairs"] / "gamelabel_10k"
        written = sorted(out_dir.glob("*.json"))
        assert len(written) == 2, f"expected 2 pairs (tie + malformed row dropped), got {len(written)}"

        records = [json.loads(p.read_text()) for p in written]
        by_prompt = {r["prompt"]: r for r in records}

        assert by_prompt["a red square"]["preferred"] == "a"
        assert by_prompt["a red square"]["vote_margin"] == 2
        assert by_prompt["b wins here"]["preferred"] == "b"
        assert by_prompt["b wins here"]["vote_margin"] == 4
        assert "a tie case" not in by_prompt
        assert "malformed row" not in by_prompt

        for r in records:
            assert r["origin"] == "gamelabel_10k"
            assert r["label_source"] == "human"
            a_path, b_path = out_dir / r["image_a"], out_dir / r["image_b"]
            assert a_path.exists() and b_path.exists()
            Image.open(a_path).load()
            Image.open(b_path).load()

    @pytest.mark.asyncio
    async def test_missing_repo_id_returns_empty_not_crash(self, fetcher_and_paths):
        fetcher, tmp_path = fetcher_and_paths
        spec = DatasetSpec(
            display_name="x", source_type="huggingface", category="dpo_preference_general",
            repo_id=None, fetch_config={"download_mode": "gamelabel_csv"},
        )
        result = await fetcher._fetch_gamelabel_csv("gamelabel_10k", spec, tmp_path / "dest")
        assert result == []

    @pytest.mark.asyncio
    async def test_schema_mismatch_logs_and_returns_empty_not_guesses(self, fetcher_and_paths):
        """If the live schema ever drifts from what was confirmed, this
        must fail loudly (empty result, logged columns) rather than
        silently misparsing whatever columns happen to exist."""
        fetcher, tmp_path = fetcher_and_paths
        df = pd.DataFrame([{"totally": "different", "columns": "here"}])
        snap_dir = _make_parquet_snapshot(tmp_path, df)
        spec = DatasetSpec(
            display_name="GameLabel-10K", source_type="huggingface", category="dpo_preference_general",
            repo_id="Jonathan-Zhou/GameLabel-10k", fetch_config={"download_mode": "gamelabel_csv"},
        )
        with patch("huggingface_hub.snapshot_download", return_value=str(snap_dir)):
            result = await fetcher._fetch_gamelabel_csv("gamelabel_10k", spec, tmp_path / "dest")
        assert result == []
