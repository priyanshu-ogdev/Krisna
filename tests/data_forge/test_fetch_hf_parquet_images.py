"""Tests for fetcher.py's _fetch_hf_parquet_images.

Regression coverage for the bug found reviewing rico_core/rico_semantic:
both live HF repos (creative-graphic-design/Rico, Voxel51/rico) ship
images embedded as bytes inside parquet files, not as loose .jpg/.png
files — but their fetch_config used `file_patterns: ["*.jpg","*.png",
"*.json"]` against the plain `_fetch_huggingface` -> `_scan_downloaded_
files` path, which globs snapshot_download's *output directory* for
those extensions. Since neither repo's real file tree contains any file
matching those patterns, snapshot_download's `allow_patterns` would
silently download nothing, and the scan would find zero images — the
single most foundational dataset in the UI-domain corpus producing zero
training records with no error raised anywhere.

These tests build a real local parquet file with embedded image bytes
(no network) and monkeypatch huggingface_hub.snapshot_download to point
at it, so the actual decode path is exercised end-to-end.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from PIL import Image

from data_forge.config import DatasetSpec
from data_forge.data.fetcher import DatasetFetcher


def _make_embedded_image_parquet(path: Path, n: int, image_col: str = "screenshot", caption_col: str | None = None) -> None:
    rows = []
    for i in range(n):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), color=(i * 10 % 255, 20, 30)).save(buf, format="PNG")
        row = {image_col: {"bytes": buf.getvalue(), "path": None}, "ui_number": i}
        if caption_col:
            row[caption_col] = f"caption for screen {i}"
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path)


@pytest.fixture
def fetcher(config):
    return DatasetFetcher(config)


class TestFetchHfParquetImages:
    async def test_decodes_embedded_images_and_returns_records(self, fetcher, config, tmp_path):
        parquet_dir = tmp_path / "fake_snapshot"
        parquet_dir.mkdir()
        _make_embedded_image_parquet(parquet_dir / "train-00000.parquet", n=5, image_col="screenshot")

        spec = DatasetSpec(
            display_name="rico_core", source_type="huggingface", repo_id="creative-graphic-design/Rico",
            category="mobile_ui", expected_record_count=5,
            fetch_config={"download_mode": "hf_parquet_images", "image_column": "screenshot"},
        )
        dest = config.data_root / "raw" / "rico_core"

        with patch("huggingface_hub.snapshot_download", return_value=str(parquet_dir)):
            records = await fetcher._fetch_hf_parquet_images("rico_core", spec, dest)

        assert len(records) == 5
        for rec in records:
            assert (config.data_root / rec["image_path"]).exists()
            assert rec["image_width"] == 32
            assert rec["image_height"] == 32
            assert "content_hash_sha256" in rec

    async def test_auto_detects_image_column_when_not_specified(self, fetcher, config, tmp_path):
        """rico_semantic's real fix: image_column is deliberately left
        unset in datasets.yaml since Voxel51/rico's exact column name
        wasn't independently confirmed — auto-detection has to actually
        work, not just exist as an unused code path."""
        parquet_dir = tmp_path / "fake_snapshot"
        parquet_dir.mkdir()
        _make_embedded_image_parquet(parquet_dir / "train-00000.parquet", n=3, image_col="whatever_column_name")

        spec = DatasetSpec(
            display_name="rico_semantic", source_type="huggingface", repo_id="Voxel51/rico",
            category="mobile_ui", expected_record_count=3,
            fetch_config={"download_mode": "hf_parquet_images"},  # no image_column override
        )
        dest = config.data_root / "raw" / "rico_semantic"

        with patch("huggingface_hub.snapshot_download", return_value=str(parquet_dir)):
            records = await fetcher._fetch_hf_parquet_images("rico_semantic", spec, dest)

        assert len(records) == 3

    async def test_metadata_only_config_with_no_image_column_returns_empty_not_crash(self, fetcher, config, tmp_path):
        """Regression guard for RICO's "default" config specifically —
        pure metadata (ui_number, app_package_name, ...), no image column
        at all. Must fail loudly (empty list + error log) rather than
        crash or silently fabricate an image column."""
        parquet_dir = tmp_path / "fake_snapshot"
        parquet_dir.mkdir()
        pd.DataFrame([{"ui_number": i, "app_package_name": "com.example"} for i in range(3)]).to_parquet(
            parquet_dir / "metadata-00000.parquet"
        )

        spec = DatasetSpec(
            display_name="rico_default_config", source_type="huggingface", repo_id="creative-graphic-design/Rico",
            category="mobile_ui", expected_record_count=3,
            fetch_config={"download_mode": "hf_parquet_images"},
        )
        dest = config.data_root / "raw" / "rico_default_config"

        with patch("huggingface_hub.snapshot_download", return_value=str(parquet_dir)):
            records = await fetcher._fetch_hf_parquet_images("rico_default_config", spec, dest)

        assert records == []

    async def test_caption_column_captured_when_present(self, fetcher, config, tmp_path):
        parquet_dir = tmp_path / "fake_snapshot"
        parquet_dir.mkdir()
        _make_embedded_image_parquet(
            parquet_dir / "train-00000.parquet", n=2, image_col="screenshot", caption_col="alt_text",
        )

        spec = DatasetSpec(
            display_name="test", source_type="huggingface", repo_id="fake/repo",
            category="mobile_ui", expected_record_count=2,
            fetch_config={"download_mode": "hf_parquet_images", "image_column": "screenshot", "caption_column": "alt_text"},
        )
        dest = config.data_root / "raw" / "test"

        with patch("huggingface_hub.snapshot_download", return_value=str(parquet_dir)):
            records = await fetcher._fetch_hf_parquet_images("test", spec, dest)

        assert len(records) == 2
        assert records[0]["source_caption"] == "caption for screen 0"

    async def test_sample_size_caps_decoded_records(self, fetcher, config, tmp_path):
        parquet_dir = tmp_path / "fake_snapshot"
        parquet_dir.mkdir()
        _make_embedded_image_parquet(parquet_dir / "train-00000.parquet", n=20, image_col="screenshot")

        spec = DatasetSpec(
            display_name="test", source_type="huggingface", repo_id="fake/repo",
            category="mobile_ui", expected_record_count=20,
            fetch_config={"download_mode": "hf_parquet_images", "image_column": "screenshot", "sample_size": 5},
        )
        dest = config.data_root / "raw" / "test"

        with patch("huggingface_hub.snapshot_download", return_value=str(parquet_dir)):
            records = await fetcher._fetch_hf_parquet_images("test", spec, dest)

        assert len(records) == 5

    async def test_config_subfolder_restricts_allow_patterns(self, fetcher, config, tmp_path):
        """RICO's specific fix: config_subfolder must scope
        snapshot_download's allow_patterns so the "default" (metadata-
        only) config's parquet files are never even downloaded when
        targeting "ui-screenshots-and-view-hierarchies"."""
        spec = DatasetSpec(
            display_name="rico_core", source_type="huggingface", repo_id="creative-graphic-design/Rico",
            category="mobile_ui", expected_record_count=1,
            fetch_config={"download_mode": "hf_parquet_images", "config_subfolder": "ui-screenshots-and-view-hierarchies"},
        )
        dest = config.data_root / "raw" / "rico_core"

        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as mock_dl:
            await fetcher._fetch_hf_parquet_images("rico_core", spec, dest)

        mock_dl.assert_called_once()
        assert mock_dl.call_args.kwargs["allow_patterns"] == ["ui-screenshots-and-view-hierarchies/*.parquet"]

    async def test_missing_repo_id_returns_empty(self, fetcher, config, tmp_path):
        spec = DatasetSpec(
            display_name="test", source_type="huggingface", repo_id=None,
            category="mobile_ui", expected_record_count=1,
            fetch_config={"download_mode": "hf_parquet_images"},
        )
        records = await fetcher._fetch_hf_parquet_images("test", spec, config.data_root / "raw" / "test")
        assert records == []

    async def test_nested_caption_column_captured(self, fetcher, config, tmp_path):
        """Android Control and similar datasets store instructions inside nested json objects."""
        parquet_dir = tmp_path / "fake_snapshot"
        parquet_dir.mkdir()
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="PNG")
        rows = [
            {
                "png": {"bytes": buf.getvalue(), "path": None},
                "json": {
                    "messages": [
                        {"role": "instruction", "content": "Launch the music player and search for jazz"},
                        {"role": "assistant", "content": "Done"}
                    ]
                }
            }
        ]
        pd.DataFrame(rows).to_parquet(parquet_dir / "train-00000.parquet")

        spec = DatasetSpec(
            display_name="android_control", source_type="huggingface", repo_id="fake/android_control",
            category="mobile_ui", expected_record_count=1,
            fetch_config={
                "download_mode": "hf_parquet_images",
                "image_column": "png",
                "caption_column": "json.messages.0.content"
            },
        )
        dest = config.data_root / "raw" / "android_control"
        with patch("huggingface_hub.snapshot_download", return_value=str(parquet_dir)):
            records = await fetcher._fetch_hf_parquet_images("android_control", spec, dest)

        assert len(records) == 1
        assert records[0]["source_caption"] == "Launch the music player and search for jazz"

    async def test_viewport_crop_on_extreme_aspect_ratio(self, fetcher, config, tmp_path):
        """Multimodal Mind2Web and full-page web captures have extreme heights (>4:1) that
        would otherwise trigger s08_encoding exclusions. Viewport cropping slices the top fold."""
        parquet_dir = tmp_path / "fake_snapshot"
        parquet_dir.mkdir()
        buf = io.BytesIO()
        # 100 wide by 500 tall (5:1 aspect ratio)
        Image.new("RGB", (100, 500), color=(50, 60, 70)).save(buf, format="PNG")
        rows = [{"screenshot": {"bytes": buf.getvalue(), "path": None}, "task": "Book a flight"}]
        pd.DataFrame(rows).to_parquet(parquet_dir / "train-00000.parquet")

        spec = DatasetSpec(
            display_name="mind2web", source_type="huggingface", repo_id="fake/mind2web",
            category="web_ui", expected_record_count=1,
            fetch_config={
                "download_mode": "hf_parquet_images",
                "image_column": "screenshot",
                "caption_column": "task",
                "viewport_crop": True,
                "max_aspect_ratio": 2.0
            },
        )
        dest = config.data_root / "raw" / "mind2web"
        with patch("huggingface_hub.snapshot_download", return_value=str(parquet_dir)):
            records = await fetcher._fetch_hf_parquet_images("mind2web", spec, dest)

        assert len(records) == 1
        assert records[0]["image_width"] == 100
        assert records[0]["image_height"] == 200  # 100 * 2.0 = 200
        saved_img = Image.open(config.data_root / records[0]["image_path"])
        assert saved_img.size == (100, 200)

    async def test_alpha_composite_onto_white(self, fetcher, config, tmp_path):
        """Transparent UI icons/elements should be composited onto clean white rather than black."""
        parquet_dir = tmp_path / "fake_snapshot"
        parquet_dir.mkdir()
        buf = io.BytesIO()
        # Completely transparent RGBA image
        Image.new("RGBA", (32, 32), color=(0, 0, 0, 0)).save(buf, format="PNG")
        rows = [{"image": {"bytes": buf.getvalue(), "path": None}}]
        pd.DataFrame(rows).to_parquet(parquet_dir / "train-00000.parquet")

        spec = DatasetSpec(
            display_name="screenspot", source_type="huggingface", repo_id="fake/screenspot",
            category="cross_platform_ui", expected_record_count=1,
            fetch_config={"download_mode": "hf_parquet_images", "image_column": "image"},
        )
        dest = config.data_root / "raw" / "screenspot"
        with patch("huggingface_hub.snapshot_download", return_value=str(parquet_dir)):
            records = await fetcher._fetch_hf_parquet_images("screenspot", spec, dest)

        assert len(records) == 1
        saved_img = Image.open(config.data_root / records[0]["image_path"])
        assert saved_img.mode == "RGB"
        # Transparent pixel should be white (255, 255, 255)
        assert saved_img.getpixel((0, 0)) == (255, 255, 255)

    async def test_streaming_multi_file_parquet_stops_at_sample_size(self, fetcher, config, tmp_path):
        """Multi-parquet datasets (e.g. SeeClick 255 files, AndroidControl 85 files) must be
        streamed file-by-file and stop immediately once sample_size is satisfied."""
        parquet_dir = tmp_path / "fake_snapshot"
        parquet_dir.mkdir()
        for f_idx in range(3):
            _make_embedded_image_parquet(parquet_dir / f"train-{f_idx:05d}.parquet", n=5, image_col="image")

        spec = DatasetSpec(
            display_name="seeclick", source_type="huggingface", repo_id="fake/seeclick",
            category="web_ui", expected_record_count=15,
            fetch_config={"download_mode": "hf_parquet_images", "image_column": "image", "sample_size": 7},
        )
        dest = config.data_root / "raw" / "seeclick"
        with patch("huggingface_hub.snapshot_download", return_value=str(parquet_dir)):
            records = await fetcher._fetch_hf_parquet_images("seeclick", spec, dest)

        assert len(records) == 7


class TestRicoSemanticImageColumnConfirmed:
    """Regression guard for the specific open item closed this revision:
    rico_semantic's image_column used to be left unset for auto-detection
    (Voxel51/rico's exact column name wasn't independently confirmed at
    the time). Confirmed directly against the live dataset card's own
    Data Studio preview table header ("| image  image |") — now
    hardcoded in datasets.yaml rather than relying on auto-detection at
    fetch time. This test locks that in against the real config file, not
    just a synthetic DatasetSpec, so a future edit that silently removes
    the override would fail here.
    """

    def test_datasets_yaml_hardcodes_confirmed_image_column(self):
        from pathlib import Path

        from data_forge.config import load_config

        # BUG FIX (found relocating this repo into the krisna/ monorepo):
        # bare relative "configs/..." strings assumed pytest ran with cwd
        # inside data-forge/ itself. Tests now run from the monorepo root
        # (krisna/) — resolve explicitly against this file's real location
        # instead of relying on cwd. See docs/architecture/DIRECTORY_LAYOUT.md.
        configs_dir = Path(__file__).parent.parent.parent / "data-forge" / "configs"
        cfg = load_config(
            str(configs_dir / "pipeline.yaml"),
            str(configs_dir / "models.yaml"),
            str(configs_dir / "datasets.yaml"),
        )
        spec = cfg.datasets["rico_semantic"]
        assert spec.fetch_config.get("image_column") == "image"
        assert spec.fetch_config.get("download_mode") == "hf_parquet_images"

    def test_rico_core_and_rico_semantic_use_different_confirmed_column_names(self):
        """The two RICO repos are independently exported and do NOT
        share a column name — rico_core (creative-graphic-design/Rico)
        uses "screenshot", rico_semantic (Voxel51/rico) uses "image".
        Both confirmed directly against their live dataset cards, not
        assumed to match just because they're "the same underlying
        screens." """
        from pathlib import Path

        from data_forge.config import load_config

        configs_dir = Path(__file__).parent.parent.parent / "data-forge" / "configs"
        cfg = load_config(
            str(configs_dir / "pipeline.yaml"),
            str(configs_dir / "models.yaml"),
            str(configs_dir / "datasets.yaml"),
        )
        assert cfg.datasets["rico_core"].fetch_config.get("image_column") == "screenshot"
        assert cfg.datasets["rico_semantic"].fetch_config.get("image_column") == "image"
