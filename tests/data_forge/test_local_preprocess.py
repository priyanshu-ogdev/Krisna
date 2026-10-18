from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from data_forge.cli import main
from data_forge.manifest import Manifest
from PIL import Image


def test_preprocess_is_local_and_stops_at_fetched(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    source = tmp_path / "figma_source"
    source.mkdir()
    Image.new("RGB", (10, 8), (20, 40, 60)).save(source / "screen.png")
    (source / "screen.json").write_text(
        json.dumps({"filekey": "file-1", "node_id": "node:1", "processed_metadata": {"elements": []}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    result = CliRunner().invoke(
        main,
        ["preprocess", "--dataset", "figma2code", "--source-path", str(source)],
    )

    assert result.exit_code == 0, result.output
    assert '"models_loaded": false' in result.output
    assert '"network_used": false' in result.output
    assert (data_root / "audit_reports" / "preprocess_figma2code.json").is_file()

    manifest = Manifest(data_root / "manifests" / "manifest.db")
    try:
        records = manifest.query_by_dataset("figma2code")
        assert len(records) == 1
        assert records[0].status == "fetched"
        assert records[0].source_record_id == "node:1"
        assert records[0].source_group_id == "file-1"
        assert records[0].license_verified is None
    finally:
        manifest.close()
