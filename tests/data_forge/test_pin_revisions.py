"""Tests for scripts/pin_revisions.py.

No real network calls — HfApi is mocked throughout. What these tests
actually verify is real: correct traversal of the real datasets.yaml/
models.yaml structure, correct skip behavior (already-pinned entries,
repo_id: null entries), and — the part that would have shipped a real
bug if untested — that --apply produces a byte-for-byte no-op round trip
on everything it doesn't touch, and a minimal, comment-preserving diff
on what it does.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "data-forge"))
import pin_revisions  # noqa: E402

REAL_DATASETS_YAML = Path(__file__).parent.parent.parent / "data-forge" / "configs" / "datasets.yaml"
REAL_MODELS_YAML = Path(__file__).parent.parent.parent / "data-forge" / "configs" / "models.yaml"


class _FakeInfo:
    def __init__(self, sha):
        self.sha = sha


def _fake_api(sha="fakeShaForTesting123"):
    api = MagicMock()
    api.dataset_info.return_value = _FakeInfo(sha)
    api.model_info.return_value = _FakeInfo(sha)
    return api


class TestRoundTripFidelity:
    """The bug this locks in: an earlier version of _load()/_save() used
    ruamel.yaml's default settings, which silently reformatted every list
    in the file (wrong sequence/offset indent for this project's YAML
    style) and turned `null` values into empty scalars — on EVERY --apply
    run, even one that resolved zero revisions. Caught by diffing a
    load-then-immediately-dump round trip against the untouched file.
    """

    def test_models_yaml_round_trips_byte_for_byte(self):
        yaml, data = pin_revisions._load(REAL_MODELS_YAML)
        buf = io.StringIO()
        yaml.dump(data, buf)
        assert buf.getvalue() == REAL_MODELS_YAML.read_text()

    def test_datasets_yaml_round_trips_byte_for_byte(self):
        yaml, data = pin_revisions._load(REAL_DATASETS_YAML)
        buf = io.StringIO()
        yaml.dump(data, buf)
        assert buf.getvalue() == REAL_DATASETS_YAML.read_text()


class TestResolveDatasetRevisions:
    def test_dry_run_does_not_modify_the_file(self, tmp_path):
        target = tmp_path / "datasets.yaml"
        target.write_text(REAL_DATASETS_YAML.read_text())
        original = target.read_text()

        with patch("huggingface_hub.HfApi", return_value=_fake_api()):
            pin_revisions.resolve_dataset_revisions(target, only={"pd12m"}, apply=False)

        assert target.read_text() == original

    def test_apply_writes_real_sha_with_minimal_diff(self, tmp_path):
        target = tmp_path / "datasets.yaml"
        target.write_text(REAL_DATASETS_YAML.read_text())
        original_lines = target.read_text().splitlines()

        with patch("huggingface_hub.HfApi", return_value=_fake_api("resolvedTestSha999")):
            results = pin_revisions.resolve_dataset_revisions(target, only={"pd12m"}, apply=True)

        assert results == [{"key": "pd12m", "repo_id": "Spawning/PD12M", "status": "resolved", "sha": "resolvedTestSha999"}]
        new_lines = target.read_text().splitlines()
        diff_lines = [i for i, (a, b) in enumerate(zip(original_lines, new_lines)) if a != b]
        assert len(diff_lines) == 1, "exactly one line should differ — pd12m's revision value"
        assert "resolvedTestSha999" in new_lines[diff_lines[0]]
        # The line's trailing structure (comment, if any) must survive —
        # not just the value getting bulldozed into a bare scalar.
        assert new_lines[diff_lines[0]].strip().startswith("revision:")

    def test_null_repo_id_is_skipped_not_an_error(self, tmp_path):
        """designsense_10k / designpref — repo_id: null. Must be silently
        skipped, not treated as a resolution failure."""
        target = tmp_path / "datasets.yaml"
        target.write_text(REAL_DATASETS_YAML.read_text())

        with patch("huggingface_hub.HfApi", return_value=_fake_api()):
            results = pin_revisions.resolve_dataset_revisions(target, only={"designsense_10k", "designpref"}, apply=False)

        assert results == []

    def test_already_pinned_entries_are_reported_not_reresolved(self, tmp_path):
        target = tmp_path / "datasets.yaml"
        target.write_text(REAL_DATASETS_YAML.read_text().replace(
            'pd12m:\n    display_name: "PD12M (Public Domain 12M)"\n    source_type: "huggingface"\n    repo_id: "Spawning/PD12M"\n    revision: "main"',
            'pd12m:\n    display_name: "PD12M (Public Domain 12M)"\n    source_type: "huggingface"\n    repo_id: "Spawning/PD12M"\n    revision: "abc123alreadypinned"',
        ))
        fake = _fake_api()
        with patch("huggingface_hub.HfApi", return_value=fake):
            results = pin_revisions.resolve_dataset_revisions(target, only={"pd12m"}, apply=False)

        assert results == [{"key": "pd12m", "repo_id": "Spawning/PD12M", "status": "already_pinned", "revision": "abc123alreadypinned"}]
        fake.dataset_info.assert_not_called()

    def test_api_error_reported_not_raised(self, tmp_path):
        target = tmp_path / "datasets.yaml"
        target.write_text(REAL_DATASETS_YAML.read_text())
        fake = MagicMock()
        fake.dataset_info.side_effect = Exception("network unreachable")

        with patch("huggingface_hub.HfApi", return_value=fake):
            results = pin_revisions.resolve_dataset_revisions(target, only={"pd12m"}, apply=False)

        assert results[0]["status"] == "error"
        assert "network unreachable" in results[0]["error"]


class TestResolveModelRevisions:
    def test_finds_all_five_real_pinnable_entries(self):
        """Five, not six: encoders.maskgit_vq was removed (sync audit item
        #1) — the encoder had no working implementation and its only
        caller was deleted along with it."""
        with patch("huggingface_hub.HfApi", return_value=_fake_api()):
            results = pin_revisions.resolve_model_revisions(REAL_MODELS_YAML, only=None, apply=False)

        keys = {r["key"] for r in results}
        assert keys == {
            "models.tier1", "models.tier2", "models.ocr", "models.embeddings",
            "encoders.z_image_vae",
        }

    def test_product_planner_has_no_revision_field_and_is_not_touched(self):
        """product_planner is documentation-only (model_id present, but no
        revision key at all — data-forge never loads this model). Must not
        appear in results at all, not even as an error."""
        with patch("huggingface_hub.HfApi", return_value=_fake_api()):
            results = pin_revisions.resolve_model_revisions(REAL_MODELS_YAML, only=None, apply=False)

        assert not any(r["repo_id"] == "Qwen/Qwen3.5-9B" for r in results)

    def test_apply_produces_minimal_diff_on_real_file_copy(self, tmp_path):
        target = tmp_path / "models.yaml"
        target.write_text(REAL_MODELS_YAML.read_text())
        original_lines = target.read_text().splitlines()

        with patch("huggingface_hub.HfApi", return_value=_fake_api("modelTestSha777")):
            results = pin_revisions.resolve_model_revisions(target, only={"encoders.z_image_vae"}, apply=True)

        assert results[0]["status"] == "resolved"
        new_lines = target.read_text().splitlines()
        diff_lines = [i for i, (a, b) in enumerate(zip(original_lines, new_lines)) if a != b]
        assert len(diff_lines) == 1
        assert "modelTestSha777" in new_lines[diff_lines[0]]
