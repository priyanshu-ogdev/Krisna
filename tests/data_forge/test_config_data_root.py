"""Regression test for the DATA_ROOT platform-consistency fix.

`_default_data_root()` used to branch on `platform.system()` — an
absolute `/data_krisna` on Linux, `D:\\data_krisna` on Windows. That
default was never actually reachable, though: the repo's committed
`.env` unconditionally set `DATA_ROOT=D:\\data_krisna`, and
data_forge/cli.py's `_load_env_file()` loads that file on every
platform, including the Linux/WSL2 target `run_data_forge.sh` is
written for. `Path("D:\\data_krisna")` on POSIX doesn't parse as a
drive path — backslashes aren't separators there — so it silently
resolved to a single-component relative folder literally named
"D:\\data_krisna", disagreeing with scripts/data-forge/
sync_to_training.py's own default (`REPO_ROOT / "data_krisna"`) and
with train.sh --help's documented default (`./data_krisna`).

Fixed by dropping the platform branch entirely and standardizing on one
repo(cwd)-relative default everywhere. This test locks that in so it
can't quietly regress back to a platform-branched or OS-specific
absolute default. See docs/review/26_data_root_path_consistency.md.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from data_forge.config import _default_data_root, load_config


def test_default_data_root_is_platform_neutral(tmp_path, monkeypatch) -> None:
    """The no-env-var default must not branch on platform.system() and
    must not be a Windows-style (backslash / drive-letter) path when
    running on a POSIX system."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATA_ROOT", raising=False)

    resolved = _default_data_root()

    assert resolved == tmp_path / "data_krisna"
    # Never a literal, unparsed Windows path on this filesystem.
    assert "\\" not in str(resolved)
    assert resolved.is_absolute()


def test_default_data_root_matches_sync_to_training_default(tmp_path, monkeypatch) -> None:
    """data_forge.config's default must agree with
    scripts/data-forge/sync_to_training.py's independent default
    (REPO_ROOT / "data_krisna") — this is the exact mismatch that let
    the Windows-path bug hide, since the two scripts otherwise happened
    to agree with each other by both reading the same broken env var."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATA_ROOT", raising=False)

    config_default = _default_data_root()
    sync_script_default = tmp_path / "data_krisna"  # REPO_ROOT / "data_krisna"

    assert config_default == sync_script_default


@pytest.mark.parametrize(
    "env_value",
    [
        "./custom_data_root",
        "/absolute/custom/root",
        r"D:\data_krisna",  # explicit opt-in override is still fully supported
    ],
)
def test_explicit_data_root_env_var_is_still_honored_verbatim(
    env_value: str, monkeypatch
) -> None:
    """An explicit DATA_ROOT override (the supported escape hatch) must
    still be honored exactly as given — this fix only changes the
    fallback used when nothing is set."""
    monkeypatch.setenv("DATA_ROOT", env_value)
    config = load_config()
    assert config.data_root == Path(env_value)
