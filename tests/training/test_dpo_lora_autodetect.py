"""Regression test for docs/review/27_dpo_base_lora_wiring.md's fix:
scripts/training/train_polish_dpo.sh must never silently run the DPO
stage disconnected from the base LoRA fine-tune when a real checkpoint
for it exists on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "training"))

from dpo_lora_autodetect import resolve_lora_adapter_path  # noqa: E402


def _make_fake_lora_checkpoint(tmp_path: Path) -> Path:
    lora_dir = tmp_path / "checkpoints" / "polish_default_lora"
    lora_dir.mkdir(parents=True)
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"fake")
    return lora_dir


def test_auto_detects_base_lora_when_checkpoint_exists(tmp_path) -> None:
    lora_dir = _make_fake_lora_checkpoint(tmp_path)
    cfg = {"lora-adapter-path": None, "output-dir": "x"}

    new_cfg, message = resolve_lora_adapter_path(cfg, lora_dir)

    assert new_cfg["lora-adapter-path"] == str(lora_dir)
    assert "auto-continuing" in message
    assert "WARNING" not in message


def test_warns_and_leaves_unset_when_no_checkpoint_exists(tmp_path) -> None:
    missing_dir = tmp_path / "checkpoints" / "polish_default_lora"
    cfg = {"lora-adapter-path": None, "output-dir": "x"}

    new_cfg, message = resolve_lora_adapter_path(cfg, missing_dir)

    assert new_cfg["lora-adapter-path"] is None
    assert "WARNING" in message
    assert "fresh adapter" in message


def test_explicit_config_value_is_never_overridden(tmp_path) -> None:
    lora_dir = _make_fake_lora_checkpoint(tmp_path)
    explicit_path = "models/dpo_checkpoints/stage1_general"
    cfg = {"lora-adapter-path": explicit_path, "output-dir": "x"}

    new_cfg, message = resolve_lora_adapter_path(cfg, lora_dir)

    assert new_cfg["lora-adapter-path"] == explicit_path
    assert explicit_path in message


def test_missing_key_entirely_is_treated_same_as_none(tmp_path) -> None:
    """A config dict that never had the key at all (e.g. a hand-edited
    YAML that dropped the line) must be handled the same way as an
    explicit `null` — .get() with no default returns None either way,
    but this locks that equivalence in."""
    lora_dir = _make_fake_lora_checkpoint(tmp_path)
    cfg = {"output-dir": "x"}  # no "lora-adapter-path" key at all

    new_cfg, message = resolve_lora_adapter_path(cfg, lora_dir)

    assert new_cfg["lora-adapter-path"] == str(lora_dir)


def test_does_not_mutate_caller_dict(tmp_path) -> None:
    lora_dir = _make_fake_lora_checkpoint(tmp_path)
    cfg = {"lora-adapter-path": None, "output-dir": "x"}

    resolve_lora_adapter_path(cfg, lora_dir)

    assert cfg["lora-adapter-path"] is None  # original untouched
