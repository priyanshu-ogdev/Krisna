"""Config consistency tests for the Polish-Default LoRA rank/alpha
setting across the base fine-tune and DPO continuation configs.

Regression test for a bug found in
docs/review/34_model_review_3_polish_default.md: both
polish_stage1_default_lora.yaml and polish_default_lora_z_image.yaml set
`rank: 32` but never set `lora_alpha` at all, silently deferring to the
official train_dreambooth_lora_z_image.py script's own default of 4 —
mismatched against rank=32 (verified directly against the real script's
argparse: --rank and --lora_alpha both default to 4, the standard
alpha==rank convention, which these configs broke the moment they
overrode rank alone). LoRA's effective update magnitude scales by
alpha/rank, so training would have silently produced a much weaker
adaptation than the rank actually paid for — no error, just a quiet
quality regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "training" / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / name).read_text())


@pytest.mark.parametrize(
    "config_name", ["polish_stage1_default_lora.yaml", "polish_default_lora_z_image.yaml"]
)
def test_base_lora_config_sets_alpha_equal_to_rank(config_name):
    cfg = _load(config_name)
    assert "rank" in cfg, f"{config_name} must set rank explicitly"
    assert "lora_alpha" in cfg, (
        f"{config_name} must set lora_alpha explicitly — leaving it unset silently "
        "defers to train_dreambooth_lora_z_image.py's own default (4), which will "
        "disagree with an overridden rank unless alpha is pinned to match."
    )
    assert cfg["lora_alpha"] == cfg["rank"], (
        f"{config_name}: lora_alpha ({cfg['lora_alpha']}) must equal rank "
        f"({cfg['rank']}) — the alpha==rank convention this project follows "
        "everywhere else (see train_dpo.py's fresh-adapter LoraConfig)."
    )


def test_base_and_dpo_stage_agree_on_rank():
    """The DPO stage (docs/review/27+ auto-detection) continues training
    the SAME adapter the base stage produced — a rank mismatch between
    the two would be a real capacity inconsistency, not just a style
    nit (see docs/review/02_polish_tier.md for why this was pinned)."""
    base_cfg = _load("polish_stage1_default_lora.yaml")
    dpo_cfg = _load("polish_stage2_dpo_general.yaml")
    assert base_cfg["rank"] == dpo_cfg["lora-rank"]


def test_two_base_config_variants_agree_with_each_other():
    """polish_stage1_default_lora.yaml (canonical) and
    polish_default_lora_z_image.yaml (verbose twin) must never silently
    drift on the values that actually affect training — same pattern as
    the sketch-tier config pairs checked elsewhere in this project."""
    canonical = _load("polish_stage1_default_lora.yaml")
    verbose = _load("polish_default_lora_z_image.yaml")
    for key in ("rank", "lora_alpha", "resolution", "learning_rate", "max_train_steps"):
        assert canonical[key] == verbose[key], f"{key} differs between the two config variants"
