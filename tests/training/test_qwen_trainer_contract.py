from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from krivex_training.polish.qwen_trainer_contract import (
    CONDITIONING_CONTRACT,
    MODEL_ID,
    MODEL_REVISION,
    PIPELINE_CLASS,
    build_accelerate_command,
    format_command,
    prepare_trainer_config,
)


def _config() -> dict:
    return {
        "diffusers_dir": "third_party/diffusers",
        "pretrained_model_name_or_path": MODEL_ID,
        "pretrained_model_revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
        "conditioning_contract": CONDITIONING_CONTRACT,
        "base_weights_frozen": True,
        "dataset_name": "data/pairs",
        "instance_prompt": "a polished production UI screen",
        "caption_column": "caption",
        "validation_prompt": None,
        "validation_image": None,
        "lr_scheduler": "cosine",
        "val_fraction": None,
        "val_every": None,
        "cond_image_column": "condition_image",
        "image_column": "image",
        "random_flip": False,
        "mixed_precision": "bf16",
        "use_aspect_ratio_buckets": True,
        "aspect_ratio_buckets": "1024,1024;832,1248;1248,832",
        "resolution": 1024,
        "rank": 64,
        "lora_alpha": 32,
        "max_train_steps": 5000,
    }


def test_config_maps_pinned_revision_and_strips_repository_metadata() -> None:
    prepared = prepare_trainer_config(_config())
    assert prepared["revision"] == MODEL_REVISION
    assert "pretrained_model_revision" not in prepared
    assert "pipeline_class" not in prepared
    assert "val_fraction" not in prepared


def test_config_rejects_native_unsupported_validation_controls() -> None:
    cfg = _config()
    cfg["val_every"] = 200
    with pytest.raises(ValueError, match="val_every.*unsupported"):
        prepare_trainer_config(cfg)


def test_config_rejects_unsupported_scheduler() -> None:
    cfg = _config()
    cfg["lr_scheduler"] = "cosine_with_warmup"
    with pytest.raises(ValueError, match="unsupported lr_scheduler"):
        prepare_trainer_config(cfg)


def test_config_rejects_non_native_bucket_syntax() -> None:
    cfg = _config()
    cfg["aspect_ratio_buckets"] = "1.0,0.75,1.333333"
    with pytest.raises(ValueError, match="height,width"):
        prepare_trainer_config(cfg)


def test_config_requires_validation_image_for_validation_prompt() -> None:
    cfg = _config()
    cfg["validation_prompt"] = "polish this UI"
    with pytest.raises(ValueError, match="validation_image"):
        prepare_trainer_config(cfg)


def test_command_builder_redacts_hub_token() -> None:
    cfg = _config()
    cfg["hub_token"] = "super-secret"
    rendered = format_command(build_accelerate_command(cfg, "third_party/diffusers"))
    assert "super-secret" not in rendered
    assert "<redacted>" in rendered


def test_checked_in_config_is_accepted_by_trainer_boundary() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "training/configs/polish_qwen_image21_lora.yaml").read_text())
    prepared = prepare_trainer_config(cfg)
    assert prepared["revision"] == MODEL_REVISION
    assert prepared["cond_image_column"] == "condition_image"


def test_qlora_consumer_config_accepted_and_validated() -> None:
    cfg = _config()
    cfg["load_in_4bit"] = True
    cfg["bnb_4bit_quant_type"] = "nf4"
    cfg["bnb_4bit_compute_dtype"] = "bfloat16"
    cfg["bnb_4bit_use_double_quant"] = True
    cfg["rank"] = 32

    prepared = prepare_trainer_config(cfg)
    assert prepared["revision"] == MODEL_REVISION
    assert "load_in_4bit" not in prepared
    assert "bnb_4bit_quant_type" not in prepared


def test_qlora_rejects_invalid_quant_type() -> None:
    cfg = _config()
    cfg["load_in_4bit"] = True
    cfg["bnb_4bit_quant_type"] = "invalid_quant"
    with pytest.raises(ValueError, match="unsupported bnb_4bit_quant_type"):
        prepare_trainer_config(cfg)


def test_command_builder_emits_use_dora_flag_when_true() -> None:
    cfg = _config()
    cfg["use_dora"] = True
    cmd = build_accelerate_command(cfg, "third_party/diffusers")
    assert "--use_dora" in cmd


def test_command_builder_omits_use_dora_flag_when_false() -> None:
    cfg = _config()
    cfg["use_dora"] = False
    cmd = build_accelerate_command(cfg, "third_party/diffusers")
    assert "--use_dora" not in cmd
    assert not any("--use_dora" in arg for arg in cmd)


def test_checked_in_config_emits_use_dora_in_accelerate_command() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "training/configs/polish_qwen_image21_lora.yaml").read_text())
    cmd = build_accelerate_command(cfg, "third_party/diffusers")
    assert "--use_dora" in cmd


def test_verify_diffusers_checkout_rejects_missing_dora_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from krivex_training.polish.qwen_trainer_contract import DIFFUSERS_REVISION, verify_diffusers_checkout

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    trainer_dir = tmp_path / "examples" / "dreambooth"
    trainer_dir.mkdir(parents=True)
    trainer_script = trainer_dir / "train_dreambooth_lora_qwenimage21_img2img.py"

    # Script without --use_dora or use_dora
    trainer_script.write_text("class QwenImage21Pipeline:\n    cond_image_column = 'cond'\n    image_column = 'img'\n")

    monkeypatch.setattr(
        "subprocess.check_output",
        lambda cmd, **kwargs: DIFFUSERS_REVISION if "rev-parse" in cmd else "",
    )

    with pytest.raises(ValueError, match="missing.*use_dora"):
        verify_diffusers_checkout(tmp_path)


def test_verify_diffusers_checkout_accepts_patched_dora_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from krivex_training.polish.qwen_trainer_contract import DIFFUSERS_REVISION, verify_diffusers_checkout

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    trainer_dir = tmp_path / "examples" / "dreambooth"
    trainer_dir.mkdir(parents=True)
    trainer_script = trainer_dir / "train_dreambooth_lora_qwenimage21_img2img.py"

    # Fully patched script
    trainer_script.write_text(
        "class QwenImage21Pipeline:\n"
        "    cond_image_column = 'cond'\n"
        "    image_column = 'img'\n"
        "    # --use_dora flag\n"
        "    use_dora = True\n"
    )

    monkeypatch.setattr(
        "subprocess.check_output",
        lambda cmd, **kwargs: DIFFUSERS_REVISION if "rev-parse" in cmd else "",
    )

    script, dirty = verify_diffusers_checkout(tmp_path)
    assert script == trainer_script
    assert dirty is False

