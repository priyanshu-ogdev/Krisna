"""Auto-detection helper for scripts/training/train_polish_dpo.sh.

BUG FIX (docs/review/27_dpo_base_lora_wiring.md): both general-DPO configs
(polish_stage2_dpo_general.yaml, dpo_z_image_stage1_general.yaml) ship with
`lora-adapter-path: null`. Per train_dpo.py's own --lora-adapter-path help
text, an unset value means "DPO trains a fresh adapter on top of the base
pretrained weights" — so running the documented, canonical flow
(./train.sh --all, or train_polish_dpo.sh with its default config and no
manual edit) used to silently throw away the polish-default stage's LoRA
fine-tune entirely. The DPO stage would preference-align the vanilla
pretrained Z-Image-Turbo, not the UI-domain-adapted one train.sh --list's
own docstring describes ("...via diffusers' own dreambooth-LoRA script,
THEN refined further by polish-dpo"), and the PRD's §6 model-stack table
describes as ONE continuous "LoRA fine-tune + Diffusion-DPO" line, not two
independent adapters.

Extracted into its own importable module (rather than left inline in the
bash heredoc) specifically so this logic has a real, direct unit test
(tests/training/test_dpo_lora_autodetect.py) instead of only being
exercised by a full end-to-end training run.
"""

from __future__ import annotations

from pathlib import Path


def resolve_lora_adapter_path(cfg: dict, base_lora_output_dir: str | Path) -> tuple[dict, str]:
    """Given a parsed DPO training config dict, fill in `lora-adapter-path`
    when the config itself leaves it unset (None or missing), by pointing
    it at the base fine-tune's canonical output directory — but ONLY if a
    real checkpoint is actually present there.

    Never silently continues without an adapter when one was expected: if
    lora-adapter-path is unset AND no base checkpoint is found, the config
    is returned unmodified but the returned message is a loud warning, not
    a quiet confirmation, so the caller can print it regardless of outcome.

    An explicit value already in `cfg` (including an explicit domain-stage
    override, e.g. the Stage 2/domain DPO recipe) is never touched.

    Returns (possibly-updated cfg, human-readable message to print).
    """
    cfg = dict(cfg)  # don't mutate the caller's dict

    if cfg.get("lora-adapter-path") is not None:
        return cfg, (
            f"[polish-dpo] Using explicit lora-adapter-path from config: "
            f"{cfg['lora-adapter-path']}"
        )

    base_lora_dir = Path(base_lora_output_dir)
    # diffusers' dreambooth-LoRA scripts save directly into --output_dir
    # (e.g. pytorch_lora_weights.safetensors there), not a subdirectory —
    # matching what policy_pipe.load_lora_weights()/train_dpo.py's own
    # `--lora-adapter-path` expects (see train_dpo.py's
    # `policy_pipe.load_lora_weights(args.lora_adapter_path)` call).
    has_weights = base_lora_dir.exists() and any(
        p.suffix == ".safetensors" for p in base_lora_dir.glob("*.safetensors")
    )
    if has_weights:
        cfg["lora-adapter-path"] = str(base_lora_dir)
        return cfg, (
            f"[polish-dpo] lora-adapter-path was unset — auto-continuing "
            f"from the base fine-tune at {base_lora_dir}."
        )

    return cfg, (
        f"[polish-dpo] WARNING: lora-adapter-path is unset and no base LoRA "
        f"fine-tune was found at {base_lora_dir} (run "
        "./scripts/training/train_polish_default_lora.sh first). Proceeding "
        "WITHOUT it — DPO will train a fresh adapter on top of the base "
        "pretrained weights, NOT the UI-domain fine-tune. This is very "
        "likely not what you want for the default (general) DPO stage; "
        "set lora-adapter-path explicitly to silence this warning if it's "
        "intentional."
    )
