"""Active Polish training contract: Qwen-Image-2.1 image-conditioned LoRA.

Training delegates architecture-specific behavior to Diffusers' official
``train_dreambooth_lora_qwenimage21_img2img.py`` script. The repository owns
dataset pairing, configuration, and adapter provenance; it does not recreate
the Qwen noise schedule or transformer loss locally.

The former Z-Image, Qwen-Image-Edit, and custom flow-matching DPO paths were
removed because they represented incompatible pipeline and training contracts.
See ``docs/architecture/LEGACY_REJECTIONS.md`` and
``docs/research/phase5_agentic_workflow_design.md`` for the research rationale.
"""
