"""Polish tier training — Z-Image and Qwen-Image-Edit-2511 LoRA.

Unlike the Sketch and Planner training packages, this one does NOT
implement a from-scratch training loop. Diffusion transformer training has
enough architecture-specific detail (noise schedules, VAE latent scaling,
flow-matching loss formulation, exact conditioning mechanics) that getting
it right from a spec sheet — for models this recent, without a GPU here to
actually verify against — is a real risk of shipping a subtly wrong
training loop instead of a working one.

Diffusers ships and maintains OFFICIAL LoRA training scripts for both of
these model families. That's the right thing to build on:

  - Z-Image: `train_dreambooth_lora_z_image.py`
    (github.com/huggingface/diffusers/blob/main/examples/dreambooth/
    train_dreambooth_lora_z_image.py). Base model is
    `Tongyi-MAI/Z-Image` — the UNDISTILLED foundation model, not
    `Z-Image-Turbo`. This corrects an earlier draft of this project's PRD
    conversation, which referenced a "Z-Image-De-Turbo" checkpoint that
    could not be confirmed to exist under that name; `Tongyi-MAI/Z-Image`
    is the real, confirmed undistilled model that serves the same role
    (full gradient signal for LoRA/fine-tuning, vs. Turbo's distilled
    weights which are reportedly less reliable to train against).
    `inference/polish_default_backend.py`'s `ZImageTurboBackend` then
    loads that LoRA onto the Turbo checkpoint for fast serving — the
    adapter's module names/shapes match (same DiT architecture), but
    actual output quality from that undistilled-trained /
    Turbo-inference-time combination is UNVERIFIED, only confirmed to load
    without error.

  - Qwen-Image-Edit-2511: **no official diffusers training script exists**
    as of this build. Confirmed directly from a diffusers maintainer's own
    reply in github.com/huggingface/diffusers/discussions/12469: "we
    don't have a training script for Qwen-image-edit-2509 at the moment.
    The only option with diffusers you have right now, is to maybe adapt
    the kontext training script to qwen image edit." Two real options,
    neither fabricated here as a working script:
      1. DiffSynth-Studio (github.com/modelscope/DiffSynth-Studio) — a
         separate, actively maintained framework from ModelScope with
         documented LoRA/full training support for the Qwen-Image family,
         including edit variants.
      2. Adapt `train_dreambooth_lora_flux_kontext.py` (diffusers' FLUX
         Kontext edit-training script — the closest architectural analog:
         image + text -> edited image) to Qwen-Image-Edit-2511's model
         loading calls. This is real adaptation work this project has NOT
         done — attempting to fabricate a working version of it without
         GPU access to actually verify each step would be worse than
         leaving the gap documented.
    `dataset_prep.py` in this package is usable for either path — it
    builds the standard dreambooth-style instance-image directory
    diffusers' scripts (and DiffSynth-Studio) both expect, independent of
    which trainer actually consumes it.

Once you have a LoRA safetensors file from either path,
`inference/polish_quality_backend.py`'s `QwenImageEditBackend` already
knows how to load it (`KRISNA_POLISH_QUALITY_LORA_PATH`) — the gap is in
producing the adapter, not in serving one once you have it.
"""
