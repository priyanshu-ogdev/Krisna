import pytest
import os
import torch
from unittest.mock import MagicMock
from PIL import Image

from krisna_inference.orchestrator.model_registry import ModelSpec, Tier
from krisna_inference.backends.polish_default_backend import (
    ZImageTurboBackend,
    _prompt_from_constraints,
)
from krisna_inference.backends.blob_store_singleton import get_blob_store


def test_bridge_strength_config(monkeypatch):
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="z_image_turbo",
        vram_gb=14.0,
        quantization="bf16",
    )
    # Default is 0.75
    backend = ZImageTurboBackend(spec)
    assert backend.bridge_strength == 0.75

    # Override via env var
    monkeypatch.setenv("KRISNA_POLISH_DEFAULT_BRIDGE_STRENGTH", "0.65")
    backend_env = ZImageTurboBackend(spec)
    assert backend_env.bridge_strength == 0.65


def test_prompt_from_constraints():
    constraints = {
        "style": "glassmorphism dashboard with dark background",
        "palette": ["#0F172A", "#38BDF8", "#F8FAFC"],
        "layout_hints": "sidebar on left, 3 metrics cards on top, chart in center",
    }
    prompt = _prompt_from_constraints(constraints)
    assert "glassmorphism dashboard" in prompt
    assert "#0F172A" in prompt
    assert "sidebar on left" in prompt


@pytest.mark.asyncio
async def test_bridge_conditioning_latent_injection(tmp_path):
    spec = ModelSpec(
        tier=Tier.POLISH_DEFAULT,
        name="z_image_turbo",
        vram_gb=14.0,
        quantization="bf16",
    )
    backend = ZImageTurboBackend(spec, bridge_strength=0.70)
    
    # Save a mock sketch image to BlobStore
    store = get_blob_store()
    sketch_img = Image.new("RGB", (256, 256), color=(50, 100, 150))
    sketch_ref = store.save_image(sketch_img, prefix="test_sketch")

    # Mock pipeline
    mock_pipe = MagicMock()
    mock_pipe.vae.parameters.return_value = [torch.zeros(1, device="cpu")]
    mock_pipe.vae.dtype = torch.float32
    mock_pipe.dtype = torch.float32
    mock_pipe.transformer.in_channels = 16
    mock_pipe.transformer.config.axes_lens = [256, 128, 128]
    mock_pipe.vae.config.scaling_factor = 0.3611
    mock_pipe.vae.config.shift_factor = 0.0

    # Mock VAE encode returning a mock latent distribution
    mock_latent_dist = MagicMock()
    mock_latent_dist.sample.return_value = torch.ones(1, 16, 128, 128)
    mock_pipe.vae.encode.return_value.latent_dist = mock_latent_dist

    # Mock prepare_latents returning expected shape
    mock_pipe.prepare_latents.return_value = torch.zeros(1, 16, 128, 128)

    # Mock pipe output
    out_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
    mock_pipe.return_value.images = [out_img]

    backend._pipe = mock_pipe
    backend._loaded = True

    # Run backend with handoff_image_ref and a locked region (canonical [x, y, w, h] navbar)
    res = await backend.run(
        handoff_image_ref=sketch_ref,
        constraints={
            "style": "clean modern SaaS",
            "locked_regions": [{"bbox": [0.0, 0.0, 1.0, 0.2], "reason": "navbar"}],
        },
        seed=123,
    )

    assert res["tier"] == "polish_default"
    assert res["image_ref"].startswith("blob://")
    
    # Verify mock_pipe was called with latents passed
    call_kwargs = mock_pipe.call_args.kwargs
    assert "latents" in call_kwargs
    passed_latents = call_kwargs["latents"]
    assert passed_latents is not None
    assert passed_latents.shape == (1, 16, 128, 128)
    
    # In the locked region (y from 0 to int(0.2*128)=25, x from 0 to 128), latents should equal z_sketch exactly
    # z_sketch = 1.0 * 0.3611 = 0.3611
    locked_slice = passed_latents[:, :, 0:25, :]
    assert torch.allclose(locked_slice, torch.tensor(0.3611), atol=1e-4)

    # Verify callback_on_step_end is wired and enforces locked regions on subsequent steps
    assert "callback_on_step_end" in call_kwargs
    cb = call_kwargs["callback_on_step_end"]
    # At final step (timestep=0.0), clamped slice must equal z_sketch exactly
    kw_test = {"latents": torch.randn(1, 16, 128, 128)}
    kw_out = cb(mock_pipe, 8, 0.0, kw_test)
    assert torch.allclose(kw_out["latents"][:, :, 0:25, :], torch.tensor(0.3611), atol=1e-4)
