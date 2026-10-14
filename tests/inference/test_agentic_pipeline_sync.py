from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from krisna_inference.backends.common import BlobStore
from krisna_inference.backends.planner_backend_llama import PlannerBackendLlama
from krisna_inference.backends.planner_schema import PlannerResponse
from krisna_inference.orchestrator.model_registry import ModelSpec, Tier


def test_planner_llama_prompt_incorporates_prior_critique():
    spec = ModelSpec(tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=6.5, quantization="GGUF")
    backend = PlannerBackendLlama(spec)

    constraints = {"style": "dark neo-brutalist", "palette": ["#000000", "#ffff00"]}
    prior_critique = {
        "overall_score": 0.72,
        "dimensions": {
            "visual_hierarchy": {"score": 0.65, "note": "CTA button is hard to discern."},
            "aesthetic": {"score": 0.80, "note": "Clean typography."},
        },
        "suggested_edits": [
            {"region": "hero_cta", "instruction": "Increase contrast and size of primary action button"},
            {"region": "navbar", "instruction": "Add subtle divider line below navigation"},
        ],
    }

    messages, retrieved = backend._prompt("Refine checkout flow", constraints, [], prior_critique, review=False)
    system_prompt = messages[0]["content"]

    assert "Prior review:" in system_prompt
    assert "0.72" in system_prompt
    assert "dark neo-brutalist" in system_prompt


def test_planner_schema_validation_with_constraint_updates():
    payload = {
        "schema_version": "planner.delta.v1",
        "reply_text": "Here is the planned design for the mobile checkout flow.",
        "design_state_delta": {
            "stage": "sketching",
            "constraint_updates": {"style": "minimalist", "palette": ["#ffffff", "#000000"]},
            "tool_call": None,
            "reasoning_note": "Focused on clear CTA.",
        },
    }
    response = PlannerResponse.model_validate(payload)
    assert response.reply_text == "Here is the planned design for the mobile checkout flow."
    assert response.design_state_delta.stage == "sketching"
    assert response.design_state_delta.constraint_updates["style"] == "minimalist"
    assert response.design_state_delta.constraint_updates["palette"] == ["#ffffff", "#000000"]
    assert response.design_state_delta.reasoning_note == "Focused on clear CTA."


def _png_bytes() -> bytes:
    pytest.importorskip("PIL")
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (2, 2), color="red").save(output, format="PNG")
    return output.getvalue()


def test_planner_prompt_attaches_base64_image_content_block():
    spec = ModelSpec(tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=6.5, quantization="GGUF")
    backend = PlannerBackendLlama(spec)
    image = _png_bytes()

    messages, _ = backend._prompt(
        "Review this wireframe",
        {},
        [],
        None,
        True,
        image_bytes=image,
    )

    content = messages[-1]["content"]
    assert content[0] == {"type": "text", "text": "Review this wireframe"}
    image_block = content[1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")
    encoded = image_block["image_url"]["url"].split(",", 1)[1]
    import base64

    assert base64.b64decode(encoded) == image


def test_planner_prompt_loads_blob_image_reference(monkeypatch, tmp_path):
    spec = ModelSpec(tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=6.5, quantization="GGUF")
    backend = PlannerBackendLlama(spec)
    store = BlobStore(root=tmp_path / "blobs")
    pytest.importorskip("PIL")
    from PIL import Image

    ref = store.save_image(Image.new("RGB", (2, 2), color="blue"), prefix="wireframe")

    monkeypatch.setattr(
        "krisna_inference.backends.planner_backend_llama.get_blob_store",
        lambda: store,
    )
    messages, _ = backend._prompt("Review", {}, [], None, True, wireframe_ref=ref)

    assert messages[-1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_strict_visual_review_rejects_missing_image(monkeypatch):
    spec = ModelSpec(tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=6.5, quantization="GGUF")
    backend = PlannerBackendLlama(spec)
    backend._loaded = True
    monkeypatch.setenv("KRISNA_USE_REAL_BACKENDS", "1")

    with pytest.raises(ValueError, match="requires a valid image"):
        await backend.run(review=True)


@pytest.mark.asyncio
async def test_strict_visual_review_rejects_missing_projector(monkeypatch):
    spec = ModelSpec(tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=6.5, quantization="GGUF")
    backend = PlannerBackendLlama(spec)
    backend._loaded = True
    backend._llama = object()
    monkeypatch.setenv("KRISNA_USE_REAL_BACKENDS", "1")

    with pytest.raises(RuntimeError, match="multimodal projector"):
        await backend.run(review=True, image_bytes=_png_bytes())


@pytest.mark.asyncio
async def test_strict_visual_review_rejects_invalid_image(tmp_path, monkeypatch):
    spec = ModelSpec(tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=6.5, quantization="GGUF")
    backend = PlannerBackendLlama(spec, mmproj_path=str(tmp_path / "mmproj.gguf"))
    backend._loaded = True
    backend._llama = object()
    Path(backend.mmproj_path).write_bytes(b"projector")
    monkeypatch.setenv("KRISNA_USE_REAL_BACKENDS", "1")

    with pytest.raises(ValueError, match="Invalid Planner review image"):
        await backend.run(review=True, image_bytes=b"not an image")
