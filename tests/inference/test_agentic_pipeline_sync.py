from __future__ import annotations

import pytest

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
