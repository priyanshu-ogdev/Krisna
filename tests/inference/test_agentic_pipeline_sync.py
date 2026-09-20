from __future__ import annotations

import pytest

from krisna_inference.backends.planner_backend import PlannerBackend
from krisna_inference.orchestrator.model_registry import ModelSpec, Tier


def test_planner_build_system_prompt_incorporates_prior_critique():
    spec = ModelSpec(tier=Tier.PLANNER, name="Qwen3.5-9B", vram_gb=6.5, quantization="NF4")
    backend = PlannerBackend(spec)

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

    system_prompt = backend._build_system_prompt(constraints, retrieved=[], prior_critique=prior_critique)

    assert "Prior Critic Feedback & Suggested Edits:" in system_prompt
    assert "Overall Score: 0.72" in system_prompt
    assert "visual_hierarchy: CTA button is hard to discern." in system_prompt
    assert "[hero_cta] Increase contrast and size of primary action button" in system_prompt
    assert "[navbar] Add subtle divider line below navigation" in system_prompt


def test_planner_extract_json_delta_with_constraint_updates():
    text = (
        "Here is the planned design for the mobile checkout flow.\n\n"
        '{"stage": "sketching", "constraint_updates": {"style": "minimalist", "palette": ["#ffffff", "#000000"]}, "tool_call": null, "reasoning_note": "Focused on clear CTA."}'
    )
    delta, error, span = PlannerBackend._extract_json_delta(text)
    assert error is None
    assert delta is not None
    assert delta["stage"] == "sketching"
    assert delta["constraint_updates"]["style"] == "minimalist"
    assert delta["constraint_updates"]["palette"] == ["#ffffff", "#000000"]
    assert delta["reasoning_note"] == "Focused on clear CTA."
    assert span is not None
