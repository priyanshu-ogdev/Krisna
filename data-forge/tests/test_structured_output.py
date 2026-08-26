"""Tests for structured output Pydantic models."""

import pytest

from data_forge.inference.structured_output import (
    CaptionOutput,
    LicenseOutput,
    QualityOutput,
    SafetyOutput,
    StructureOutput,
)


class TestCaptionOutput:
    def test_valid(self):
        data = {"caption": "A" * 25, "ui_elements_mentioned": ["button"], "confidence": 0.9}
        out = CaptionOutput.model_validate(data)
        assert out.confidence == 0.9

    def test_caption_too_short(self):
        with pytest.raises(Exception):
            CaptionOutput(caption="short", ui_elements_mentioned=["b"], confidence=0.5)

    def test_schema_generation(self):
        schema = CaptionOutput.model_json_schema()
        assert "caption" in schema["properties"]
        assert schema["required"] == ["caption", "ui_elements_mentioned", "confidence"]


class TestStructureOutput:
    def test_nested_elements(self):
        data = {
            "elements": [
                {"type": "container", "bbox": [0, 0, 1, 1], "children": [
                    {"type": "button", "bbox": [0.1, 0.1, 0.3, 0.15], "label": "Submit", "children": []},
                ]},
            ],
            "layout_type": "form",
            "hierarchy_depth": 2,
        }
        out = StructureOutput.model_validate(data)
        assert len(out.elements) == 1
        assert len(out.elements[0].children) == 1


class TestSafetyOutput:
    def test_valid_tiers(self):
        for tier in ["safe", "borderline", "unsafe"]:
            out = SafetyOutput(
                tier=tier, confidence=0.9,
                rationale="Test rationale that is long enough.", flags=[]
            )
            assert out.tier == tier

    def test_invalid_tier(self):
        with pytest.raises(Exception):
            SafetyOutput(tier="maybe", confidence=0.5, rationale="Hmm not sure about this.", flags=[])


class TestLicenseOutput:
    def test_conservative_defaults(self):
        out = LicenseOutput(
            license_type="Unknown", confidence=0.3,
            source_citation="No clear license text found on the page."
        )
        assert out.redistribution_allowed is False
        assert out.commercial_use_allowed is False
        assert out.research_only is False


class TestQualityOutput:
    def test_score_range(self):
        out = QualityOutput(
            aesthetic_score=0.75, resolution_adequate=True,
            is_complete_ui=True, design_era="modern", confidence=0.88
        )
        assert 0.0 <= out.aesthetic_score <= 1.0
