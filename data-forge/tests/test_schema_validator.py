"""Tests for the schema validator."""

from pathlib import Path

import pytest

from data_forge.data.schema_validator import SchemaValidator


@pytest.fixture
def validator() -> SchemaValidator:
    schemas_dir = Path(__file__).parent.parent / "configs" / "schemas"
    return SchemaValidator(schemas_dir)


class TestSchemaValidator:
    def test_valid_caption(self, validator: SchemaValidator):
        data = {
            "caption": "A mobile login screen with a large blue button and two input fields.",
            "ui_elements_mentioned": ["button", "text_input"],
            "confidence": 0.9,
        }
        valid, errors = validator.validate_caption(data)
        assert valid is True
        assert errors == []

    def test_invalid_caption_missing_field(self, validator: SchemaValidator):
        data = {"caption": "A short caption that is enough chars.", "confidence": 0.9}
        valid, errors = validator.validate_caption(data)
        assert valid is False
        assert any("ui_elements_mentioned" in e for e in errors)

    def test_valid_safety(self, validator: SchemaValidator):
        data = {
            "tier": "safe",
            "confidence": 0.95,
            "rationale": "Normal mobile app with standard UI elements.",
            "flags": [],
        }
        valid, errors = validator.validate_safety(data)
        assert valid is True

    def test_invalid_safety_bad_tier(self, validator: SchemaValidator):
        data = {
            "tier": "questionable",
            "confidence": 0.5,
            "rationale": "Some concerns here with the content.",
            "flags": [],
        }
        valid, errors = validator.validate_safety(data)
        assert valid is False

    def test_valid_license(self, validator: SchemaValidator):
        data = {
            "license_type": "Apache 2.0",
            "redistribution_allowed": True,
            "commercial_use_allowed": True,
            "attribution_required": True,
            "research_only": False,
            "confidence": 0.92,
            "source_citation": "https://example.com — 'Licensed under Apache 2.0'",
        }
        valid, errors = validator.validate_license(data)
        assert valid is True
