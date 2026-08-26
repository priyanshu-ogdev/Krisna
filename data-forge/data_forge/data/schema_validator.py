"""JSON Schema validation for all model outputs and manifest records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from data_forge.logging_setup import get_logger

log = get_logger("data.schema_validator")


class SchemaValidator:
    """Validates data against JSON Schema files in configs/schemas/."""

    def __init__(self, schemas_dir: Path) -> None:
        self._schemas_dir = schemas_dir
        self._cache: dict[str, dict[str, Any]] = {}

    def _load_schema(self, name: str) -> dict[str, Any]:
        if name not in self._cache:
            schema_path = self._schemas_dir / f"{name}.json"
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema not found: {schema_path}")
            self._cache[name] = json.loads(schema_path.read_text(encoding="utf-8"))
        return self._cache[name]

    def validate(
        self, data: dict[str, Any], schema_name: str
    ) -> tuple[bool, list[str]]:
        """Validate data against a named schema.

        Returns:
            (is_valid, list_of_errors)
        """
        schema = self._load_schema(schema_name)
        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(data))

        if errors:
            error_msgs = []
            for err in errors:
                path = ".".join(str(p) for p in err.absolute_path) or "(root)"
                error_msgs.append(f"{path}: {err.message}")
            log.warning(
                "schema_validation_failed",
                schema=schema_name,
                error_count=len(errors),
                first_error=error_msgs[0] if error_msgs else "",
            )
            return False, error_msgs

        return True, []

    def validate_caption(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        return self.validate(data, "caption_output")

    def validate_structure(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        return self.validate(data, "structure_output")

    def validate_safety(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        return self.validate(data, "safety_output")

    def validate_license(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        return self.validate(data, "license_output")

    def validate_audit(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        return self.validate(data, "audit_output")

    def validate_ocr(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        return self.validate(data, "ocr_output")
