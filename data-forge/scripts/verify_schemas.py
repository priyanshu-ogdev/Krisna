"""Pre-flight schema verification.

Compares each hand-authored JSON Schema in configs/schemas/ against the
corresponding Pydantic model's generated schema. The static files are
deliberately simplified (draft-07, no titles/descriptions, additionalProperties:
false) compared to Pydantic's `model_json_schema()` output, so this does a
*structural* comparison (required fields + property types/enums) rather than
raw dict equality — a byte-for-byte diff would fail on every file regardless
of real drift, which is just as useless as the previous version's silent
"skip if the filename doesn't exist" bug.
"""

import json
import sys
from pathlib import Path

from data_forge.inference.structured_output import (
    AuditOutput,
    CaptionOutput,
    LicenseOutput,
    OCROutput,
    SafetyOutput,
    StructureOutput,
)

# Filenames must match what's actually in configs/schemas/ ("_output.json",
# not "_schema.json" — the old mapping pointed at files that don't exist,
# so every check was silently skipped via `continue` and the script always
# printed "All schemas perfectly match" regardless of real drift.
MODELS = {
    "caption_output.json": CaptionOutput,
    "safety_output.json": SafetyOutput,
    "structure_output.json": StructureOutput,
    "license_output.json": LicenseOutput,
    "ocr_output.json": OCROutput,
    "audit_output.json": AuditOutput,
}
# manifest_record.json is intentionally NOT checked here: it documents the
# manifest DB row shape (a plain dataclass in manifest.py), not a Pydantic
# structured-output model, so there's nothing to diff it against.


def _resolve(schema: dict, prop: dict) -> dict:
    """Resolve a $ref against the schema's $defs, if present."""
    if "$ref" in prop:
        ref_name = prop["$ref"].rsplit("/", 1)[-1]
        return schema.get("$defs", {}).get(ref_name, prop)
    return prop


def _prop_type_signature(schema: dict, prop: dict) -> str:
    prop = _resolve(schema, prop)
    if "type" in prop:
        t = prop["type"]
        if isinstance(t, list):
            return "anyOf:" + ",".join(sorted(str(x) for x in t))
        return str(t)
    if "enum" in prop:
        return "enum:" + ",".join(sorted(str(v) for v in prop["enum"]))
    if "anyOf" in prop:
        return "anyOf:" + ",".join(
            sorted(_prop_type_signature(schema, p) for p in prop["anyOf"])
        )
    return "unknown"


def compare(filename: str, static: dict, generated: dict) -> list[str]:
    """Return a list of human-readable structural mismatches (empty = OK)."""
    problems: list[str] = []

    static_required = set(static.get("required", []))
    generated_required = set(generated.get("required", []))
    if static_required != generated_required:
        problems.append(
            f"required fields differ: static={sorted(static_required)} "
            f"vs model={sorted(generated_required)}"
        )

    static_props = static.get("properties", {})
    generated_props = generated.get("properties", {})

    missing_in_model = set(static_props) - set(generated_props)
    extra_in_model = set(generated_props) - set(static_props)
    if missing_in_model:
        problems.append(f"properties in schema file but not in model: {sorted(missing_in_model)}")
    if extra_in_model:
        problems.append(f"properties in model but not in schema file: {sorted(extra_in_model)}")

    for name in set(static_props) & set(generated_props):
        s_sig = _prop_type_signature(static, static_props[name])
        g_sig = _prop_type_signature(generated, generated_props[name])
        if s_sig != g_sig:
            problems.append(f"property '{name}' type differs: static={s_sig} vs model={g_sig}")

    return problems


def main() -> None:
    base_dir = Path(__file__).parent.parent
    schemas_dir = base_dir / "configs" / "schemas"

    mismatches = 0
    missing = 0
    for filename, model in MODELS.items():
        schema_path = schemas_dir / filename
        if not schema_path.exists():
            print(f"ERROR: {filename} does not exist in configs/schemas/")
            missing += 1
            continue

        static_schema = json.loads(schema_path.read_text())
        generated_schema = model.model_json_schema()

        problems = compare(filename, static_schema, generated_schema)
        if problems:
            print(f"ERROR: Schema mismatch for {filename}")
            for p in problems:
                print(f"    - {p}")
            mismatches += 1
        else:
            print(f"OK: {filename} is structurally in sync.")

    if mismatches > 0 or missing > 0:
        print(f"\n{mismatches} mismatch(es), {missing} missing file(s).")
        sys.exit(1)
    else:
        print("All schemas structurally match their Pydantic models!")
        sys.exit(0)


if __name__ == "__main__":
    main()
