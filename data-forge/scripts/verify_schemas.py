import json
import sys
from pathlib import Path
from data_forge.inference.structured_output import CaptionOutput, SafetyOutput, StructureOutput, LicenseOutput, OCROutput

def main():
    base_dir = Path(__file__).parent.parent
    schemas_dir = base_dir / "configs" / "schemas"
    
    models = {
        "caption_schema.json": CaptionOutput,
        "safety_schema.json": SafetyOutput,
        "structure_schema.json": StructureOutput,
        "license_schema.json": LicenseOutput,
        "ocr_schema.json": OCROutput
    }
    
    mismatches = 0
    for filename, model in models.items():
        schema_path = schemas_dir / filename
        if not schema_path.exists():
            print(f"Warning: {filename} does not exist in configs/schemas/")
            continue
            
        static_schema = json.loads(schema_path.read_text())
        generated_schema = model.model_json_schema()
        
        # Fast equality check
        if static_schema != generated_schema:
            print(f"ERROR: Schema mismatch for {filename}")
            mismatches += 1
        else:
            print(f"OK: {filename} is perfectly synced.")
            
    if mismatches > 0:
        sys.exit(1)
    else:
        print("All schemas perfectly match Pydantic models!")
        sys.exit(0)

if __name__ == "__main__":
    main()
