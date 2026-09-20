"""Tests for data-forge package structure, module exports, stage registration, and CLI resolution."""

from __future__ import annotations

import pytest


def test_top_level_package_exports():
    import data_forge

    assert data_forge.__version__ == "0.14.0"
    assert hasattr(data_forge, "Manifest")
    assert hasattr(data_forge, "PipelineConfig")
    assert hasattr(data_forge, "load_config")
    assert hasattr(data_forge, "Orchestrator")
    assert hasattr(data_forge, "Stage")
    assert hasattr(data_forge, "StageResult")
    assert hasattr(data_forge, "register_all_stages")
    assert hasattr(data_forge, "validate_stage_ordering")


def test_subpackage_exports():
    import data_forge.agents as agents
    import data_forge.data as data
    import data_forge.inference as inference
    import data_forge.registry as registry
    import data_forge.stages as stages
    import data_forge.utils as utils

    # data
    assert hasattr(data, "DedupEngine")
    assert hasattr(data, "DatasetFetcher")
    assert hasattr(data, "StorageManager")

    # inference
    assert hasattr(inference, "ModelEngine")
    assert hasattr(inference, "InferenceClient")
    assert hasattr(inference, "OCREngine")
    assert hasattr(inference, "CaptionOutput")

    # agents
    assert hasattr(agents, "AuditAgent")
    assert hasattr(agents, "LicenseAgent")
    assert hasattr(agents, "check_unsloth_support")

    # registry
    assert hasattr(registry, "RegistryWatcher")

    # utils
    assert hasattr(utils, "load_image")
    assert hasattr(utils, "hash_file")

    # stages
    assert hasattr(stages, "Stage")
    assert hasattr(stages, "StageResult")
    assert hasattr(stages, "register_all_stages")
    assert len(stages.STAGE_MODULES) >= 20


def test_s05_ocr_enrichment_modularization():
    from data_forge.orchestrator import get_registered_stages
    from data_forge.stages.s05_ocr_enrichment import OCREnrichmentStage
    from data_forge.stages.s05_recaption import OCREnrichmentStage as LegacyOCR

    assert OCREnrichmentStage is LegacyOCR
    reg = get_registered_stages()
    assert "s05_ocr_enrichment" in reg
    assert reg["s05_ocr_enrichment"] is OCREnrichmentStage
    assert OCREnrichmentStage.name == "s05_ocr_enrichment"
    assert "s06_structure" in OCREnrichmentStage.requires


def test_pipeline_config_version():
    from data_forge.cli import _find_configs_dir
    from data_forge.config import load_config

    cfg_dir = _find_configs_dir()
    config = load_config(
        pipeline_yaml=cfg_dir / "pipeline.yaml",
        models_yaml=cfg_dir / "models.yaml",
        datasets_yaml=cfg_dir / "datasets.yaml",
    )
    assert config.version == "0.14.0"


def test_stage_ordering_clean():
    from data_forge.orchestrator import validate_stage_ordering
    from data_forge.stages import register_all_stages

    register_all_stages()
    violations = validate_stage_ordering()
    assert violations == [], f"Unexpected stage ordering violations: {violations}"


def test_cli_stage_filtering_resolution():
    from click.testing import CliRunner
    from data_forge.cli import main

    runner = CliRunner()
    # Test --stages with canonical name and shorthand OCR in dry-run
    res = runner.invoke(main, ["run", "--dry-run", "--stages", "s02_dedup,5-ocr,12"])
    assert res.exit_code == 0, f"CLI invocation failed: {res.output}"
    assert "s02_dedup" in res.output
    assert "s05_ocr_enrichment" in res.output
    assert "s12_model_data_export" in res.output
