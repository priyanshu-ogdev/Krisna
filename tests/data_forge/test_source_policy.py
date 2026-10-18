"""Focused tests for dataset provenance metadata and training-source policy."""

from __future__ import annotations

from pathlib import Path

from data_forge.config import (
    DatasetSpec,
    PipelineConfig,
    SourceEligibilityPolicy,
    load_config,
)
from data_forge.stages.s00_manifest_planning import _production_preflight


def test_dataset_spec_defaults_are_conservative() -> None:
    spec = DatasetSpec(display_name="unknown", source_type="url", category="unknown")

    assert spec.training_eligible is False
    assert spec.synthetic is False
    assert spec.synthetic_status == "unknown"
    assert spec.provenance_url is None
    assert spec.quality_tier == "unrated"
    assert spec.training_targets == []


def test_real_config_exposes_source_metadata_and_policy() -> None:
    config = load_config(
        pipeline_yaml=Path("data-forge/configs/pipeline.yaml"),
        models_yaml=Path("data-forge/configs/models.yaml"),
        datasets_yaml=Path("data-forge/configs/datasets.yaml"),
    )

    assert config.source_policy == SourceEligibilityPolicy(
        approved_model_training_exports=True,
        allow_synthetic=False,
        require_verified_license=True,
        require_immutable_revisions=True,
    )
    pd12m = config.datasets["pd12m"]
    # The checked-in catalog is a candidate registry; no source is approved
    # until its license and immutable revision receipt are verified.
    assert pd12m.training_eligible is False
    assert pd12m.is_synthetic is False
    assert pd12m.provenance_url == "https://huggingface.co/datasets/Spawning/PD12M"
    assert pd12m.quality_tier == "unrated"
    assert pd12m.training_targets == []
    assert pd12m.enabled is False

    websight = config.datasets["websight"]
    assert websight.training_eligible is False
    assert websight.is_synthetic is True
    assert websight.enabled is False

    # Preference sources are human-labelled/real according to their registry
    # notes, but remain ineligible until the license and immutable revision
    # gates are satisfied.  Do not conflate preference data with synthetic
    # data merely because it is deferred from the current export.
    for key in ("pickapic_v2", "hpdv2", "gamelabel_10k"):
        assert config.datasets[key].synthetic_status == "real"
        assert config.datasets[key].is_synthetic is False
        assert config.datasets[key].training_eligible is False
        assert config.datasets[key].enabled is False


def test_disabled_audit_entries_do_not_consume_capacity_or_pairs() -> None:
    disabled = DatasetSpec(
        display_name="Disabled",
        source_type="huggingface",
        category="web_ui",
        enabled=False,
        expected_record_count=1000,
        fetch_config={"download_mode": "preference_pair", "sample_size": 500},
    )

    assert disabled.storage_relevant_record_count() == 0
    assert disabled.preference_pair_relevant_count() == 0


def test_training_policy_requires_verified_license_and_immutable_revision() -> None:
    config = PipelineConfig(
        source_policy=SourceEligibilityPolicy(),
        datasets={
            "candidate": DatasetSpec(
                display_name="Candidate",
                source_type="huggingface",
                category="web_ui",
                training_eligible=True,
                synthetic_status="real",
                license_status="unverified",
                revision="main",
            )
        },
    )

    violations = config.training_policy_violations()
    assert violations == {
        "candidate": [
            "provenance_url must be configured",
            "license_status must be 'verified'",
            "source revision must be a full immutable commit SHA",
            "training_targets must assign the source to an export",
        ]
    }


def test_training_policy_rejects_synthetic_and_accepts_pinned_verified_source() -> None:
    config = PipelineConfig(
        datasets={
            "synthetic": DatasetSpec(
                display_name="Synthetic",
                source_type="huggingface",
                category="web_ui",
                training_eligible=True,
                synthetic=True,
                synthetic_status="synthetic",
                license_status="verified",
                revision="a" * 40,
                provenance_url="https://example.test/synthetic",
                training_targets=["polish_qwen_image21"],
            ),
            "approved": DatasetSpec(
                display_name="Approved",
                source_type="huggingface",
                category="web_ui",
                training_eligible=True,
                synthetic_status="real",
                license_status="verified",
                revision="b" * 40,
                provenance_url="https://example.test/approved",
                training_targets=["polish_qwen_image21"],
            ),
        }
    )

    assert config.training_policy_violations() == {
        "synthetic": ["synthetic source is not allowed"]
    }
    try:
        config.validate_training_policy()
    except ValueError as error:
        assert "synthetic" in str(error)
    else:  # pragma: no cover - protects the assertion if validation is removed
        raise AssertionError("synthetic source should fail the training policy")


def test_github_policy_uses_fetch_commit_sha() -> None:
    spec = DatasetSpec(
        display_name="GitHub source",
        source_type="github",
        category="ui_critique",
        training_eligible=True,
        synthetic_status="real",
        license_status="verified",
        branch="main",
        fetch_config={"commit_sha": "c" * 40},
        provenance_url="https://example.test/github-source",
        training_targets=["polish_qwen_image21"],
    )

    assert spec.immutable_revision() == "c" * 40
    assert spec.policy_violations(SourceEligibilityPolicy()) == []


def test_production_preflight_surfaces_opted_in_source_policy_failure() -> None:
    config = PipelineConfig(
        datasets={
            "candidate": DatasetSpec(
                display_name="Candidate",
                source_type="huggingface",
                category="web_ui",
                training_eligible=True,
                synthetic_status="real",
                license_status="unverified",
                revision="main",
            )
        }
    )

    blockers = _production_preflight(config)
    assert any("dataset candidate: training policy" in blocker for blocker in blockers)


def test_preflight_skips_training_gates_for_eval_and_annotation_sources() -> None:
    config = PipelineConfig(
        datasets={
            "eval": DatasetSpec(
                display_name="Eval",
                source_type="huggingface",
                category="eval",
                eval_only=True,
                revision="main",
                license_status="unverified",
            ),
            "annotations": DatasetSpec(
                display_name="Annotations",
                source_type="github",
                category="annotations",
                annotation_only=True,
                branch="main",
                license_status="unverified",
            ),
        }
    )

    blockers = _production_preflight(config)

    assert not any("dataset eval:" in blocker for blocker in blockers)
    assert not any("dataset annotations:" in blocker for blocker in blockers)


def test_uicrit_join_requires_annotation_target_and_fetch_prerequisites() -> None:
    from data_forge.config import StageConfig

    config = PipelineConfig(
        stages={"s01_5_uicrit_join": StageConfig(enabled=True)},
        datasets={
            "uicrit": DatasetSpec(
                display_name="UICrit",
                source_type="github",
                category="ui_critique",
                annotation_only=True,
            )
        },
    )

    blockers = _production_preflight(config)

    assert "UICrit join requires at least one target image dataset" not in blockers
    assert "UICrit join requires s01_fetch to be enabled" in blockers
    assert any("UICrit join target image datasets are not configured" in blocker for blocker in blockers)
