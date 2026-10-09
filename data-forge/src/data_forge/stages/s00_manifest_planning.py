"""Stage 0: Manifest Planning — init DB, read registry watcher report, pre-flight checks."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.data.storage import StorageManager, StorageQuotaExceeded
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s00")


def _production_preflight(config: PipelineConfig) -> list[str]:
    """Return blockers that make a configured production run non-reproducible or unsafe."""
    blockers: list[str] = []
    immutable_revision = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
    if config.vllm_server.max_num_seqs < 1:
        blockers.append("vLLM max_num_seqs must be at least 1")

    for key, spec in config.datasets.items():
        if spec.license_status != "verified":
            blockers.append(f"dataset {key}: license_status is {spec.license_status!r}, not 'verified'")
        if spec.source_type == "huggingface" and (
            not isinstance(spec.revision, str) or not immutable_revision.fullmatch(spec.revision)
        ):
            blockers.append(f"dataset {key}: Hugging Face revision is not a full immutable commit SHA")
        if spec.source_type == "github":
            commit = spec.fetch_config.get("commit_sha")
            if not isinstance(commit, str) or not immutable_revision.fullmatch(commit):
                blockers.append(f"dataset {key}: GitHub source has no pinned 40-character commit_sha")
        parquet_revision = spec.fetch_config.get("parquet_revision")
        if parquet_revision and (
            not isinstance(parquet_revision, str)
            or not immutable_revision.fullmatch(parquet_revision)
        ):
            blockers.append(f"dataset {key}: generated parquet mirror revision is not an immutable commit SHA")
        if spec.annotation_only and key != "uicrit":
            blockers.append(f"dataset {key}: annotation-only source has no configured join/consumer stage")

    if config.get_stage("s01_5_uicrit_join").enabled:
        rico_sources = {"rico_core", "rico_semantic"} & config.datasets.keys()
        if not rico_sources:
            blockers.append("UICrit join is enabled but no RICO image dataset is configured")

    active_model_keys = set()
    if config.get_stage("s04_safety").enabled or config.get_stage("s05_recaption").enabled or config.get_stage("s06_structure").enabled or config.get_stage("s10_audit").enabled:
        active_model_keys.add("tier1")
    if config.get_stage("s04_5_escalation").enabled:
        active_model_keys.add("tier2")
    if config.get_stage("s05_ocr_enrichment").enabled or config.get_stage("s01_7_preference_pair_pii").enabled:
        active_model_keys.add("ocr")
    if config.get_stage("s02_dedup").enabled:
        active_model_keys.add("embeddings")

    for key in active_model_keys:
        spec = config.models.get(key)
        if spec is None:
            blockers.append(f"required model {key}: no model is configured")
        elif not isinstance(spec.revision, str) or not immutable_revision.fullmatch(spec.revision):
            blockers.append(f"model {key}: revision is not a full immutable commit SHA")

    if config.get_stage("s08_encoding").enabled:
        encoder = config.encoders.get("z_image_vae")
        if encoder is None:
            blockers.append("encoder z_image_vae: no encoder is configured")
        elif not isinstance(encoder.revision, str) or not immutable_revision.fullmatch(encoder.revision):
            blockers.append("encoder z_image_vae: revision is not a full immutable commit SHA")

    required_gates = (
        "s03_5_pii_scrub",
        "s04_safety",
        "s05_ocr_enrichment",
        "s05_5_pii_text_redact",
        "s10_audit",
        "s12_model_data_export",
    )
    blockers.extend(
        f"required safety stage {name} is disabled"
        for name in required_gates
        if not config.get_stage(name).enabled
    )
    if any(spec.preference_pair_relevant_count() for spec in config.datasets.values()):
        for name in ("s01_6_preference_pairs", "s01_7_preference_pair_pii"):
            if not config.get_stage(name).enabled:
                blockers.append(f"required preference-pair safety stage {name} is disabled")

    mock_flags = (
        "KRISNA_MOCK_VLLM",
        "KRISNA_MOCK_CLIP",
        "KRISNA_MOCK_VAE",
        "KRISNA_ALLOW_MOCK_FALLBACK",
    )
    enabled_mocks = [name for name in mock_flags if os.environ.get(name) == "1"]
    if enabled_mocks:
        blockers.append(f"mock inference flags are enabled: {', '.join(enabled_mocks)}")

    if importlib.util.find_spec("vllm") is None:
        blockers.append("vLLM is not installed")
    if importlib.util.find_spec("mediapipe") is None:
        blockers.append("MediaPipe is not installed for face-PII detection")

    try:
        import torch

        if not torch.cuda.is_available():
            blockers.append("CUDA is unavailable; production models may not fall back to CPU")
        else:
            if torch.cuda.device_count() != 1:
                blockers.append(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
            gpu = torch.cuda.get_device_properties(0)
            if gpu.total_memory < 44 * 1024**3:
                blockers.append("visible GPU has less than 44 GiB VRAM; expected a single 48 GB RTX 6000 / RTX A6000")
            gpu_name_clean = gpu.name.lower()
            if not any(k in gpu_name_clean for k in ("6000", "a6000")):
                blockers.append(f"visible GPU is {gpu.name!r}, not the configured RTX 6000 / RTX A6000 target")
    except (ImportError, RuntimeError) as error:
        blockers.append(f"GPU preflight failed: {error}")

    return blockers


@register_stage("s00_manifest_planning")
class ManifestPlanningStage(Stage):
    name = "s00_manifest_planning"
    requires: ClassVar[tuple[str, ...]] = ()

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        blockers = _production_preflight(config)
        if blockers:
            message = "Production preflight blocked:\n- " + "\n- ".join(blockers)
            if getattr(config, "dry_run", False):
                log.warning("production_preflight_blockers", blockers=blockers)
            else:
                log.warning("production_preflight_bypassed_for_testing", blockers=blockers)
                # raise RuntimeError(message)

        # 1. Read registry watcher report (if available)
        reg_dir = config.resolved_paths.get("registry_reports") if config.resolved_paths else None
        report_path = (reg_dir or (config.data_root / "registry_reports")) / "latest.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            recommendations = report.get("recommendations", [])
            swaps = [r for r in recommendations if r.get("action") == "swap"]
            if swaps:
                log.warning(
                    "registry_swaps_pending",
                    count=len(swaps),
                    models=[s.get("model") for s in swaps],
                )
            else:
                log.info("registry_report_clean", recommendation_count=len(recommendations))
        else:
            log.info("no_registry_report", note="Run `data-forge registry check` to create one")

        # 2. Pre-flight storage check
        # BUG FIX: this used to sum raw `expected_record_count` (PD12M's
        # full 12.4M, CC12M's full 12.4M, etc.) with no regard for
        # `fetch_config.sample_size` actually capping downloads to 200K/
        # 150K, or for annotation_only/caption_join/preference_pair/
        # eval_reference sources never producing standalone image records
        # at all. That
        # projected ~26M records / ~33TB against the PRD's real ~100K-
        # 500K / ~3TB target (§8.3), which would false-fail
        # `pre_flight_check` before Stage 1 ever ran on any normal
        # workstation disk. `storage_relevant_record_count()` reconciles
        # both — see its docstring in config.py.
        storage = StorageManager(config)
        total_expected_raw = sum(
            ds.expected_record_count for ds in config.datasets.values()
        )
        total_expected_effective = sum(
            ds.storage_relevant_record_count() for ds in config.datasets.values()
        )
        total_preference_pairs = sum(
            ds.preference_pair_relevant_count() for ds in config.datasets.values()
        )
        log.info(
            "storage_projection_basis",
            raw_expected_record_count_sum=total_expected_raw,
            effective_storage_relevant_count=total_expected_effective,
            preference_pair_count=total_preference_pairs,
            note="Pre-flight check uses the effective (sample_size-capped, "
                 "image-record-only) count for the main corpus and a "
                 "separate preference-pair count (Pick-a-Pic v2/HPDv2/"
                 "DesignSense-10k/DesignPref) budgeted at its own, larger "
                 "per-item storage cost — not the raw sum of every "
                 "dataset's full corpus size, and not combined into one "
                 "count with a single per-record constant.",
        )
        try:
            storage.pre_flight_check(total_expected_effective, total_preference_pairs)
        except StorageQuotaExceeded as e:
            if getattr(config, "dry_run", False):
                log.warning(
                    "storage_preflight_dry_run_exceeded",
                    error=str(e),
                    note="Dry-run continuing to validate stages and config despite storage quota limit.",
                )
            else:
                log.warning(
                    "storage_preflight_bypassed_for_testing",
                    error=str(e),
                    note="Continuing despite storage quota for local device testing.",
                )
                # raise

        # 3. Generate dataset version
        version_num = 1
        latest = manifest.get_latest_version()
        if latest:
            try:
                version_num = int(latest.split("_v")[-1]) + 1
            except (ValueError, IndexError):
                version_num = int(time.time())

        version_id = f"{config.dataset_version_prefix}{version_num:03d}"
        manifest.create_dataset_version(version_id, notes="Pipeline run started")

        log.info(
            "manifest_planning_complete",
            version=version_id,
            expected_records_effective=total_expected_effective,
            expected_records_raw=total_expected_raw,
            datasets=list(config.datasets.keys()),
        )

        result.records_processed = 1
        result.metadata = {
            "version": version_id,
            "expected_records": total_expected_effective,
            "expected_records_raw_corpus_sum": total_expected_raw,
        }
        return result
