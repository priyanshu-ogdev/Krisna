"""Pipeline orchestrator with chunk-based model swapping.

Processes the dataset in chunks (default 10,000 images). For each chunk, it loads
a model ONCE, runs all applicable stages for the whole chunk, unloads, and loads
the next model. This minimizes PCIe model-swapping overhead on 48GB VRAM.

Execution phases per chunk:
  Phase 1: CLIP embeddings (dedup)
  Phase 2: Tier-1 VLM (quality, PII/OCR-text, safety, recaption, structure)
  Phase 3: OCR specialist (text extraction)
  Phase 4: Tier-2 VLM (escalation — borderline records only)
  Phase 5: Encoders (Tri-Path VAE/VQ encoding)

Between chunks, each phase tears down its model subprocess to prevent CUDA
memory fragmentation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest

log = get_logger("orchestrator")


@dataclass
class StageResult:
    """Result from running a single stage."""

    stage_name: str
    records_processed: int = 0
    records_failed: int = 0
    records_excluded: int = 0
    duration_seconds: float = 0.0
    metadata: dict[str, Any] | None = None

    @property
    def success(self) -> bool:
        return self.records_failed == 0


@dataclass
class ChunkResult:
    """Aggregated results from processing one chunk through all phases."""

    chunk_id: str
    record_ids: list[str]
    stage_results: list[StageResult]
    duration_seconds: float = 0.0


class Orchestrator:
    """Chunk-based pipeline orchestrator.

    Usage:
        config = load_config()
        manifest = Manifest(db_path)
        orch = Orchestrator(config, manifest)
        await orch.execute_pipeline()
    """

    def __init__(self, config: PipelineConfig, manifest: Manifest) -> None:
        self.config = config
        self.manifest = manifest
        self._stages: dict[str, Any] = {}  # Lazy-loaded stage instances
        self._checkpoint_dir = config.data_root / config.paths.checkpoints

    def _checkpoint_path(self, stage_name: str, chunk_id: str) -> Path:
        return self._checkpoint_dir / f"{stage_name}_{chunk_id}.done"

    def _is_stage_complete(self, stage_name: str, chunk_id: str) -> bool:
        return self._checkpoint_path(stage_name, chunk_id).exists()

    def _mark_stage_complete(self, stage_name: str, chunk_id: str) -> None:
        path = self._checkpoint_path(stage_name, chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"stage": stage_name, "chunk": chunk_id, "time": time.time()}),
            encoding="utf-8",
        )

    def _get_stage(self, stage_name: str) -> Any:
        """Lazy-load a stage class instance."""
        if stage_name not in self._stages:
            stage_cls = _STAGE_REGISTRY.get(stage_name)
            if stage_cls is None:
                raise ValueError(f"Unknown stage: {stage_name}")
            self._stages[stage_name] = stage_cls()
        return self._stages[stage_name]

    async def _run_stage(
        self,
        stage_name: str,
        record_ids: list[str],
        chunk_id: str,
        engine: Any | None = None,
    ) -> StageResult:
        """Run a single stage on a set of record IDs."""
        stage_config = self.config.get_stage(stage_name)
        if not stage_config.enabled:
            log.info("stage_skipped", stage=stage_name, reason="disabled")
            return StageResult(stage_name=stage_name)

        if self.config.checkpoint_enabled and self._is_stage_complete(stage_name, chunk_id):
            log.info("stage_skipped", stage=stage_name, chunk=chunk_id, reason="checkpoint_exists")
            return StageResult(stage_name=stage_name)

        log.info(
            "stage_starting",
            stage=stage_name,
            chunk=chunk_id,
            record_count=len(record_ids),
        )
        start = time.monotonic()

        stage = self._get_stage(stage_name)
        result = await stage.run(
            manifest=self.manifest,
            config=self.config,
            record_ids=record_ids,
            engine=engine,
        )

        result.duration_seconds = time.monotonic() - start

        if self.config.checkpoint_enabled:
            self._mark_stage_complete(stage_name, chunk_id)

        log.info(
            "stage_completed",
            stage=stage_name,
            chunk=chunk_id,
            processed=result.records_processed,
            failed=result.records_failed,
            excluded=result.records_excluded,
            duration_s=round(result.duration_seconds, 2),
        )
        return result

    async def execute_pipeline(
        self,
        stages_filter: list[str] | None = None,
        dry_run: bool = False,
        resume: bool = True,
    ) -> list[ChunkResult]:
        """Execute the full pipeline with chunk-based model swapping.

        Args:
            stages_filter: If set, only run these stages (by name).
            dry_run: Validate config and walk stages without running inference.
            resume: If True, skip chunks/stages that have completion checkpoints.
        """
        log.info(
            "pipeline_starting",
            version=self.config.version,
            chunk_size=self.config.chunk_size,
            dry_run=dry_run,
            resume=resume,
        )
        pipeline_start = time.monotonic()

        # ── Stage 0: Manifest Planning (runs once, not per-chunk) ────────
        if self._should_run("s00_manifest_planning", stages_filter):
            await self._run_stage(
                "s00_manifest_planning",
                record_ids=[],
                chunk_id="global",
            )

        # ── Stage 1: Fetch (runs once, populates manifest) ──────────────
        if self._should_run("s01_fetch", stages_filter):
            await self._run_stage(
                "s01_fetch",
                record_ids=[],
                chunk_id="global",
            )

        # ── Chunk-based processing ──────────────────────────────────────
        chunks = self.manifest.split_into_chunks(self.config.chunk_size)
        if not chunks:
            log.warning("no_records_to_process")
            return []

        log.info("chunks_planned", count=len(chunks), chunk_size=self.config.chunk_size)

        chunk_results: list[ChunkResult] = []

        for i, record_ids in enumerate(chunks):
            chunk_id = f"chunk_{i:04d}"
            chunk_start = time.monotonic()
            stage_results: list[StageResult] = []

            log.info(
                "chunk_starting",
                chunk=chunk_id,
                records=len(record_ids),
                progress=f"{i + 1}/{len(chunks)}",
            )

            if dry_run:
                log.info("chunk_dry_run", chunk=chunk_id, records=len(record_ids))
                continue

            # ── Phase 1: Embedding models (CLIP for dedup) ──────────
            if self._should_run("s02_dedup", stages_filter):
                from data_forge.inference.engine import ModelEngine
                async with ModelEngine.clip_session(self.config) as engine:
                    result = await self._run_stage(
                        "s02_dedup", record_ids, chunk_id, engine
                    )
                    stage_results.append(result)
                    # Filter out excluded records for downstream stages
                    record_ids = self._filter_active(record_ids)

            # ── Phase 2: Tier-1 VLM ─────────────────────────────────
            tier1_stages = [
                "s03_quality",
                "s03_5_pii_scrub",
                "s04_safety",
                "s05_recaption",
                "s06_structure",
            ]
            runnable_tier1 = [s for s in tier1_stages if self._should_run(s, stages_filter)]

            if runnable_tier1:
                from data_forge.inference.engine import ModelEngine

                # PII scrub uses MediaPipe, not vLLM — but runs in the Tier-1 phase
                # because it needs to happen between quality and safety
                if "s03_5_pii_scrub" in runnable_tier1:
                    result = await self._run_stage(
                        "s03_5_pii_scrub", record_ids, chunk_id
                    )
                    stage_results.append(result)
                    runnable_tier1.remove("s03_5_pii_scrub")

                vlm_stages = [s for s in runnable_tier1 if s != "s03_5_pii_scrub"]
                if vlm_stages:
                    async with ModelEngine.vllm_session(self.config, "tier1") as engine:
                        for stage_name in vlm_stages:
                            record_ids = self._filter_active(record_ids)
                            if not record_ids:
                                break
                            result = await self._run_stage(
                                stage_name, record_ids, chunk_id, engine
                            )
                            stage_results.append(result)

            # ── Phase 3: OCR Specialist ─────────────────────────────
            if self._should_run("s05_recaption", stages_filter):
                ocr_config = self.config.get_stage("s05_recaption")
                if ocr_config.get("ocr_enrichment", True):
                    from data_forge.inference.engine import ModelEngine
                    record_ids = self._filter_active(record_ids)
                    if record_ids:
                        async with ModelEngine.vllm_session(self.config, "ocr") as engine:
                            result = await self._run_stage(
                                "s05_ocr_enrichment", record_ids, chunk_id, engine
                            )
                            stage_results.append(result)

            # ── Phase 4: Tier-2 Escalation ──────────────────────────
            if self._should_run("s04_5_escalation", stages_filter):
                # Only escalated records need Tier-2
                borderline_ids = self._get_borderline_ids(record_ids)
                if borderline_ids:
                    from data_forge.inference.engine import ModelEngine
                    async with ModelEngine.vllm_session(self.config, "tier2") as engine:
                        result = await self._run_stage(
                            "s04_5_escalation", borderline_ids, chunk_id, engine
                        )
                        stage_results.append(result)

            # ── Deterministic stages (no GPU model needed) ──────────
            for stage_name in ["s07_routing"]:
                if self._should_run(stage_name, stages_filter):
                    record_ids = self._filter_active(record_ids)
                    if record_ids:
                        result = await self._run_stage(
                            stage_name, record_ids, chunk_id
                        )
                        stage_results.append(result)

            # ── Phase 5: Tri-Path Encoding (VAEs/VQ) ────────────────
            if self._should_run("s08_encoding", stages_filter):
                record_ids = self._filter_active(record_ids)
                if record_ids:
                    from data_forge.inference.engine import ModelEngine
                    async with ModelEngine.encoder_session(self.config) as engine:
                        result = await self._run_stage(
                            "s08_encoding", record_ids, chunk_id, engine
                        )
                        stage_results.append(result)

            chunk_duration = time.monotonic() - chunk_start
            chunk_results.append(
                ChunkResult(
                    chunk_id=chunk_id,
                    record_ids=record_ids,
                    stage_results=stage_results,
                    duration_seconds=chunk_duration,
                )
            )
            log.info(
                "chunk_completed",
                chunk=chunk_id,
                duration_s=round(chunk_duration, 2),
                stages_run=len(stage_results),
            )

        # ── Post-chunk global stages ────────────────────────────────
        all_record_ids = [rid for chunk in chunks for rid in chunk]

        if self._should_run("s09_heldout", stages_filter):
            await self._run_stage(
                "s09_heldout", all_record_ids, "global"
            )

        if self._should_run("s10_audit", stages_filter):
            # Audit runs on training_pool records only
            training_ids = [r.id for r in self.manifest.get_training_pool()]
            if training_ids:
                from data_forge.inference.engine import ModelEngine
                async with ModelEngine.vllm_session(self.config, "tier1") as engine:
                    await self._run_stage(
                        "s10_audit", training_ids, "global", engine
                    )

        pipeline_duration = time.monotonic() - pipeline_start
        stats = self.manifest.stats()
        log.info(
            "pipeline_completed",
            duration_s=round(pipeline_duration, 2),
            total_records=stats["total_records"],
            training_pool=stats["training_pool_count"],
            heldout=stats["heldout_count"],
            excluded=stats["excluded_count"],
        )

        return chunk_results

    def _should_run(self, stage_name: str, stages_filter: list[str] | None) -> bool:
        if stages_filter is not None:
            return stage_name in stages_filter
        return self.config.get_stage(stage_name).enabled

    def _filter_active(self, record_ids: list[str]) -> list[str]:
        """Return only record IDs that are not in a terminal/excluded status."""
        if not record_ids:
            return []
        records = self.manifest.get_records_by_ids(record_ids)
        from data_forge.manifest import TERMINAL_STATUSES
        return [r.id for r in records if r.status not in TERMINAL_STATUSES]

    def _get_borderline_ids(self, record_ids: list[str]) -> list[str]:
        """Get records that need Tier-2 escalation (borderline safety or pending review)."""
        records = self.manifest.get_records_by_ids(record_ids)
        return [
            r.id for r in records
            if r.safety_tier == "borderline"
            or r.status == "excluded_pending_review"
        ]


# ── Stage Registry ──────────────────────────────────────────────────────────
# Maps stage names to their implementation classes. Populated via imports
# to avoid circular dependencies.

_STAGE_REGISTRY: dict[str, type] = {}


def register_stage(name: str):  # type: ignore[no-untyped-def]
    """Decorator to register a stage class in the global registry."""

    def decorator(cls: type) -> type:
        _STAGE_REGISTRY[name] = cls
        return cls

    return decorator


def get_registered_stages() -> dict[str, type]:
    """Return a copy of the stage registry."""
    return dict(_STAGE_REGISTRY)
