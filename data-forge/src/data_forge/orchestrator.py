"""Pipeline orchestrator with chunk-based model swapping.

Processes the dataset in configured chunks (50,000 records in production; 10,000
in the local override). For each chunk, it loads a model once, runs bounded
inference windows through applicable stages, and then swaps models. This
amortizes model startup overhead while keeping request and image batches bounded.

Execution phases per chunk:
  Phase 1: CLIP embeddings (dedup)
  Phase 2: Tier-1 VLM (quality, PII/OCR-text redaction, safety)
  Phase 3: Tier-2 VLM (escalation — borderline records only)
  Phase 4: Tier-1 VLM (recaption, structure)
  Phase 5: OCR specialist (text extraction) + text-PII redaction
  Phase 6: Encoders (Tri-Path VAE/VQ encoding)

Phases 2 and 4 were one combined phase until this order was found to
silently drop every record Tier-2 escalation rescues — recaption/
structure's safety_tier=="safe" filter only sees a rescued record if it
runs AFTER escalation resolves it, not in the same pass as the initial
(pre-escalation) safety classification. See
docs/review/16_preprocessing_ordering_audit.md for the full account and
the regression test (tests/data_forge/test_orchestrator.py) that
verifies the real call order rather than just each stage's own filter
logic in isolation. (This docstring itself had drifted stale relative
to the already-fixed code below — found and corrected during a later
merge; see docs/review/21_frontend_and_scripts_merge.md.)

Between chunks, each phase tears down its model subprocess to prevent CUDA
memory fragmentation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
# BUG FIX: this module used to define its own separate `StageResult`
# dataclass with the identical field set as stages/base.py's `StageResult`
# — every stage subclass (s00-s11) constructs and returns the base.py
# version, while orchestrator.py's `_run_stage` and `ChunkResult` were
# type-hinted against its own local, drifted copy. They happened to stay
# structurally identical so nothing broke yet, but this is exactly the kind
# of duplication that silently diverges the next time either dataclass gets
# a new field (e.g. the default `metadata` value already differed: `None`
# here vs. `field(default_factory=dict)` in base.py). Import the one
# canonical definition instead of shadowing it.
from data_forge.stages.base import StageResult
from data_forge.utils.audit_gate import pipeline_config_fingerprint

log = get_logger("orchestrator")


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
        self._resume = True
        self._cached_pipeline_fingerprint: str | None = None

    def _checkpoint_path(self, stage_name: str, chunk_id: str) -> Path:
        return self._checkpoint_dir / f"{stage_name}_{chunk_id}.done"

    def _pipeline_fingerprint(self) -> str:
        if self._cached_pipeline_fingerprint is None:
            self._cached_pipeline_fingerprint = pipeline_config_fingerprint(self.config)
        return self._cached_pipeline_fingerprint

    def _input_fingerprint(self, record_ids: list[str], stage_name: str) -> str:
        digest = hashlib.sha256()
        if not record_ids:
            state = self.manifest._conn.execute(
                "SELECT COUNT(*) AS record_count, MAX(updated_at) AS latest_update FROM records"
            ).fetchone()
            digest.update(
                f"{state['record_count']}\0{state['latest_update']}\n".encode("utf-8")
            )
            if stage_name in {
                "s01_6_preference_pairs",
                "s01_7_preference_pair_pii",
                "s08_5_dpo_encoding",
            }:
                roots = [self.config.resolved_paths.get("preference_pairs")]
                if stage_name == "s08_5_dpo_encoding":
                    roots.append(self.config.resolved_paths.get("dpo_latents"))
                for root in roots:
                    if root and root.exists():
                        for path in sorted(item for item in root.rglob("*") if item.is_file()):
                            stat = path.stat()
                            digest.update(
                                f"{path.relative_to(root)}\0{stat.st_size}\0"
                                f"{stat.st_mtime_ns}\n".encode("utf-8")
                            )
        else:
            for start in range(0, len(record_ids), 800):
                batch = record_ids[start : start + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = self.manifest._conn.execute(
                    f"SELECT id, status, updated_at FROM records WHERE id IN ({placeholders})",
                    batch,
                ).fetchall()
                for row in sorted(rows, key=lambda item: item["id"]):
                    digest.update(f"{row['id']}\0{row['status']}\0{row['updated_at']}\n".encode("utf-8"))
        return digest.hexdigest()

    def _is_stage_complete(
        self, stage_name: str, chunk_id: str, record_ids: list[str]
    ) -> bool:
        path = self._checkpoint_path(stage_name, chunk_id)
        if stage_name in {"s00_manifest_planning", "s12_model_data_export"} or not path.is_file():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            data.get("pipeline_fingerprint") == self._pipeline_fingerprint()
            and data.get("input_fingerprint") == self._input_fingerprint(record_ids, stage_name)
        )

    def _mark_stage_complete(
        self, stage_name: str, chunk_id: str, record_ids: list[str]
    ) -> None:
        path = self._checkpoint_path(stage_name, chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(json.dumps({
            "stage": stage_name,
            "chunk": chunk_id,
            "time": time.time(),
            "pipeline_fingerprint": self._pipeline_fingerprint(),
            "input_fingerprint": self._input_fingerprint(record_ids, stage_name),
        }), encoding="utf-8")
        temp_path.replace(path)

    def _get_stage(self, stage_name: str) -> Any:
        """Lazy-load a stage class instance."""
        if not _STAGE_REGISTRY:
            from data_forge.stages import register_all_stages

            register_all_stages()
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

        if (
            self._resume
            and self.config.checkpoint_enabled
            and self._is_stage_complete(stage_name, chunk_id, record_ids)
        ):
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

        if self.config.fail_fast and not result.success:
            log.error(
                "stage_failed_fail_fast",
                stage=stage_name,
                chunk=chunk_id,
                failed=result.records_failed,
            )
            raise RuntimeError(
                f"Stage {stage_name} ({chunk_id}) failed with {result.records_failed} "
                f"failed records (fail_fast=True)"
            )

        if (
            self.config.checkpoint_enabled
            and result.success
            and not (
                stage_name == "s10_audit"
                and result.metadata.get("pipeline_passes") is False
            )
        ):
            self._mark_stage_complete(stage_name, chunk_id, record_ids)

        log.info(
            "stage_completed",
            stage=stage_name,
            chunk=chunk_id,
            processed=result.records_processed,
            failed=result.records_failed,
            excluded=result.records_excluded,
            duration_s=round(result.duration_seconds, 2),
            processed_per_second=round(
                result.records_processed / result.duration_seconds, 3
            ) if result.duration_seconds else 0.0,
            submitted_records=len(record_ids),
        )
        return result

    async def execute_pipeline(
        self,
        stages_filter: list[str] | None = None,
        dry_run: bool = False,
        resume: bool = True,
        limit: int | None = None,
    ) -> list[ChunkResult]:
        """Execute the full pipeline with chunk-based model swapping.

        Args:
            stages_filter: If set, only run these stages (by name).
            dry_run: Validate config and walk stages without running inference.
            resume: If True, skip chunks/stages that have completion checkpoints.
            limit: If set, cap the total number of records processed this run —
                for smoke tests (e.g. `--chunk-size 100 --limit 100`, documented
                in README.md's walkthrough, which this parameter previously had
                no way to satisfy: the CLI didn't expose it and this method
                didn't accept it, so that exact documented command would have
                failed with a Click "no such option" error before reaching any
                pipeline logic at all).
        """
        log.info(
            "pipeline_starting",
            version=self.config.version,
            chunk_size=self.config.chunk_size,
            dry_run=dry_run,
            resume=resume,
        )
        self._resume = resume
        self.config.dry_run = dry_run
        pipeline_start = time.monotonic()

        # ── Stage 0: Manifest Planning (runs once, not per-chunk) ────────
        if self._should_run("s00_manifest_planning", stages_filter):
            await self._run_stage(
                "s00_manifest_planning",
                record_ids=[],
                chunk_id="global",
            )

        # ── Stage 1: Fetch (runs once, populates manifest) ──────────────
        # Runs inside a live Tier-1 vLLM session so the inline License
        # Verification Agent actually has a model to call (previously this
        # ran with engine=None, silently skipping every license check).
        if self._should_run("s01_fetch", stages_filter):
            if dry_run:
                log.info("stage_dry_run", stage="s01_fetch")
            else:
                from data_forge.inference.engine import ModelEngine
                async with ModelEngine.vllm_session(self.config, "tier1") as engine:
                    await self._run_stage(
                        "s01_fetch",
                        record_ids=[],
                        chunk_id="global",
                        engine=engine,
                    )

        # ── Stage 1.5: UICrit Join (runs once, global, after fetch) ─────
        # No VLM needed — pure parsing + in-memory filename-stem matching
        # against already-ingested RICO records. See
        # stages/s01_5_uicrit_join.py's module docstring for why this
        # exists as its own stage rather than folding into s01_fetch: it
        # depends on RICO's records already being in the manifest, which
        # only holds true once the whole (unchunked) fetch phase above has
        # fully completed for every dataset, not per-chunk.
        if self._should_run("s01_5_uicrit_join", stages_filter):
            if dry_run:
                log.info("stage_dry_run", stage="s01_5_uicrit_join")
            else:
                await self._run_stage(
                    "s01_5_uicrit_join",
                    record_ids=[],
                    chunk_id="global",
                    engine=None,
                )

        # ── Stage 1.6: Preference-Pair Post-Processing (runs once, global) ─
        # Dedup + face-blur (MediaPipe, same utility s03_5_pii_scrub uses)
        # PLUS Tier-1 safety classification — same NSFW/harmful-content gate
        # the main image manifest gets in s04_safety.py, applied here
        # directly rather than routing pairs through the manifest (they
        # never enter it — see fetcher.py's "not manifest records" note).
        # BUG FIX: an earlier revision of this stage ran with engine=None
        # and never classified preference-pair images for safety at all —
        # a real gap, since Pick-a-Pic/HPDv2 images are T2I-model outputs,
        # not manifest-vetted content, and nothing else in this pipeline
        # screens them. No AI-judge *labeling* happens here (these are real
        # human comparisons) — this is content-safety filtering, a
        # different, non-optional check, not the RLHF-loop pattern this
        # project rules out.
        if self._should_run("s01_6_preference_pairs", stages_filter):
            if dry_run:
                log.info("stage_dry_run", stage="s01_6_preference_pairs")
            else:
                pref_root = (
                    self.config.resolved_paths.get("preference_pairs")
                    if self.config.resolved_paths
                    else (self.config.data_root / "preference_pairs")
                )
                has_pairs = pref_root and pref_root.exists() and any(pref_root.glob("*/*.json"))
                if has_pairs:
                    from data_forge.inference.engine import ModelEngine
                    async with ModelEngine.vllm_session(self.config, "tier1") as engine:
                        await self._run_stage(
                            "s01_6_preference_pairs",
                            record_ids=[],
                            chunk_id="global",
                            engine=engine,
                        )
                else:
                    log.info("s01_6_skipped_no_pairs", note="No preference pairs found on disk to process")

        if self._should_run("s01_7_preference_pair_pii", stages_filter):
            pref_root = self.config.resolved_paths.get("preference_pairs")
            has_pairs = pref_root and pref_root.exists() and any(pref_root.glob("*/*.json"))
            if has_pairs:
                from data_forge.inference.engine import ModelEngine
                async with ModelEngine.vllm_session(self.config, "ocr") as engine:
                    await self._run_stage(
                        "s01_7_preference_pair_pii",
                        record_ids=[],
                        chunk_id="global",
                        engine=engine,
                    )

        if dry_run:
            log.info("dry_run_pipeline_completed", stages_checked=stages_filter or "all")
            return []

        # ── Chunk-based processing ──────────────────────────────────────
        chunks = self.manifest.split_into_chunks(self.config.chunk_size)

        if limit is not None:
            flat_ids = [rid for chunk in chunks for rid in chunk][:limit]
            chunks = [
                flat_ids[i : i + self.config.chunk_size]
                for i in range(0, len(flat_ids), self.config.chunk_size)
            ]
            log.info("record_limit_applied", limit=limit, resulting_chunks=len(chunks))

        if not chunks:
            post_chunk_stages = {
                "s08_5_dpo_encoding",
                "s09_heldout",
                "s10_audit",
                "s11_registry_watcher",
                "s12_model_data_export",
            }
            if not any(self._should_run(s, stages_filter) for s in post_chunk_stages):
                log.warning("no_records_to_process")
                return []

        log.info("chunks_planned", count=len(chunks), chunk_size=self.config.chunk_size)

        chunk_results: list[ChunkResult] = [
            ChunkResult(chunk_id=f"chunk_{i:04d}", record_ids=rec_ids, stage_results=[])
            for i, rec_ids in enumerate(chunks)
        ]

        if dry_run:
            return chunk_results

        # ── Phase 1: Embedding models (CLIP for dedup) ──────────
        if self._should_run("s02_dedup", stages_filter):
            from data_forge.inference.engine import ModelEngine
            from rich.progress import track
            async with ModelEngine.clip_session(self.config) as engine:
                for i, record_ids in track(enumerate(chunks), total=len(chunks), description="Phase 1: Dedup"):
                    chunk_id = f"chunk_{i:04d}"
                    log.info("chunk_starting", chunk=chunk_id, phase="1_dedup", records=len(record_ids))
                    res = await self._run_stage("s02_dedup", record_ids, chunk_id, engine)
                    chunk_results[i].stage_results.append(res)

        # ── Phase 2, 3, 4, 5: VLM Processing ────────────────────────────
        tier1_stages = ["s03_quality", "s03_5_pii_scrub", "s04_safety"]
        runnable_tier1 = [s for s in tier1_stages if self._should_run(s, stages_filter)]
        recaption_structure_stages = [
            s for s in ["s05_recaption", "s06_structure"] if self._should_run(s, stages_filter)
        ]
        run_ocr = (
            self._should_run("s05_ocr_enrichment", stages_filter)
            or (
                stages_filter is not None
                and "s05_recaption" in stages_filter
                and self.config.get_stage("s05_recaption").get("ocr_enrichment", True)
            )
        ) and self.config.get_stage("s05_ocr_enrichment").enabled

        ocr_shares_tier1 = bool(
            self.config.models.get("ocr")
            and self.config.models.get("tier1")
            and self.config.models["ocr"].model_id == self.config.models["tier1"].model_id
        )

        from data_forge.inference.engine import ModelEngine
        from rich.progress import track

        async def _exec_phase2(engine: Any) -> None:
            if not runnable_tier1:
                return
            for i, record_ids in track(enumerate(chunks), total=len(chunks), description="Phase 2: Tier-1 VLM"):
                chunk_id = f"chunk_{i:04d}"
                active_ids = self._filter_active(record_ids)
                if not active_ids:
                    continue
                log.info("chunk_starting", chunk=chunk_id, phase="2_tier1_eval", records=len(active_ids))
                for idx in range(0, len(active_ids), 4096):
                    subchunk_ids = active_ids[idx : idx + 4096]
                    for stage_name in runnable_tier1:
                        subchunk_ids = self._filter_active(subchunk_ids)
                        if not subchunk_ids:
                            break
                        stage_engine = None if stage_name == "s03_5_pii_scrub" else engine
                        res = await self._run_stage(stage_name, subchunk_ids, f"{chunk_id}_{idx}", stage_engine)
                        chunk_results[i].stage_results.append(res)

        async def _exec_phase3_escalation(engine: Any) -> None:
            for i, record_ids in track(enumerate(chunks), total=len(chunks), description="Phase 3: Tier-2 Escalation"):
                chunk_id = f"chunk_{i:04d}"
                borderline_ids = self._get_borderline_ids(record_ids)
                if borderline_ids:
                    log.info("chunk_starting", chunk=chunk_id, phase="3_escalation", records=len(borderline_ids))
                    for idx in range(0, len(borderline_ids), 4096):
                        subchunk_ids = borderline_ids[idx : idx + 4096]
                        res = await self._run_stage("s04_5_escalation", subchunk_ids, f"{chunk_id}_{idx}", engine)
                        chunk_results[i].stage_results.append(res)

        async def _exec_phase4(engine: Any) -> None:
            if not recaption_structure_stages:
                return
            stages_to_run = list(recaption_structure_stages)
            if ocr_shares_tier1 and run_ocr and "s05_ocr_enrichment" not in stages_to_run:
                stages_to_run.append("s05_ocr_enrichment")

            # Check if self._run_stage is the standard un-mocked method
            is_standard_run_stage = (
                getattr(self._run_stage, "__func__", self._run_stage) == Orchestrator._run_stage
            )
            use_unified_coordinator = (
                is_standard_run_stage
                and engine is not None
                and len(stages_to_run) > 1
                and bool(self.config.models.get("tier1"))
            )

            coordinator = None
            if use_unified_coordinator:
                from data_forge.inference.tier1 import Tier1Engine
                from data_forge.inference.ocr import OCREngine
                from data_forge.inference.vlm_batch import VLMUnifiedPassCoordinator

                tier1_inst = Tier1Engine(engine, self.config)
                ocr_inst = (
                    OCREngine(engine, self.config)
                    if "s05_ocr_enrichment" in stages_to_run
                    else None
                )
                coordinator = VLMUnifiedPassCoordinator(
                    self.config,
                    tier1_inst,
                    ocr_inst,
                )

            for i, record_ids in track(enumerate(chunks), total=len(chunks), description="Phase 4: Extract"):
                chunk_id = f"chunk_{i:04d}"
                active_ids = self._filter_active(record_ids)
                if not active_ids:
                    continue
                log.info("chunk_starting", chunk=chunk_id, phase="4_tier1_extract", records=len(active_ids))
                for idx in range(0, len(active_ids), 2048):
                    subchunk_ids = active_ids[idx : idx + 2048]
                    subchunk_tag = f"{chunk_id}_{idx}"

                    # If not using coordinator, run sequentially via _run_stage
                    if not use_unified_coordinator or coordinator is None:
                        for stage_name in stages_to_run:
                            subchunk_ids = self._filter_active(subchunk_ids)
                            if not subchunk_ids:
                                break
                            res = await self._run_stage(stage_name, subchunk_ids, subchunk_tag, engine)
                            chunk_results[i].stage_results.append(res)
                        continue

                    # Filter stages that still need execution (checkpoint-aware)
                    needed_stages = [
                        s for s in stages_to_run
                        if not (
                            self._resume
                            and self.config.checkpoint_enabled
                            and self._is_stage_complete(s, subchunk_tag, subchunk_ids)
                        )
                    ]

                    if not needed_stages:
                        for s in stages_to_run:
                            log.info("stage_skipped", stage=s, chunk=subchunk_tag, reason="checkpoint_exists")
                            chunk_results[i].stage_results.append(StageResult(stage_name=s))
                        continue

                    if len(needed_stages) == 1:
                        subchunk_ids = self._filter_active(subchunk_ids)
                        if subchunk_ids:
                            res = await self._run_stage(needed_stages[0], subchunk_ids, subchunk_tag, engine)
                            chunk_results[i].stage_results.append(res)
                        continue

                    # Unified pass for active records
                    subchunk_ids = self._filter_active(subchunk_ids)
                    if not subchunk_ids:
                        continue

                    raw_records = self.manifest.get_records_by_ids(subchunk_ids)
                    # Phase 4 recaption and structure only process safe records
                    safe_records = [r for r in raw_records if r.safety_tier == "safe"]
                    if not safe_records:
                        for s in needed_stages:
                            chunk_results[i].stage_results.append(StageResult(stage_name=s))
                        continue

                    run_rec = "s05_recaption" in needed_stages
                    run_struct = "s06_structure" in needed_stages
                    run_ocr_unified = "s05_ocr_enrichment" in needed_stages

                    start_t = time.monotonic()
                    contexts = await coordinator.process_subchunk(
                        safe_records,
                        run_recaption=run_rec,
                        run_structure=run_struct,
                        run_ocr=run_ocr_unified,
                    )
                    duration = time.monotonic() - start_t

                    updates = coordinator.to_manifest_updates(
                        contexts,
                        run_recaption=run_rec,
                        run_structure=run_struct,
                        run_ocr=run_ocr_unified,
                    )
                    if updates:
                        self.manifest.bulk_update_records(updates, stage="vlm_unified_phase4")

                    success_count = sum(1 for c in contexts if c.status not in ("excluded_failed", "excluded"))
                    failed_count = sum(1 for c in contexts if c.status == "excluded_failed")
                    excluded_count = sum(1 for c in contexts if c.status == "excluded")

                    for s in needed_stages:
                        res = StageResult(
                            stage_name=s,
                            records_processed=success_count,
                            records_failed=failed_count,
                            records_excluded=excluded_count,
                            duration_seconds=duration,
                            metadata={"unified_vlm_pass": True},
                        )
                        chunk_results[i].stage_results.append(res)

                        if self.config.fail_fast and not res.success:
                            log.error(
                                "stage_failed_fail_fast",
                                stage=s,
                                chunk=subchunk_tag,
                                failed=res.records_failed,
                            )
                            raise RuntimeError(
                                f"Stage {s} ({subchunk_tag}) failed with {res.records_failed} "
                                f"failed records (fail_fast=True)"
                            )

                        if self.config.checkpoint_enabled and res.success:
                            self._mark_stage_complete(s, subchunk_tag, subchunk_ids)

        async def _exec_phase5(engine: Any) -> None:
            if not run_ocr:
                return
            has_active = any(self._filter_active(chunk) for chunk in chunks)
            if not has_active:
                return
            for i, record_ids in track(enumerate(chunks), total=len(chunks), description="Phase 5: OCR"):
                chunk_id = f"chunk_{i:04d}"
                active_ids = self._filter_active(record_ids)
                if active_ids:
                    log.info("chunk_starting", chunk=chunk_id, phase="5_ocr", records=len(active_ids))
                    for idx in range(0, len(active_ids), 2048):
                        subchunk_ids = active_ids[idx : idx + 2048]
                        subchunk_ids = self._filter_active(subchunk_ids)
                        if not subchunk_ids:
                            continue
                        res = await self._run_stage("s05_ocr_enrichment", subchunk_ids, f"{chunk_id}_{idx}", engine)
                        chunk_results[i].stage_results.append(res)

        ocr_handled_in_tier1 = False
        if runnable_tier1:
            escalation_needed = False
            async with ModelEngine.vllm_session(self.config, "tier1") as engine:
                await _exec_phase2(engine)
                if self._should_run("s04_5_escalation", stages_filter):
                    escalation_needed = any(self._get_borderline_ids(chunk) for chunk in chunks)
                if not escalation_needed:
                    await _exec_phase4(engine)
                    if ocr_shares_tier1 and run_ocr and recaption_structure_stages:
                        ocr_handled_in_tier1 = True

            if escalation_needed:
                async with ModelEngine.vllm_session(self.config, "tier2") as engine:
                    await _exec_phase3_escalation(engine)
                if recaption_structure_stages:
                    async with ModelEngine.vllm_session(self.config, "tier1") as engine:
                        await _exec_phase4(engine)
                        if ocr_shares_tier1 and run_ocr:
                            ocr_handled_in_tier1 = True
        else:
            if self._should_run("s04_5_escalation", stages_filter):
                if any(self._get_borderline_ids(chunk) for chunk in chunks):
                    async with ModelEngine.vllm_session(self.config, "tier2") as engine:
                        await _exec_phase3_escalation(engine)
            if recaption_structure_stages:
                async with ModelEngine.vllm_session(self.config, "tier1") as engine:
                    await _exec_phase4(engine)
                    if ocr_shares_tier1 and run_ocr:
                        ocr_handled_in_tier1 = True

        if run_ocr and not ocr_handled_in_tier1:
            has_active = any(self._filter_active(chunk) for chunk in chunks)
            if has_active:
                model_to_use = "tier1" if ocr_shares_tier1 else "ocr"
                async with ModelEngine.vllm_session(self.config, model_to_use) as engine:
                    await _exec_phase5(engine)

        # ── Phase 5.5: Text PII Redaction ────────────────────────
        if self._should_run("s05_5_pii_text_redact", stages_filter):
            from rich.progress import track
            for i, record_ids in track(enumerate(chunks), total=len(chunks), description="Phase 5.5: Text PII"):
                chunk_id = f"chunk_{i:04d}"
                active_ids = self._filter_active(record_ids)
                if active_ids:
                    res = await self._run_stage("s05_5_pii_text_redact", active_ids, chunk_id)
                    chunk_results[i].stage_results.append(res)

        # ── Post-chunk global stages ────────────────────────────────
        all_record_ids = [rid for chunk in chunks for rid in chunk]
        if not all_record_ids:
            rows = self.manifest._conn.execute("SELECT id FROM records").fetchall()
            all_record_ids = [r["id"] for r in rows]

        # Route only after all source-ordered fetch chunks have been tagged.
        if self._should_run("s07_routing", stages_filter):
            await self._run_stage("s07_routing", all_record_ids, "global")

        # Keep the VAE resident over the entire corpus, rather than
        # reloading it for every processing chunk.
        if self._should_run("s08_encoding", stages_filter):
            routed_ids = (
                self.manifest.get_ids_by_status("routed")
                if hasattr(self.manifest, "get_ids_by_status")
                else [r["id"] for r in self.manifest._conn.execute("SELECT id FROM records WHERE status = 'routed'").fetchall()]
            )
            encoding_chunks = [
                routed_ids[i : i + self.config.chunk_size]
                for i in range(0, len(routed_ids), self.config.chunk_size)
            ]
            if encoding_chunks:
                from data_forge.inference.engine import ModelEngine
                from rich.progress import track
                async with ModelEngine.encoder_session(self.config) as engine:
                    for i, encoding_ids in track(enumerate(encoding_chunks), total=len(encoding_chunks), description="Phase 6: VAE"):
                        await self._run_stage(
                            "s08_encoding",
                            encoding_ids,
                            f"encode_chunk_{i:04d}",
                            engine,
                        )

        # DPO Latent Encoding — encodes the deduped/PII-scrubbed
        # preference pairs from s01_6 into Z-Image-Turbo's latent space
        # (the only fine-tuned renderer; Qwen-Image-Edit-2511 stays
        # frozen, so it never needs training latents at all). Runs once,
        # global, after the main manifest's s08_encoding phase — the two
        # are independent (preference pairs never entered the manifest)
        # but share the same encoder_session pattern.
        if self._should_run("s08_5_dpo_encoding", stages_filter):
            pref_root = (
                self.config.resolved_paths.get("preference_pairs")
                if self.config.resolved_paths
                else (self.config.data_root / "preference_pairs")
            )
            has_pairs = pref_root and pref_root.exists() and any(pref_root.glob("*/*.json"))
            if has_pairs:
                from data_forge.inference.engine import ModelEngine
                async with ModelEngine.encoder_session(self.config) as engine:
                    await self._run_stage(
                        "s08_5_dpo_encoding", record_ids=[], chunk_id="global", engine=engine
                    )
            else:
                log.info("s08_5_skipped_no_pairs", note="No preference pairs found on disk to process")

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
            else:
                await self._run_stage(
                    "s10_audit", [], "global", engine=None
                )

        # REMOVED: the Gemma 4 31B "Critic Tier" data-generation pass that
        # used to run here. It produced AI-judge-labeled preference data
        # (self-distillation, not calibration) — exactly the RLHF-loop
        # pattern the PRD's no-RLHF revision rules out for this project.
        # Gemma 4 now ships frozen as a product-side, on-demand critique
        # feature only; data-forge no longer trains it or consumes its
        # output as training data. Real human preference signal comes
        # from s01_6_preference_pairs/s08_5_dpo_encoding instead (Pick-a-
        # Pic v2, HPDv2, DesignSense-10k, DesignPref).

        # Final stage — no GPU model needed, pure filesystem/manifest
        # organization. Runs last deliberately: it reads training_pool,
        # encoding completeness, and critique_output, all of which need
        # every stage above (including the Critic Tier) to have already
        # run for the export to reflect the pipeline's actual final state.
        if self._should_run("s12_model_data_export", stages_filter):
            await self._run_stage(
                "s12_model_data_export", all_record_ids, "global"
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
        if hasattr(self.manifest, "filter_active_ids"):
            return self.manifest.filter_active_ids(record_ids)
        records = self.manifest.get_records_by_ids(record_ids)
        from data_forge.manifest import TERMINAL_STATUSES
        return [r.id for r in records if r.status not in TERMINAL_STATUSES]

    def _get_borderline_ids(self, record_ids: list[str]) -> list[str]:
        """Get records that need Tier-2 escalation (borderline safety or pending review)."""
        if not record_ids:
            return []
        if hasattr(self.manifest, "get_borderline_ids"):
            return self.manifest.get_borderline_ids(record_ids)
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


# ── Ordering consistency check ────────────────────────────────────────────
# `requires: ClassVar[tuple[str, ...]]` is declared on every Stage subclass,
# but nothing above ever reads it — execute_pipeline()'s actual order is a
# hand-maintained sequence of phases, entirely separate from that attribute.
# The two happened to agree everywhere except one place (the PII-scrub
# ordering bug fixed above), which is exactly the risk of two sources of
# truth for the same thing: they can silently drift, and only one of them
# is what actually runs. This doesn't make execution dynamically
# requires-driven (that's a bigger change than this fix), but it does turn
# the previously-decorative `requires` field into an enforced invariant:
# any future stage addition/reorder that violates its own declared
# dependency will fail loudly here instead of shipping silently.

EXECUTION_ORDER: tuple[str, ...] = (
    "s00_manifest_planning",
    "s01_fetch",
    "s01_5_uicrit_join",
    "s01_6_preference_pairs",
    "s01_7_preference_pair_pii",
    "s02_dedup",
    "s03_quality",
    "s03_5_pii_scrub",
    "s04_safety",
    "s04_5_escalation",       # Phase 3: must resolve borderline->safe before recaption runs
    "s05_recaption",          # Phase 4
    "s06_structure",
    "s05_ocr_enrichment",     # Phase 5
    "s05_5_pii_text_redact",
    "s07_routing",
    "s08_encoding",           # Phase 6
    "s08_5_dpo_encoding",
    "s09_heldout",
    "s10_audit",
    "s12_model_data_export",
)


#: Stages that are deliberately NOT part of EXECUTION_ORDER because they're
#: invoked through their own separate entry point rather than the per-record
#: pipeline flow. s11_registry_watcher runs via `data-forge registry check`
#: (a standalone, cron-triggered command — see cli.py's `registry_check`),
#: not from execute_pipeline(). Confirmed by tracing cli.py directly, not
#: assumed — the point of this allow-list existing at all is to make that
#: an explicit, checked fact instead of the validator just going quiet on it.
STANDALONE_ENTRY_POINT_STAGES: frozenset[str] = frozenset({"s11_registry_watcher"})


def validate_stage_ordering() -> list[str]:
    """Check every registered stage's declared `requires` against the
    orchestrator's actual hardcoded execution order (EXECUTION_ORDER above).

    Returns a list of human-readable violation messages — empty means
    consistent. Call this from `data-forge doctor` and from a test, not
    from the hot path (it's a startup/CI-time check, not a per-run one).
    """
    violations: list[str] = []
    if not _STAGE_REGISTRY:
        from data_forge.stages import register_all_stages

        register_all_stages()
    position = {name: i for i, name in enumerate(EXECUTION_ORDER)}

    for name, stage_cls in _STAGE_REGISTRY.items():
        if name in STANDALONE_ENTRY_POINT_STAGES:
            continue
        requires = getattr(stage_cls, "requires", ())
        if name not in position:
            violations.append(
                f"{name} is registered but not listed in EXECUTION_ORDER and "
                f"not in STANDALONE_ENTRY_POINT_STAGES — it will never "
                f"actually run from anywhere."
            )
            continue
        for dep in requires:
            if dep not in position:
                violations.append(
                    f"{name} declares requires=({dep!r}, ...) but {dep!r} "
                    f"is not in EXECUTION_ORDER at all."
                )
            elif position[dep] >= position[name]:
                violations.append(
                    f"{name} declares requires=({dep!r}, ...) but "
                    f"EXECUTION_ORDER actually runs {dep!r} at or after "
                    f"{name!r} (position {position[dep]} vs {position[name]})."
                )
    return violations
