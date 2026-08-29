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
            await self._run_stage(
                "s01_5_uicrit_join",
                record_ids=[],
                chunk_id="global",
                engine=None,
            )

        # ── Stage 1.6: Planner Conversation Synthesis (runs once, global) ─
        # Needs Tier-1 for generation, unlike s01_5's pure parsing/matching
        # — separate vLLM session rather than reusing s01_fetch's, since
        # this must run AFTER s01_5's join has populated uicrit_human
        # critique_output records, which s01_fetch's own session has
        # already closed by that point.
        if self._should_run("s01_6_planner_synthesis", stages_filter):
            from data_forge.inference.engine import ModelEngine
            async with ModelEngine.vllm_session(self.config, "tier1") as engine:
                await self._run_stage(
                    "s01_6_planner_synthesis",
                    record_ids=[],
                    chunk_id="global",
                    engine=engine,
                )

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

                # BUG FIX: this block's own comment says PII scrub "needs to
                # happen between quality and safety," but the previous
                # implementation pulled it out and ran it unconditionally
                # BEFORE this phase's vLLM session even opened — i.e. before
                # s03_quality, not after it. Every PII-redacted record was
                # having its aesthetic/quality score computed against the
                # already-blurred image instead of the original, which is
                # exactly backwards (a blurred face region can skew a
                # sharpness/detail-based aesthetic score for reasons that
                # have nothing to do with the image's actual quality).
                #
                # Fixed by interleaving s03_5_pii_scrub inside the same open
                # vLLM session, between s03_quality and the rest — it doesn't
                # need the `engine` argument (MediaPipe, not vLLM), so this
                # costs nothing in extra model-swap overhead; the vLLM
                # process just sits idle for that one stage's duration.
                one_vlm_stage_order = [
                    s for s in ["s03_quality", "s03_5_pii_scrub", "s04_safety", "s05_recaption", "s06_structure"]
                    if s in runnable_tier1
                ]
                if one_vlm_stage_order:
                    async with ModelEngine.vllm_session(self.config, "tier1") as engine:
                        for stage_name in one_vlm_stage_order:
                            record_ids = self._filter_active(record_ids)
                            if not record_ids:
                                break
                            stage_engine = None if stage_name == "s03_5_pii_scrub" else engine
                            result = await self._run_stage(
                                stage_name, record_ids, chunk_id, stage_engine
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

                        # Text-PII redaction needs OCR output, so it runs
                        # right after OCR — not back in Stage 3.5, where
                        # rec.ocr_output was always empty. Deterministic
                        # (regex + PIL), no GPU model needed.
                        if self._should_run("s05_5_pii_text_redact", stages_filter):
                            record_ids = self._filter_active(record_ids)
                            if record_ids:
                                result = await self._run_stage(
                                    "s05_5_pii_text_redact", record_ids, chunk_id
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
            for stage_name in ["s07_routing", "s07_5_edit_pairs"]:
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

        # Critic Tier (Gemma 4 31B) — separate vLLM session, deliberately
        # not folded into the s10_audit session above: different model
        # family entirely (Gemma vs. Qwen), and keeping it a distinct
        # phase means it can be skipped independently (it's an additive,
        # non-blocking data source per the PRD — see s10_5_critic_preference.py's
        # module docstring) without touching the audit gate itself.
        if self._should_run("s10_5_critic_preference", stages_filter):
            training_ids = [r.id for r in self.manifest.get_training_pool()]
            audited_ids = [
                r.id for r in self.manifest.query_by_status("audited")
            ]
            critic_input_ids = list({*training_ids, *audited_ids})
            if critic_input_ids:
                from data_forge.inference.engine import ModelEngine
                async with ModelEngine.vllm_session(self.config, "critic") as engine:
                    await self._run_stage(
                        "s10_5_critic_preference", critic_input_ids, "global", engine
                    )

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
    "s01_6_planner_synthesis",
    "s02_dedup",
    "s03_quality",
    "s03_5_pii_scrub",
    "s04_safety",
    "s05_recaption",
    "s06_structure",
    "s05_ocr_enrichment",
    "s05_5_pii_text_redact",
    "s04_5_escalation",
    "s07_routing",
    "s07_5_edit_pairs",
    "s08_encoding",
    "s09_heldout",
    "s10_audit",
    "s10_5_critic_preference",
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
