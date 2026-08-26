"""Stage 2: Dedup — exact hash + FAISS semantic near-duplicate removal."""

from __future__ import annotations

from typing import Any

from data_forge.config import PipelineConfig
from data_forge.data.dedup import DedupEngine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s02")


@register_stage("s02_dedup")
class DedupStage(Stage):
    name = "s02_dedup"
    requires = ["s01_fetch"]

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s02_dedup")

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status == "fetched"]

        if not records:
            log.info("no_records_to_dedup")
            return result

        # Phase 1: Exact hash dedup
        exact_dupes = 0
        if stage_cfg.get("exact_hash_dedup", True):
            seen_hashes: dict[str, str] = {}
            for rec in records:
                if rec.content_hash_sha256:
                    existing = manifest.check_hash_exists(rec.content_hash_sha256)
                    if existing and existing != rec.id:
                        manifest.update_record(
                            record_id=rec.id,
                            stage="dedup",
                            new_status="excluded_duplicate",
                            reason=f"Exact hash match with {existing}",
                            duplicate_of=existing,
                            exclusion_reason="exact_hash_duplicate",
                        )
                        exact_dupes += 1

        log.info("exact_dedup_done", duplicates=exact_dupes)

        # Phase 2: Semantic near-duplicate via FAISS
        remaining = [r for r in records if r.status == "fetched"]
        if not remaining or engine is None:
            result.records_processed = len(records)
            result.records_excluded = exact_dupes
            return result

        # Generate CLIP embeddings
        image_paths = []
        valid_records = []
        for rec in remaining:
            if rec.image_path:
                img_path = config.data_root / rec.image_path
                if img_path.exists():
                    image_paths.append(img_path)
                    valid_records.append(rec)

        if not valid_records:
            result.records_processed = len(records)
            result.records_excluded = exact_dupes
            return result

        batch_size = stage_cfg.get("embedding_batch_size", 256)
        embeddings = DedupEngine.generate_embeddings(
            image_paths=image_paths,
            clip_model=engine.clip_model,
            clip_processor=engine.clip_processor,
            batch_size=batch_size,
        )

        # Build index and find duplicates
        threshold = stage_cfg.get("similarity_threshold", 0.95)
        dedup_engine = DedupEngine(similarity_threshold=threshold)
        record_ids_valid = [r.id for r in valid_records]
        dedup_engine.build_index(embeddings, record_ids_valid)
        duplicates = dedup_engine.find_duplicates(embeddings, record_ids_valid)

        # Mark duplicates (keep the first in each pair)
        marked: set[str] = set()
        semantic_dupes = 0
        for id_a, id_b, sim in duplicates:
            if id_b not in marked:
                manifest.update_record(
                    record_id=id_b,
                    stage="dedup",
                    new_status="excluded_duplicate",
                    reason=f"Semantic duplicate of {id_a} (sim={sim:.4f})",
                    duplicate_of=id_a,
                    exclusion_reason="semantic_duplicate",
                )
                marked.add(id_b)
                semantic_dupes += 1

        # Update surviving records
        for rec in valid_records:
            if rec.id not in marked:
                manifest.update_record(
                    record_id=rec.id, stage="dedup", new_status="deduped"
                )

        # Save index for reuse
        index_path = config.resolved_paths["manifests"] / "faiss_index.bin"
        dedup_engine.save_index(index_path)

        total_excluded = exact_dupes + semantic_dupes
        result.records_processed = len(records)
        result.records_excluded = total_excluded
        result.metadata = {
            "exact_duplicates": exact_dupes,
            "semantic_duplicates": semantic_dupes,
        }

        log.info(
            "dedup_complete",
            total=len(records),
            exact_dupes=exact_dupes,
            semantic_dupes=semantic_dupes,
            surviving=len(records) - total_excluded,
        )
        return result
