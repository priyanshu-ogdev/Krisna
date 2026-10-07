"""Stage 2: Dedup — exact hash + FAISS semantic near-duplicate removal."""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.data.dedup import DedupEngine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.hashing import perceptual_hash, sha256_file

log = get_logger("stages.s02")


@register_stage("s02_dedup")
class DedupStage(Stage):
    name = "s02_dedup"
    requires: ClassVar[tuple[str, ...]] = ("s01_fetch",)

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

        # ── Phase 1: High-Performance Exact Hash Dedup ──────────────────
        exact_dupes = 0
        exact_duplicate_updates: list[dict[str, Any]] = []

        if stage_cfg.get("exact_hash_dedup", True):
            # 1. Fill missing content_hash_sha256 and perceptual_hash if file exists
            hashes_to_query: list[str] = []
            computed_hashes: list[tuple[str, str, str | None]] = []
            
            def _compute_hash(rec):
                sha = rec.content_hash_sha256
                phash = rec.perceptual_hash
                err = None
                if (not sha or not phash) and rec.image_path:
                    p = Path(rec.image_path)
                    img_path = p if p.is_absolute() else (config.data_root / rec.image_path)
                    if img_path.is_file():
                        try:
                            if not sha:
                                sha = sha256_file(img_path)
                            if not phash:
                                phash = perceptual_hash(img_path)
                        except Exception as e:
                            err = e
                return rec, sha, phash, err

            max_workers = min(64, max(4, (os.cpu_count() or 4) * 2))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for rec, computed_sha, computed_phash, err in executor.map(_compute_hash, records):
                    if computed_sha:
                        rec.content_hash_sha256 = computed_sha
                        rec.perceptual_hash = computed_phash
                        computed_hashes.append((rec.id, computed_sha, computed_phash))
                    elif err:
                        log.warning("hash_compute_failed", id=rec.id, error=str(err))
                    
                    if rec.content_hash_sha256:
                        hashes_to_query.append(rec.content_hash_sha256)

            if computed_hashes:
                manifest.bulk_update_hashes(computed_hashes)

            # 2. Batch query manifest for existing non-duplicate records
            existing_hash_map = manifest.check_hashes_exist(hashes_to_query)

            # 3. Detect collisions against manifest AND intra-batch
            seen_chunk_hashes: dict[str, str] = {}
            for rec in records:
                if not rec.content_hash_sha256:
                    continue

                h = rec.content_hash_sha256
                canonical_id: str | None = None

                if h in existing_hash_map and existing_hash_map[h] != rec.id:
                    canonical_id = existing_hash_map[h]
                elif h in seen_chunk_hashes and seen_chunk_hashes[h] != rec.id:
                    canonical_id = seen_chunk_hashes[h]
                else:
                    seen_chunk_hashes[h] = rec.id

                if canonical_id:
                    exact_duplicate_updates.append({
                        "record_id": rec.id,
                        "duplicate_of": canonical_id,
                        "reason": f"Exact SHA-256 match with {canonical_id}",
                        "exclusion_reason": "exact_hash_duplicate",
                    })
                    exact_dupes += 1

            if exact_duplicate_updates:
                manifest.bulk_mark_duplicates(exact_duplicate_updates, stage="dedup")

        log.info("exact_dedup_done", duplicates=exact_dupes)

        # Exclude exact duplicates from downstream dedup
        exact_excluded_ids = {u["record_id"] for u in exact_duplicate_updates}
        remaining = [r for r in records if r.id not in exact_excluded_ids]

        # ── Phase 1b: Image Path Existence Validation ──────────────────
        missing_ids: list[str] = []
        valid_records = []
        image_paths = []

        for rec in remaining:
            if not rec.image_path:
                missing_ids.append(rec.id)
                continue
            p = Path(rec.image_path)
            img_path = p if p.is_absolute() else (config.data_root / rec.image_path)
            if not img_path.is_file():
                missing_ids.append(rec.id)
                continue
            image_paths.append(img_path)
            valid_records.append(rec)

        if missing_ids:
            manifest.bulk_mark_failed(
                missing_ids,
                stage="dedup",
                reason="Image file missing or path not found on disk",
            )
            log.warning("missing_images_excluded", count=len(missing_ids))

        # Check if semantic dedup should proceed
        semantic_enabled = stage_cfg.get("semantic_dedup", True)
        if not semantic_enabled or engine is None or not valid_records:
            # Advance remaining valid records to "deduped" so downstream stages proceed
            surviving_ids = [r.id for r in valid_records]
            if surviving_ids:
                manifest.bulk_update_status(
                    surviving_ids,
                    new_status="deduped",
                    stage="dedup",
                    reason="Passed exact hash dedup (semantic dedup skipped/no engine)",
                )

            total_excluded = exact_dupes + len(missing_ids)
            result.records_processed = len(records)
            result.records_excluded = total_excluded
            result.metadata = {
                "exact_duplicates": exact_dupes,
                "missing_files": len(missing_ids),
                "semantic_duplicates": 0,
            }
            return result

        # ── Phase 2: Semantic Dedup via CLIP Embeddings + FAISS ────────
        batch_size = stage_cfg.get("embedding_batch_size", 256)
        embeddings, valid_indices, failed_indices = DedupEngine.generate_embeddings(
            image_paths=image_paths,
            clip_model=engine.clip_model,
            clip_processor=engine.clip_processor,
            batch_size=batch_size,
        )

        # Mark corrupt/unreadable images as failed
        if failed_indices:
            corrupt_ids = [valid_records[i].id for i in failed_indices]
            manifest.bulk_mark_failed(
                corrupt_ids,
                stage="dedup",
                reason="Corrupt or unreadable image file (PIL decode failure)",
            )
            log.warning("corrupt_images_excluded", count=len(corrupt_ids))

        candidate_records = [valid_records[i] for i in valid_indices]
        candidate_ids = [r.id for r in candidate_records]

        if not candidate_ids:
            total_excluded = exact_dupes + len(missing_ids) + len(failed_indices)
            result.records_processed = len(records)
            result.records_excluded = total_excluded
            result.metadata = {
                "exact_duplicates": exact_dupes,
                "missing_files": len(missing_ids),
                "corrupt_files": len(failed_indices),
                "semantic_duplicates": 0,
                "surviving_records": 0,
            }
            return result

        threshold = stage_cfg.get("similarity_threshold", 0.95)
        nprobe = stage_cfg.get("faiss_nprobe", 64)
        index_type = stage_cfg.get("faiss_index_type", "Flat")

        dedup_engine = DedupEngine(
            similarity_threshold=threshold,
            index_type=index_type,
            nprobe=nprobe,
        )

        # Step 2a: Intra-batch deduplication
        intra_dupes, intra_survivor_indices = dedup_engine.dedup_batch(
            embeddings, candidate_ids, threshold=threshold
        )
        intra_survivor_embeddings = embeddings[intra_survivor_indices]
        intra_survivor_ids = [candidate_ids[i] for i in intra_survivor_indices]

        # Step 2b: Inter-batch deduplication against persistent index
        manifests_dir = (
            config.resolved_paths.get("manifests")
            or config.paths.resolve(config.data_root).get("manifests")
            or (config.data_root / "manifests")
        )
        index_path = manifests_dir / "faiss_index.bin"
        if index_path.exists():
            dedup_engine.load_index(index_path)

        inter_dupes, true_survivor_indices = dedup_engine.search_existing(
            intra_survivor_embeddings, intra_survivor_ids, threshold=threshold
        )
        true_survivor_embeddings = intra_survivor_embeddings[true_survivor_indices]
        true_survivor_ids = [intra_survivor_ids[i] for i in true_survivor_indices]

        # Step 2c: Add ONLY true survivors to persistent index and save
        if len(true_survivor_ids) > 0:
            dedup_engine.add_records(true_survivor_embeddings, true_survivor_ids)
            dedup_engine.save_index(index_path)

        # Step 2d: Record all semantic duplicates in manifest
        semantic_duplicate_updates: list[dict[str, Any]] = []
        for dup_id, canonical_id, sim in intra_dupes:
            semantic_duplicate_updates.append({
                "record_id": dup_id,
                "duplicate_of": canonical_id,
                "reason": f"Intra-chunk semantic duplicate of {canonical_id} (sim={sim:.4f})",
                "exclusion_reason": "semantic_duplicate",
            })

        for dup_id, canonical_id, sim in inter_dupes:
            semantic_duplicate_updates.append({
                "record_id": dup_id,
                "duplicate_of": canonical_id,
                "reason": f"Semantic duplicate of historical {canonical_id} (sim={sim:.4f})",
                "exclusion_reason": "semantic_duplicate",
            })

        if semantic_duplicate_updates:
            manifest.bulk_mark_duplicates(semantic_duplicate_updates, stage="dedup")

        # Step 2e: Transition true survivors to "deduped"
        if true_survivor_ids:
            manifest.bulk_update_status(
                true_survivor_ids,
                new_status="deduped",
                stage="dedup",
                reason="Passed exact and semantic dedup",
            )

        dedup_engine.cleanup()

        total_semantic = len(semantic_duplicate_updates)
        total_corrupt = len(failed_indices)
        total_missing = len(missing_ids)
        total_excluded = exact_dupes + total_semantic + total_corrupt + total_missing

        result.records_processed = len(records)
        result.records_excluded = total_excluded
        result.metadata = {
            "exact_duplicates": exact_dupes,
            "semantic_duplicates": total_semantic,
            "intra_batch_duplicates": len(intra_dupes),
            "historical_duplicates": len(inter_dupes),
            "missing_files": total_missing,
            "corrupt_files": total_corrupt,
            "surviving_records": len(true_survivor_ids),
        }

        log.info(
            "dedup_complete",
            total=len(records),
            exact_dupes=exact_dupes,
            semantic_dupes=total_semantic,
            missing=total_missing,
            corrupt=total_corrupt,
            surviving=len(true_survivor_ids),
        )
        return result
