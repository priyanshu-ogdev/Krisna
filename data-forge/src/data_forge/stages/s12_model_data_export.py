"""Stage 12: Model Data Export — segment the manifest into per-model training folders.

UPGRADED (no-RLHF-loop revision). The old version exported five folders,
one per PRD model (planner, sketch tier, Z-Image-Turbo, Qwen-Image-Edit-
2511, Gemma-4 critic). Three of those models now ship frozen — the
planner (RAG over UICrit instead of SFT), Qwen-Image-Edit-2511 (zero-shot
ICL + SDEdit instead of a fine-tune), and the critic (on-demand product
feature, not trained at all here). data-forge only trains two things now,
so this stage exports two training folders plus three reference/support
exports the frozen models and the paper's evaluation actually need:

    model_data/
      sketch_tier_maskgit/
        images/                    <- ui_first records only, complete encodings only
        captions.jsonl             <- {record_id, caption, image_filename}; image_filename
                                       is the actual join key for filename-keyed consumers —
                                       record_id is a random uuid4, unrelated to the linked
                                       image's on-disk name (see _export_sketch_tier)
        manifest_summary.json
      polish_zimage_turbo/
        latents/                   <- every domain, complete encodings only
                                       (no consumer yet — kept for a future
                                       custom loop; see _export_zimage)
        images/                    <- what the OFFICIAL diffusers script
                                       (train_dreambooth_lora_z_image.py)
                                       actually consumes
        captions.jsonl             <- same {record_id, caption, image_filename} shape as above
        manifest_summary.json
      polish_qwen_image21/
        conditions/                 <- deterministic wireframe images only
        targets/                    <- scrubbed UI screenshots only
        manifest.jsonl              <- strict condition/target pair manifest
        exclusions.jsonl            <- records rejected by the pair gate
        manifest_summary.json
      dpo_alignment/
        general/{pickapic_v2,hpdv2}/       <- Stage-1 DPO (broad aesthetic)
        domain/{designsense_10k,designpref}/  <- Stage-2 DPO (UI/design-specific)
        manifest_summary.json
      planner_rag_corpus/
        uicrit_critiques.jsonl     <- real human critique text, retrieved at
                                       inference time, not fine-tuned on
        manifest_summary.json
      eval_external/
        taste/, partiprompts/      <- linked read-only from
                                       heldout/external_eval/; never
                                       linked into any of the folders
                                       above, by construction
        manifest_summary.json

Files are linked, not copied (see utils/link_or_copy.py) — this is an
organizational view over the existing corpus, not a second copy of it.
Idempotent: re-running after new records finish processing only adds
what's new, since link_or_copy() skips anything already in place.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image, UnidentifiedImageError

from data_forge.config import PipelineConfig
from data_forge.data.domain_tagger import _GENERAL_DESIGN_SOURCES
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest, ManifestRecord
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.stages.split_identity import source_split_identity
from data_forge.utils.audit_gate import (
    eligible_training_records,
    pipeline_config_fingerprint,
    training_pool_fingerprint,
)
from data_forge.utils.completeness import is_encoding_complete
from data_forge.utils.hashing import sha256_file
from data_forge.utils.link_or_copy import link_or_copy
from data_forge.utils.path_safety import resolve_data_path

log = get_logger("stages.s12")

# Which preference-pair sources feed Stage-1 (general) vs. Stage-2
# (UI/design-domain) DPO — see configs/datasets.yaml's category field for
# the same split. Kept as an explicit list here (not inferred from
# category strings at runtime) so a new dataset added to datasets.yaml
# doesn't silently change which DPO stage it feeds without a deliberate
# edit to this file too.
_GENERAL_DPO_SOURCES = ("pickapic_v2", "hpdv2", "gamelabel_10k")
_DOMAIN_DPO_SOURCES = ()
_EVAL_ONLY_SOURCES = ("taste", "partiprompts")


@register_stage("s12_model_data_export")
class ModelDataExportStage(Stage):
    name = "s12_model_data_export"
    requires: ClassVar[tuple[str, ...]] = ("s09_heldout",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        audit = None
        if config.get_stage("s10_audit").enabled:
            audit_path = config.resolved_paths["audit_reports"] / "latest_audit.json"
            if not audit_path.is_file():
                raise RuntimeError("Refusing model-data export: no audit report exists")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not audit.get("pipeline_passes", False):
                raise RuntimeError(
                    "Refusing model-data export: audit threshold did not pass "
                    f"(pass_rate={audit.get('pass_rate')}, threshold={audit.get('threshold')})"
                )
        root = config.resolved_paths["model_data_root"]
        summary: dict[str, dict[str, Any]] = {}

        if audit is not None:
            training_pool = eligible_training_records(manifest, config.data_root)
            audit_config = config.get_stage("s10_audit")
            if audit.get("threshold") != audit_config.get("pass_rate_threshold", 0.95):
                raise RuntimeError("Refusing model-data export: audit threshold differs from the current config")
            if audit.get("sample_rate") != audit_config.get("sample_rate", 0.03):
                raise RuntimeError("Refusing model-data export: audit sample rate differs from the current config")
            if audit.get("min_samples") != audit_config.get("min_samples", 200):
                raise RuntimeError("Refusing model-data export: audit minimum sample count differs from the current config")
            if audit.get("pipeline_fingerprint") != pipeline_config_fingerprint(config):
                raise RuntimeError("Refusing model-data export: pipeline configuration or code changed after the audit")
            current_fingerprint = training_pool_fingerprint(training_pool, config.data_root)
            if audit.get("training_pool_fingerprint") != current_fingerprint:
                raise RuntimeError("Refusing model-data export: audited training data changed after the audit")
        else:
            training_pool = manifest.get_training_pool()

        critique_records = manifest.get_all_records_with_critique()
        loop = asyncio.get_running_loop()
        summary["sketch_tier_maskgit"] = await loop.run_in_executor(None, self._export_sketch_tier, training_pool, config, root)
        summary["polish_zimage_turbo"] = await loop.run_in_executor(None, self._export_zimage, training_pool, config, root)
        summary["polish_qwen_image21"] = await loop.run_in_executor(
            None, self._export_qwen_image21, training_pool, config, root
        )
        summary["dpo_alignment"] = await loop.run_in_executor(None, self._export_dpo_pairs, config, root)
        summary["planner_rag_corpus"] = await loop.run_in_executor(None, self._export_planner_rag, critique_records, config, root)
        summary["eval_external"] = await loop.run_in_executor(None, self._export_eval_external, config, root)

        (root / "EXPORT_SUMMARY.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        result.records_processed = sum(v.get("linked", 0) for v in summary.values())
        result.metadata = summary
        log.info("model_data_export_complete", summary=summary)
        return result

    # ── Per-model exporters ──────────────────────────────────────────────

    def _export_sketch_tier(self, manifest_or_pool: Manifest | list[ManifestRecord], config: PipelineConfig, root: Path) -> dict[str, int]:
        model_dir = root / "sketch_tier_maskgit"
        model_dir.mkdir(parents=True, exist_ok=True)  # see _export_zimage's identical note
        pool = manifest_or_pool.get_training_pool() if isinstance(manifest_or_pool, Manifest) else manifest_or_pool
        records = [
            r for r in pool
            if r.domain == "ui_first" and is_encoding_complete(r)
        ]
        images_linked = 0
        captions = []
        def _process_sketch(rec):
            image_filename = None
            linked = 0
            if rec.scrubbed_image_path:
                src = resolve_data_path(config.data_root, rec.scrubbed_image_path)
                if src.is_file():
                    link_or_copy(src, model_dir / "images" / src.name)
                    linked = 1
                    image_filename = src.name

            caption_data = {
                "record_id": rec.id, "caption": rec.caption,
                "image_filename": image_filename,
                "source_caption": rec.source_caption,
                "source_dataset": rec.source_dataset,
                "domain": rec.domain,
            }
            return linked, caption_data

        max_workers = min(64, max(4, multiprocessing.cpu_count() * 2))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for linked_count, caption_data in executor.map(_process_sketch, records):
                images_linked += linked_count
                captions.append(caption_data)

        unique_captions = len({r.caption for r in records if r.caption})
        diversity_ratio = (unique_captions / len(records)) if records else 1.0
        if len(records) >= 20 and diversity_ratio < 0.20:
            log.warning(
                "severe_caption_collapse_detected",
                target="sketch_tier_maskgit",
                unique_captions=unique_captions,
                total_records=len(records),
                diversity_ratio=round(diversity_ratio, 4),
                note="Over 80% of records share identical captions. Possible mock artifact leakage."
            )

        with (model_dir / "captions.jsonl").open("w", encoding="utf-8") as f:
            for c in captions:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        self._write_summary(model_dir, {
            "records": len(records),
            "images_linked": images_linked,
            "unique_captions": unique_captions,
            "diversity_ratio": round(diversity_ratio, 4),
        })
        return {"linked": images_linked, "records": len(records), "unique_captions": unique_captions}

    def _export_zimage(self, manifest_or_pool: Manifest | list[ManifestRecord], config: PipelineConfig, root: Path) -> dict[str, int]:
        model_dir = root / "polish_zimage_turbo"
        model_dir.mkdir(parents=True, exist_ok=True)
        pool = manifest_or_pool.get_training_pool() if isinstance(manifest_or_pool, Manifest) else manifest_or_pool
        records = [r for r in pool if is_encoding_complete(r)]
        linked = 0
        images_linked = 0
        captions = []
        def _process_zimage(rec):
            local_linked = 0
            local_images_linked = 0
            image_filename = None

            if "z_image_latent" in (rec.encoding_paths or {}):
                src = resolve_data_path(config.data_root, rec.encoding_paths["z_image_latent"])
                if src.is_file():
                    link_or_copy(src, model_dir / "latents" / src.name)
                    local_linked += 1

            if rec.scrubbed_image_path:
                src = resolve_data_path(config.data_root, rec.scrubbed_image_path)
                if src.is_file():
                    link_or_copy(src, model_dir / "images" / src.name)
                    local_images_linked += 1
                    image_filename = src.name

            caption_data = {
                "record_id": rec.id, "caption": rec.caption,
                "image_filename": image_filename,
                "source_caption": rec.source_caption,
                "source_dataset": rec.source_dataset,
                "domain": rec.domain,
            }
            return local_linked, local_images_linked, caption_data

        max_workers = min(64, max(4, multiprocessing.cpu_count() * 2))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for l_linked, i_linked, c_data in executor.map(_process_zimage, records):
                linked += l_linked
                images_linked += i_linked
                captions.append(c_data)

        unique_captions = len({r.caption for r in records if r.caption})
        diversity_ratio = (unique_captions / len(records)) if records else 1.0
        if len(records) >= 20 and diversity_ratio < 0.20:
            log.warning(
                "severe_caption_collapse_detected",
                target="polish_zimage_turbo",
                unique_captions=unique_captions,
                total_records=len(records),
                diversity_ratio=round(diversity_ratio, 4),
                note="Over 80% of records share identical captions. Possible mock artifact leakage."
            )

        with (model_dir / "captions.jsonl").open("w", encoding="utf-8") as f:
            for c in captions:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        self._write_summary(model_dir, {
            "records": len(records),
            "latents_linked": linked,
            "images_linked": images_linked,
            "unique_captions": unique_captions,
            "diversity_ratio": round(diversity_ratio, 4),
            "note": "DPO alignment on top of this base fine-tune uses dpo_alignment/, "
                    "not this folder — see _export_dpo_pairs(). images/ is what the "
                    "official train_dreambooth_lora_z_image.py script actually consumes; "
                    "latents/ is kept for a future custom loop that can use precomputed "
                    "latents directly, but has no consumer yet.",
        })
        return {"linked": linked, "records": len(records), "unique_captions": unique_captions}

    def _export_qwen_image21(
        self,
        manifest_or_pool: Manifest | list[ManifestRecord],
        config: PipelineConfig,
        root: Path,
    ) -> dict[str, Any]:
        """Export materialized pairs for the Qwen-Image-2.1 LoRA contract.

        A screenshot and its structure annotations are not an image-conditioned
        pair by themselves.  The condition must be an explicitly materialized
        image produced by a deterministic wireframe renderer.  ``s08`` currently
        produces a JSON control map (and sometimes Canny edges), so neither is
        accepted here as a condition image.  A future renderer can opt in by
        storing its output under ``encoding_paths["condition_image"]``.

        This deliberate boundary is important: using the target screenshot or
        its Canny edges as the condition would create identity-copy training
        examples while making general visual records look like UI pairs.
        """
        model_dir = root / "polish_qwen_image21"
        conditions_dir = model_dir / "conditions"
        targets_dir = model_dir / "targets"
        model_dir.mkdir(parents=True, exist_ok=True)
        conditions_dir.mkdir(parents=True, exist_ok=True)
        targets_dir.mkdir(parents=True, exist_ok=True)

        pool = (
            manifest_or_pool.get_training_pool()
            if isinstance(manifest_or_pool, Manifest)
            else manifest_or_pool
        )
        records = sorted(pool, key=lambda record: record.id)
        entries: list[dict[str, Any]] = []
        exclusions: list[dict[str, str]] = []

        # Stage 9's source-disjoint mode is the only accepted split receipt
        # for an active Qwen training export.  Status alone is not enough: a
        # caller can construct a training_pool list by hand, bypassing the
        # held-out carve entirely.
        split_stage = config.get_stage("s09_heldout")
        source_disjoint = split_stage.enabled and bool(split_stage.get("source_disjoint", False))
        source_splits: dict[str, set[str]] = {}
        for rec in records:
            if rec.status in ("training_pool", "audited"):
                split = "train"
            elif rec.status == "heldout":
                split = "validation"
            else:
                continue
            source_splits.setdefault(source_split_identity(rec), set()).add(split)
        split_conflicts = {
            source for source, splits in source_splits.items() if len(splits) > 1
        }

        for rec in records:
            if not source_disjoint:
                exclusions.append({
                    "record_id": rec.id,
                    "reason": "source_disjoint_split_not_configured",
                })
                continue
            if source_split_identity(rec) in split_conflicts:
                exclusions.append({
                    "record_id": rec.id,
                    "reason": "source_disjoint_split_violation",
                })
                continue
            reason = self._qwen_pair_exclusion_reason(rec, config)
            if reason is not None:
                exclusions.append({"record_id": rec.id, "reason": reason})
                continue

            target = resolve_data_path(config.data_root, rec.scrubbed_image_path or "")
            condition_rel = (rec.encoding_paths or {}).get("condition_image")
            # _qwen_pair_exclusion_reason verifies this is present, but keep the
            # guard local so this method cannot ever fabricate a pair if its
            # eligibility predicate changes later.
            if not condition_rel:
                exclusions.append({"record_id": rec.id, "reason": "condition_renderer_unavailable"})
                continue
            condition = resolve_data_path(config.data_root, condition_rel)

            target_name = f"{rec.id}{target.suffix.lower() or '.png'}"
            condition_name = f"{rec.id}{condition.suffix.lower() or '.png'}"
            target_dest = targets_dir / target_name
            condition_dest = conditions_dir / condition_name
            try:
                link_or_copy(target, target_dest)
                link_or_copy(condition, condition_dest)
                target_sha = sha256_file(target)
                condition_sha = sha256_file(condition)
            except (OSError, ValueError) as exc:
                exclusions.append({
                    "record_id": rec.id,
                    "reason": f"pair_materialization_failed:{type(exc).__name__}",
                })
                continue

            provenance = self._qwen_source_provenance(rec, config)
            caption = (rec.caption or "").strip()
            entry = {
                "record_id": rec.id,
                "source_record_id": rec.source_record_id,
                "source_group_id": rec.source_group_id,
                "condition_image": str(condition_dest.relative_to(model_dir)).replace("\\", "/"),
                "target_image": str(target_dest.relative_to(model_dir)).replace("\\", "/"),
                # The trainer's active column is named ``image``; retain the
                # descriptive ``target_image`` field for the audit manifest.
                "image": str(target_dest.relative_to(model_dir)).replace("\\", "/"),
                # Keep both names: ``caption`` is the training column while
                # ``prompt`` makes the instruction role explicit to consumers.
                "caption": caption,
                "prompt": caption,
                # Stage 9's status is the persisted split receipt.  Do not
                # infer a split from record order or randomly re-split here.
                "split": "validation" if rec.status == "heldout" else "train",
                "source_dataset": rec.source_dataset,
                "source_revision": (
                    provenance.get("revision")
                    or provenance.get("source_revision")
                    or provenance.get("commit")
                ),
                "source_provenance": provenance,
                "condition_sha256": condition_sha,
                "target_sha256": target_sha,
                "content_hashes": {
                    "condition_image": condition_sha,
                    "target_image": target_sha,
                },
            }
            entries.append(entry)

        manifest_path = model_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

        exclusions_path = model_dir / "exclusions.jsonl"
        with exclusions_path.open("w", encoding="utf-8") as handle:
            for exclusion in exclusions:
                handle.write(json.dumps(exclusion, ensure_ascii=False, sort_keys=True) + "\n")

        reason_counts: dict[str, int] = {}
        for exclusion in exclusions:
            reason = exclusion["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        renderer_available = any(
            (rec.encoding_paths or {}).get("condition_image") for rec in records
        )
        summary = {
            "records_considered": len(records),
            "pairs_exported": len(entries),
            "conditions_linked": len(entries),
            "targets_linked": len(entries),
            "excluded_records": len(exclusions),
            "exclusions_by_reason": reason_counts,
            "source_disjoint": source_disjoint,
            "condition_renderer": {
                "available": renderer_available,
                "artifact_key": "encoding_paths.condition_image",
                "required": "deterministic_wireframe_image",
            },
            "note": (
                "Only UI records with policy-approved metadata, a valid scrubbed "
                "target, schema-valid non-freeform structure_output, and an "
                "explicit deterministic condition_image artifact are exported. "
                "The JSON control_map and Canny edges are not condition images. "
                "Records without a safe renderer output are listed in exclusions.jsonl."
                " The training-side manifest preparer rejects unassigned splits;"
                " source-identity splitting is a required pre-training step."
            ),
        }
        self._write_summary(model_dir, summary)
        return {"linked": len(entries), "records": len(records), **summary}

    @staticmethod
    def _qwen_pair_exclusion_reason(
        rec: ManifestRecord, config: PipelineConfig
    ) -> str | None:
        """Return a stable exclusion reason for a Qwen pair candidate."""
        if rec.status not in ("training_pool", "audited"):
            return "record_not_in_training_pool"
        if rec.domain != "ui_first" or rec.source_dataset in _GENERAL_DESIGN_SOURCES:
            return "not_explicit_ui_source"
        source_spec = getattr(config, "datasets", {}).get(rec.source_dataset)
        if source_spec is not None and source_spec.category in {
            "general_visual",
            "dpo_preference_general",
            "dpo_preference_domain",
        }:
            return "not_explicit_ui_source"
        if source_spec is None:
            return "source_not_registered"
        if source_spec.eval_only or source_spec.annotation_only:
            return "source_not_training_images"
        if not source_spec.training_eligible:
            return "source_not_training_approved"
        if "polish_qwen_image21" not in source_spec.training_targets:
            return "source_not_approved_for_qwen"
        if not source_spec.provenance_url:
            return "source_provenance_missing"
        if source_spec.policy_violations(config.source_policy):
            return "source_policy_failed"
        # Registry pins alone do not prove which revision produced this record.
        if (rec.license_output or {}).get("source_revision") != source_spec.immutable_revision():
            return "source_revision_receipt_mismatch"
        if rec.license_verified is not True:
            return "source_license_not_approved"
        if rec.pii_scrubbed is not True:
            return "target_not_pii_scrubbed"
        if rec.safety_tier != "safe":
            return "source_safety_not_approved"
        if not (rec.caption or "").strip():
            return "caption_missing"
        if not rec.scrubbed_image_path:
            return "scrubbed_target_missing"
        try:
            target = resolve_data_path(config.data_root, rec.scrubbed_image_path)
            with Image.open(target) as image:
                image.verify()
            if not target.is_file():
                return "scrubbed_target_missing"
        except (OSError, UnidentifiedImageError, ValueError):
            return "scrubbed_target_invalid"

        structure = rec.structure_output
        if not isinstance(structure, dict):
            return "structure_output_missing"
        # A valid empty/freeform structure is intentionally the representation
        # used for general visual records.  It cannot deterministically produce
        # a UI wireframe condition.
        if structure.get("layout_type") == "freeform" or not structure.get("elements"):
            return "structure_not_suitable_for_condition"
        try:
            from data_forge.data.schema_validator import SchemaValidator

            valid, _errors = SchemaValidator(config.schemas_dir).validate_structure(structure)
        except (OSError, ValueError, FileNotFoundError):
            return "structure_schema_unavailable"
        if not valid:
            return "structure_output_invalid"

        condition_rel = (rec.encoding_paths or {}).get("condition_image")
        if not condition_rel:
            return "condition_renderer_unavailable"
        try:
            condition = resolve_data_path(config.data_root, condition_rel)
            with Image.open(condition) as image:
                image.verify()
            if not condition.is_file():
                return "condition_image_missing"
            if (
                condition.resolve() == target.resolve()
                or sha256_file(condition) == sha256_file(target)
            ):
                return "condition_equals_target"
        except (OSError, UnidentifiedImageError, ValueError):
            return "condition_image_invalid"
        return None

    @staticmethod
    def _qwen_source_provenance(
        rec: ManifestRecord, config: PipelineConfig
    ) -> dict[str, Any]:
        """Return non-sensitive source revision/license provenance."""
        provenance: dict[str, Any] = {"source_file": rec.source_file}
        spec = getattr(config, "datasets", {}).get(rec.source_dataset)
        if spec is not None:
            for field in (
                "source_type", "repo_id", "revision", "subset",
                "license_url", "license_status", "provenance_url",
                "synthetic_status", "quality_tier",
            ):
                value = getattr(spec, field, None)
                if value is not None:
                    provenance[field] = value
            provenance["revision"] = spec.immutable_revision()
        return provenance

    def _export_dpo_pairs(self, config: PipelineConfig, root: Path) -> dict[str, int]:
        """Export Diffusion-DPO training data — Z-Image-Turbo's only
        alignment signal, since it's the only fine-tuned renderer.
        Deliberately kept as two separate subtrees (general/, domain/)
        rather than merged: the PRD's ablation (c) needs to run DPO on
        general-only vs. general+domain-augmented and compare against
        TASTE, which requires being able to select just one subtree at
        training time.
        """
        model_dir = root / "dpo_alignment"
        dpo_root = config.resolved_paths["dpo_latents"]
        counts: dict[str, int] = {}
        max_workers = min(64, max(4, multiprocessing.cpu_count() * 2))

        def _make_processor(bucket_name, source_key):
            def _proc(f):
                link_or_copy(f, model_dir / bucket_name / source_key / f.name)
                meta_f = f.with_suffix("").with_suffix(".meta.json")
                if meta_f.exists():
                    link_or_copy(meta_f, model_dir / bucket_name / source_key / meta_f.name)
                return 1
            return _proc

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for bucket_name, sources in (("general", _GENERAL_DPO_SOURCES), ("domain", _DOMAIN_DPO_SOURCES)):
                bucket_total = 0
                for source_key in sources:
                    src_dir = dpo_root / source_key
                    if not src_dir.exists():
                        counts[f"{bucket_name}/{source_key}"] = 0
                        continue
                    files = list(src_dir.glob("*.safetensors"))
                    if files:
                        proc = _make_processor(bucket_name, source_key)
                        n = sum(executor.map(proc, files))
                    else:
                        n = 0
                    counts[f"{bucket_name}/{source_key}"] = n
                    bucket_total += n
                counts[f"{bucket_name}_total"] = bucket_total

        self._write_summary(model_dir, {
            **counts,
            "note": "general/ = Stage-1 DPO (Pick-a-Pic v2, HPDv2, GameLabel-10K — broad aesthetic, "
                    "not UI-specific). domain/ = Stage-2 DPO (DesignSense-10k, "
                    "DesignPref — real human-designer UI/layout preference; may be "
                    "empty until those two datasets' repo_id is confirmed, see "
                    "configs/datasets.yaml). Train Stage-1 first, then Stage-2 as a "
                    "small-batch pass on top — do not merge the two buckets into one "
                    "training run, since the PRD's ablation (c) depends on being able "
                    "to compare general-only vs. domain-augmented against TASTE.",
            # A2 NOTE (sync audit): this folder links precomputed Z-Image latents
            # from s08_5_dpo_encoding.py (configs/pipeline.yaml s08_5_dpo_encoding:
            # enabled: false by default). It will be EMPTY on every normal production
            # run — this is expected, not a pipeline failure. The real DPO training
            # data path is:
            #   data-forge preference_pairs/ (s01_6_preference_pairs)
            #   → training/data_forge_bridge/sync_dpo_pairs.py
            #   → krisna_training.dpo.preference_store.PreferenceStore
            #   → training/polish/dpo_dataset.py::PreferencePairDataset
            #   → training/polish/train_dpo.py
            # train_dpo.py re-encodes images from BlobStore at training time;
            # it does NOT consume this folder's .safetensors latents. Enable
            # s08_5_dpo_encoding only if wiring a fast-path into train_dpo.py
            # to skip live VAE re-encoding (see s08_5_dpo_encoding.py's docstring).
            "TRAINING_PATH_NOTE": "real DPO training path: preference_pairs/ → "
                                  "sync_dpo_pairs → PreferenceStore → train_dpo.py. "
                                  "This folder is a precomputed-latents cache, disabled "
                                  "by default (s08_5_dpo_encoding: enabled: false).",
        })
        return {"linked": counts.get("general_total", 0) + counts.get("domain_total", 0)}

    def _export_planner_rag(self, records: list[ManifestRecord], config: PipelineConfig, root) -> dict[str, int]:
        """Export UICrit's real critique text for the product's RAG index.

        Replaces the old planner SFT export (linked planner_data/
        conversations/, produced by the now-removed
        s01_6_planner_synthesis.py). The planner ships frozen now — this
        is retrieval-corpus material, consumed at inference time by the
        product's own RAG index, not a training set data-forge feeds into
        a fine-tune. Kept in data-forge (not hand-exported ad hoc) so the
        corpus goes through the same license/PII/audit discipline as
        everything else in this pipeline before the product consumes it.
        """
        model_dir = root / "planner_rag_corpus"
        entries = []
        for rec in records:
            if not rec.critique_output:
                continue
            if (
                rec.status not in ("training_pool", "audited")
                or rec.license_verified is not True
                or rec.pii_scrubbed is not True
                or rec.safety_tier != "safe"
            ):
                continue
            if rec.critique_output.get("critique_source") != "uicrit_human":
                continue  # real human critique only — no AI-judge text belongs in a RAG corpus either
            entries.append({
                "record_id": rec.id,
                "image_path": rec.scrubbed_image_path or rec.image_path,
                "critique_output": rec.critique_output,
            })

        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "uicrit_critiques.jsonl").open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._write_summary(model_dir, {
            "critique_records": len(entries),
            "note": "Real UICrit human critique text only. Consumed by the product's "
                    "RAG index at inference time — the planner (Qwen3.5-9B) is frozen "
                    "and is never fine-tuned on this or anything else data-forge produces.",
        })
        return {"linked": len(entries), "records": len(entries)}

    def _export_eval_external(self, config: PipelineConfig, root) -> dict[str, int]:
        """Link TASTE/PartiPrompts read-only into model_data/ for
        convenience, from heldout/external_eval/ — never from
        training_pool/, preference_pairs/, or dpo_alignment/, so there is
        no path by which this export could pull in anything but the
        eval-only snapshot fetcher.py::_fetch_eval_reference already
        isolated. Purely a convenience mirror, not a second source of
        truth — if it's missing, the run simply hasn't fetched it yet.
        """
        model_dir = root / "eval_external"
        eval_root = config.resolved_paths["heldout"] / "external_eval"
        linked = 0
        found_sources = []
        for source_key in _EVAL_ONLY_SOURCES:
            src_dir = eval_root / source_key
            if not src_dir.exists():
                continue
            found_sources.append(source_key)
            for f in src_dir.rglob("*"):
                if f.is_file():
                    link_or_copy(f, model_dir / source_key / f.relative_to(src_dir))
                    linked += 1

        self._write_summary(model_dir, {
            "sources_found": found_sources,
            "files_linked": linked,
            "note": "Evaluation-only. Never used as training input by any exporter "
                    "above — see fetcher.py::_fetch_eval_reference and "
                    "configs/datasets.yaml's eval_only: true flags.",
        })
        return {"linked": linked}

    @staticmethod
    def _write_summary(model_dir, data: dict[str, Any]) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "manifest_summary.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
