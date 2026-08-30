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
        vq_tokens/                 <- ui_first records only, complete encodings only
        images/                    <- matching scrubbed images
        captions.jsonl
        manifest_summary.json
      polish_zimage_turbo/
        latents/                   <- every domain, complete encodings only
        captions.jsonl
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

import json
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.completeness import is_encoding_complete
from data_forge.utils.link_or_copy import link_or_copy

log = get_logger("stages.s12")

# Which preference-pair sources feed Stage-1 (general) vs. Stage-2
# (UI/design-domain) DPO — see configs/datasets.yaml's category field for
# the same split. Kept as an explicit list here (not inferred from
# category strings at runtime) so a new dataset added to datasets.yaml
# doesn't silently change which DPO stage it feeds without a deliberate
# edit to this file too.
_GENERAL_DPO_SOURCES = ("pickapic_v2", "hpdv2")
_DOMAIN_DPO_SOURCES = ("designsense_10k", "designpref")
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
        root = config.resolved_paths["model_data_root"]
        summary: dict[str, dict[str, int]] = {}

        summary["sketch_tier_maskgit"] = self._export_sketch_tier(manifest, config, root)
        summary["polish_zimage_turbo"] = self._export_zimage(manifest, config, root)
        summary["dpo_alignment"] = self._export_dpo_pairs(config, root)
        summary["planner_rag_corpus"] = self._export_planner_rag(manifest, config, root)
        summary["eval_external"] = self._export_eval_external(config, root)

        (root / "EXPORT_SUMMARY.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        result.records_processed = sum(v.get("linked", 0) for v in summary.values())
        result.metadata = summary
        log.info("model_data_export_complete", summary=summary)
        return result

    # ── Per-model exporters ──────────────────────────────────────────────

    def _export_sketch_tier(self, manifest: Manifest, config: PipelineConfig, root) -> dict[str, int]:
        model_dir = root / "sketch_tier_maskgit"
        records = [
            r for r in manifest.get_training_pool()
            if r.domain == "ui_first" and is_encoding_complete(r)
        ]
        linked = 0
        captions = []
        for rec in records:
            if "vq_tokens" in (rec.encoding_paths or {}):
                src = config.data_root / rec.encoding_paths["vq_tokens"]
                if src.exists():
                    link_or_copy(src, model_dir / "vq_tokens" / src.name)
                    linked += 1
            if rec.scrubbed_image_path:
                src = config.data_root / rec.scrubbed_image_path
                if src.exists():
                    link_or_copy(src, model_dir / "images" / src.name)
            captions.append({"record_id": rec.id, "caption": rec.caption})

        (model_dir / "captions.jsonl").write_text(
            "\n".join(json.dumps(c) for c in captions), encoding="utf-8"
        )
        self._write_summary(model_dir, {"records": len(records), "vq_tokens_linked": linked})
        return {"linked": linked, "records": len(records)}

    def _export_zimage(self, manifest: Manifest, config: PipelineConfig, root) -> dict[str, int]:
        model_dir = root / "polish_zimage_turbo"
        records = [r for r in manifest.get_training_pool() if is_encoding_complete(r)]
        linked = 0
        captions = []
        for rec in records:
            if "z_image_latent" in (rec.encoding_paths or {}):
                src = config.data_root / rec.encoding_paths["z_image_latent"]
                if src.exists():
                    link_or_copy(src, model_dir / "latents" / src.name)
                    linked += 1
            captions.append({"record_id": rec.id, "caption": rec.caption})

        (model_dir / "captions.jsonl").write_text(
            "\n".join(json.dumps(c) for c in captions), encoding="utf-8"
        )
        self._write_summary(model_dir, {
            "records": len(records),
            "latents_linked": linked,
            "note": "DPO alignment on top of this base fine-tune uses dpo_alignment/, "
                    "not this folder — see _export_dpo_pairs().",
        })
        return {"linked": linked, "records": len(records)}

    def _export_dpo_pairs(self, config: PipelineConfig, root) -> dict[str, int]:
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

        for bucket_name, sources in (("general", _GENERAL_DPO_SOURCES), ("domain", _DOMAIN_DPO_SOURCES)):
            bucket_total = 0
            for source_key in sources:
                src_dir = dpo_root / source_key
                if not src_dir.exists():
                    counts[f"{bucket_name}/{source_key}"] = 0
                    continue
                n = 0
                for f in src_dir.glob("*.safetensors"):
                    link_or_copy(f, model_dir / bucket_name / source_key / f.name)
                    meta_f = f.with_suffix("").with_suffix(".meta.json")
                    if meta_f.exists():
                        link_or_copy(meta_f, model_dir / bucket_name / source_key / meta_f.name)
                    n += 1
                counts[f"{bucket_name}/{source_key}"] = n
                bucket_total += n
            counts[f"{bucket_name}_total"] = bucket_total

        self._write_summary(model_dir, {
            **counts,
            "note": "general/ = Stage-1 DPO (Pick-a-Pic v2, HPDv2 — broad aesthetic, "
                    "not UI-specific). domain/ = Stage-2 DPO (DesignSense-10k, "
                    "DesignPref — real human-designer UI/layout preference; may be "
                    "empty until those two datasets' repo_id is confirmed, see "
                    "configs/datasets.yaml). Train Stage-1 first, then Stage-2 as a "
                    "small-batch pass on top — do not merge the two buckets into one "
                    "training run, since the PRD's ablation (c) depends on being able "
                    "to compare general-only vs. domain-augmented against TASTE.",
        })
        return {"linked": counts.get("general_total", 0) + counts.get("domain_total", 0)}

    def _export_planner_rag(self, manifest: Manifest, config: PipelineConfig, root) -> dict[str, int]:
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
        records = manifest.get_all_records_with_critique()
        entries = []
        for rec in records:
            if not rec.critique_output:
                continue
            if rec.critique_output.get("critique_source") != "uicrit_human":
                continue  # real human critique only — no AI-judge text belongs in a RAG corpus either
            entries.append({
                "record_id": rec.id,
                "image_path": rec.scrubbed_image_path or rec.image_path,
                "critique_output": rec.critique_output,
            })

        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "uicrit_critiques.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries), encoding="utf-8"
        )
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
