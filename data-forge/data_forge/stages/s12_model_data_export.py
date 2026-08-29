"""Stage 12: Model Data Export — segment the manifest into per-model training folders.

Everything upstream of this stage organizes data by *pipeline concern*
(raw/, processed/latents_zimage/, ui_critique/, ...). Nothing organizes it
by *which of the PRD's five trainable components actually consumes it* —
which is exactly the question the data-completeness audit had to answer
by tracing code manually. This stage builds that answer once, as a real
directory tree, so it doesn't have to be re-derived by reading code every
time:

    model_data/
      planner_qwen3.5_9b/
        conversations/            <- linked from planner_data/conversations/
        manifest_summary.json
      sketch_tier_maskgit/
        vq_tokens/                <- ui_first records only, complete encodings only
        images/                   <- matching scrubbed images
        captions.jsonl
        manifest_summary.json
      polish_zimage_turbo/
        latents/                  <- every domain, complete encodings only
        captions.jsonl
        manifest_summary.json
      polish_qwenimage_edit_2511/
        latents_backbone/         <- single-image latents (general renderer signal)
        edit_pairs/               <- linked from processed/edit_pairs/ (the model's
                                      actual edit-task training data, if built — see
                                      s07_5_edit_pairs.py)
        manifest_summary.json
      critic_gemma4_31b/
        human_calibration/        <- critique_source == "uicrit_human" (the REAL
                                      calibration signal — use this for QLoRA)
        self_generated/           <- critique_source == "gemma4_31b" (kept, but
                                      labeled — self-distillation, not calibration)
        manifest_summary.json

Files are linked, not copied (see utils/link_or_copy.py) — this is an
organizational view over the existing corpus, not a second copy of it.
Idempotent: re-running after new records finish processing only adds what's
new, since link_or_copy() skips anything already in place.
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

        summary["planner_qwen3.5_9b"] = self._export_planner(config, root)
        summary["sketch_tier_maskgit"] = self._export_sketch_tier(manifest, config, root)
        summary["polish_zimage_turbo"] = self._export_zimage(manifest, config, root)
        summary["polish_qwenimage_edit_2511"] = self._export_qwenimage_edit(manifest, config, root)
        summary["critic_gemma4_31b"] = self._export_critic(manifest, config, root)

        (root / "EXPORT_SUMMARY.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        result.records_processed = sum(v.get("linked", 0) for v in summary.values())
        result.metadata = summary
        log.info("model_data_export_complete", summary=summary)
        return result

    # ── Per-model exporters ──────────────────────────────────────────────

    def _export_planner(self, config: PipelineConfig, root) -> dict[str, int]:
        model_dir = root / "planner_qwen3.5_9b"
        src = config.resolved_paths["planner_conversations"]
        linked = 0
        if src.exists():
            for f in src.glob("*.jsonl"):
                strategy = link_or_copy(f, model_dir / "conversations" / f.name)
                linked += 1 if strategy != "exists" else 0
        count = sum(1 for _ in (model_dir / "conversations").glob("*.jsonl")) if (model_dir / "conversations").exists() else 0
        self._write_summary(model_dir, {"conversation_shards": count, "status": "ready" if count else "empty — see s01_6_planner_synthesis.py"})
        return {"linked": linked, "total_shards": count}

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
        self._write_summary(model_dir, {"records": len(records), "latents_linked": linked})
        return {"linked": linked, "records": len(records)}

    def _export_qwenimage_edit(self, manifest: Manifest, config: PipelineConfig, root) -> dict[str, int]:
        model_dir = root / "polish_qwenimage_edit_2511"
        records = [r for r in manifest.get_training_pool() if is_encoding_complete(r)]
        linked = 0
        for rec in records:
            if "qwen_image_latent" in (rec.encoding_paths or {}):
                src = config.data_root / rec.encoding_paths["qwen_image_latent"]
                if src.exists():
                    link_or_copy(src, model_dir / "latents_backbone" / src.name)
                    linked += 1

        edit_pairs_src = config.resolved_paths["edit_pairs"]
        pair_count = 0
        if edit_pairs_src.exists():
            for f in edit_pairs_src.rglob("*"):
                if f.is_file():
                    link_or_copy(f, model_dir / "edit_pairs" / f.relative_to(edit_pairs_src))
                    pair_count += 1

        self._write_summary(model_dir, {
            "backbone_latents_linked": linked,
            "edit_pair_files": pair_count,
            "note": "edit_pairs is this model's actual training task (edit-conditioned "
                    "refinement); latents_backbone is general-renderer signal only — "
                    "see s07_5_edit_pairs.py's docstring if edit_pair_files is 0.",
        })
        return {"linked": linked, "edit_pair_files": pair_count}

    def _export_critic(self, manifest: Manifest, config: PipelineConfig, root) -> dict[str, int]:
        model_dir = root / "critic_gemma4_31b"
        all_records = manifest.get_training_pool()
        human = []
        self_gen = []
        for rec in all_records:
            if not rec.critique_output:
                continue
            source = rec.critique_output.get("critique_source")
            entry = {
                "record_id": rec.id,
                "image_path": rec.scrubbed_image_path or rec.image_path,
                "critique_output": rec.critique_output,
            }
            if source == "uicrit_human":
                human.append(entry)
            elif source == "gemma4_31b":
                self_gen.append(entry)

        (model_dir / "human_calibration.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (model_dir / "human_calibration.jsonl").write_text(
            "\n".join(json.dumps(e) for e in human), encoding="utf-8"
        )
        (model_dir / "self_generated.jsonl").write_text(
            "\n".join(json.dumps(e) for e in self_gen), encoding="utf-8"
        )
        self._write_summary(model_dir, {
            "human_calibration_records": len(human),
            "self_generated_records": len(self_gen),
            "note": "QLoRA fine-tuning should use human_calibration.jsonl as the real "
                    "supervised signal. self_generated.jsonl is Gemma 4's own output "
                    "(s10_5_critic_preference.py) — useful for DPO preference-pair "
                    "seeding once the product's own candidate-sampling loop exists, "
                    "not for calibrating the critic against itself.",
        })
        return {"human_calibration": len(human), "self_generated": len(self_gen)}

    @staticmethod
    def _write_summary(model_dir, data: dict[str, Any]) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "manifest_summary.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
