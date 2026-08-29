"""Stage 7.5: Edit Pairs — paired data for Qwen-Image-Edit-2511's actual task.

Closes the gap traced in the data-completeness audit: Stage 8 only ever
encoded single finished images into `latents_qwenimage/`, which is fine
as general backbone/renderer signal but doesn't match what
Qwen-Image-Edit-2511 is actually meant to do at inference time — treat a
rough Stage-1 sketch as an edit target and refine it into a polished
render (see the PRD's Stage 2 design). Training that specific behavior
needs (source, instruction, target) triples, not single images.

Two sources, both landing in processed/edit_pairs/:
1. External, general-domain grounding (MagicBrush, InstructPix2Pix — see
   datasets.yaml's magicbrush/instructpix2pix entries and
   fetcher.py::_fetch_huggingface_triples). Teaches the edit-conditioning
   task SHAPE. Already written directly by the fetcher; this stage does
   not touch those.
2. UI-domain pairs, synthetically constructed HERE: take a curated
   `ui_first` image already past Stage 6 (structured), generate a
   degraded version (blur + downsample/upsample) as a stand-in for what
   Stage 1's sketch tier would hand off, pair it with the original as the
   target, and template a generic refinement instruction. This is the
   only way to get UI-domain pairs at all — none exist publicly — and it
   mirrors the real Stage-1-to-Stage-2 handoff shape more closely than
   the general-domain pairs alone.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.image_utils import load_image

log = get_logger("stages.s07_5")

_INSTRUCTION_TEMPLATES = [
    "Refine this rough mockup into a polished, high-fidelity screen.",
    "Sharpen the layout and add final detail to this draft UI.",
    "Turn this sketch into a finished, production-quality design.",
    "Polish this rough draft — improve typography, spacing, and visual detail.",
]


@register_stage("s07_5_edit_pairs")
class EditPairsStage(Stage):
    name = "s07_5_edit_pairs"
    requires: ClassVar[tuple[str, ...]] = ("s07_routing",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        records = manifest.get_records_by_ids(record_ids)
        records = [
            r for r in records
            if r.domain == "ui_first" and r.status == "routed" and r.scrubbed_image_path
        ]
        if not records:
            return result

        stage_cfg = config.get_stage("s07_5_edit_pairs")
        blur_radius = stage_cfg.get("degrade_blur_radius", 4)
        downsample_factor = stage_cfg.get("degrade_downsample_factor", 4)

        out_dir = config.resolved_paths["edit_pairs"] / "synthetic_ui"
        out_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        failed = 0
        import hashlib

        for idx, rec in enumerate(records):
            try:
                img_path = config.data_root / rec.scrubbed_image_path
                if not img_path.exists():
                    failed += 1
                    continue
                image = load_image(img_path)
                degraded = self._degrade(image, blur_radius, downsample_factor)

                pair_id = f"synth_{rec.id}"
                degraded.save(out_dir / f"{pair_id}_source.png", "PNG")
                image.save(out_dir / f"{pair_id}_target.png", "PNG")

                instruction = _INSTRUCTION_TEMPLATES[
                    int(hashlib.sha256(rec.id.encode()).hexdigest(), 16) % len(_INSTRUCTION_TEMPLATES)
                ]
                (out_dir / f"{pair_id}.json").write_text(
                    json.dumps({
                        "pair_id": pair_id,
                        "record_id": rec.id,
                        "instruction": instruction,
                        "source": f"{pair_id}_source.png",
                        "target": f"{pair_id}_target.png",
                        "origin": "synthetic_ui_degradation",
                    }),
                    encoding="utf-8",
                )
                written += 1
            except Exception as e:
                log.warning("edit_pair_generation_failed", record_id=rec.id, error=str(e))
                failed += 1

        result.records_processed = written
        result.records_failed = failed
        log.info("edit_pairs_complete", written=written, failed=failed)
        return result

    @staticmethod
    def _degrade(image, blur_radius: int, downsample_factor: int):
        """Produce a plausible stand-in for a rough Stage-1 sketch output.

        Blur (loses fine detail/texture, similar to what a coarse
        masked-generative sketch pass would produce) + a downsample/
        upsample round-trip (loses resolution, mirrors the sketch tier's
        typically-lower native resolution before Stage 2 upscales it).
        Not a claim that this perfectly matches the real sketch tier's
        output distribution — that model doesn't have a trained checkpoint
        yet (the PRD's own single biggest open risk) — just the closest
        cheap approximation available without one.
        """
        from PIL import ImageFilter

        w, h = image.size
        degraded = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        small = degraded.resize((max(1, w // downsample_factor), max(1, h // downsample_factor)))
        return small.resize((w, h))
