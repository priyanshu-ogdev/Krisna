"""Stage 8: Encoding — encode to Z-Image latents, sketch-tier VQ tokens, and control maps.

(Formerly "Tri-Path" — a Qwen-Image-Edit-2511 latent branch was removed
here; that model now ships frozen. See the branch-removal comment below
and s08_5_dpo_encoding.py for where Z-Image-Turbo's DPO-specific encoding
lives instead.)
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import torch

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.image_utils import (
    image_to_tensor,
    load_image,
    normalize_for_vae,
    pad_to_multiple,
)

log = get_logger("stages.s08")


@register_stage("s08_encoding")
class EncodingStage(Stage):
    name = "s08_encoding"
    requires: ClassVar[tuple[str, ...]] = ("s07_routing",)

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status == "routed"]
        if not records or engine is None:
            return result

        # Storage check before encoding
        from data_forge.data.storage import StorageManager
        storage = StorageManager(config)
        mid_check = storage.mid_flight_check()
        if not mid_check["safe"]:
            log.error("storage_unsafe_for_encoding", **mid_check)
            result.metadata = {"storage_check": mid_check}
            return result

        paths = config.resolved_paths
        processed = failed = 0
        total_bytes = 0

        for rec in records:
            try:
                if not rec.scrubbed_image_path:
                    # Never fall back to the original, unscrubbed image —
                    # that would let un-blurred faces/PII text reach the
                    # training encodings with no signal it happened.
                    manifest.update_record(rec.id, "encoding", new_status="excluded_failed",
                                           reason="No scrubbed_image_path (PII scrub missing/failed)",
                                           exclusion_reason="pii_scrub_missing")
                    failed += 1
                    continue

                img_path = config.data_root / rec.scrubbed_image_path
                if not img_path.exists():
                    manifest.update_record(rec.id, "encoding", new_status="excluded_failed",
                                           reason="Image missing", exclusion_reason="image_missing")
                    failed += 1
                    continue

                try:
                    from PIL import UnidentifiedImageError
                    image = load_image(img_path)
                except UnidentifiedImageError:
                    manifest.update_record(rec.id, "encoding", new_status="excluded_failed",
                                           reason="Corrupt image", exclusion_reason="image_corrupt")
                    failed += 1
                    continue
                except Exception as e:
                    manifest.update_record(rec.id, "encoding", new_status="excluded_failed",
                                           reason=f"Image load error: {e}", exclusion_reason="image_error")
                    failed += 1
                    continue
                
                w, h = image.size
                if max(w, h) > 2048 or max(w, h) / max(min(w, h), 1) > 4.0:
                    manifest.update_record(rec.id, "encoding", new_status="excluded_low_quality",
                                           reason="Extreme aspect ratio or size", exclusion_reason="extreme_aspect_ratio")
                    failed += 1
                    continue

                image = pad_to_multiple(image, 16)

                encoding_paths: dict[str, str] = {}
                record_bytes = 0

                # Branch 1: Z-Image Continuous Latents
                try:
                    z_vae = engine.get_encoder("z_image_vae")
                    z_tensor = normalize_for_vae(image_to_tensor(image)).unsqueeze(0).to("cuda", dtype=torch.float16)
                    with torch.no_grad():
                        z_latent = z_vae.encode(z_tensor).latent_dist.sample()

                    z_path = paths["latents_zimage"] / f"{rec.id}.safetensors"
                    from safetensors.torch import save_file
                    save_file({"latent": z_latent.cpu()}, str(z_path))
                    encoding_paths["z_image_latent"] = str(z_path.relative_to(config.data_root))
                    record_bytes += z_path.stat().st_size
                except Exception as e:
                    log.warning("z_image_encode_failed", record_id=rec.id, error=str(e))

                # REMOVED: "Branch 2: Qwen-Image Continuous Latents". Data-
                # forge no longer needs to encode single-image latents for
                # Qwen-Image-Edit-2511 — it ships frozen (zero-shot ICL +
                # SDEdit at inference time), so nothing in this pipeline
                # trains it. See PRD "no-RLHF-loop" revision and
                # docs/DATA_COMPLETENESS.md. If it's ever un-frozen, restore
                # an encoder branch here rather than reusing dpo_latents/
                # (see s08_5_dpo_encoding.py's docstring for why those are
                # kept separate).

                # Branch 2 (was 3): Sketch Tier VQ Tokens
                # BUG FIX / COMPLETENESS GAP: this branch had no domain
                # check at all — it ran for every record regardless of
                # domain, including general_design (PD12M/CC12M-sourced)
                # images that will never be used to train the sketch tier,
                # which the PRD explicitly scopes as UI-domain-only. That
                # wasted a meaningful amount of GPU time and disk space
                # (VQ tokens for records with no possible use) at this
                # pipeline's target corpus scale. general_design records
                # legitimately end up with three artifacts instead of
                # four — that's correct, not a completeness gap; see
                # docs/DATA_COMPLETENESS.md for why the "encoding
                # complete" check below only requires vq_tokens for
                # ui_first records specifically.
                if rec.domain == "ui_first":
                    try:
                        vq_model = engine.get_encoder("maskgit_vq")
                        vq_tensor = image_to_tensor(image).unsqueeze(0).to("cuda", dtype=torch.float16)
                        with torch.no_grad():
                            vq_output = vq_model.encode(vq_tensor)
                            # Handle different VQ model output formats
                            if isinstance(vq_output, tuple):
                                vq_tokens = vq_output[0]
                            elif hasattr(vq_output, "encoding_indices"):
                                vq_tokens = vq_output.encoding_indices
                            else:
                                vq_tokens = vq_output

                        vq_path = paths["vq_tokens_sketch"] / f"{rec.id}.pt"
                        torch.save(vq_tokens.cpu(), str(vq_path))
                        encoding_paths["vq_tokens"] = str(vq_path.relative_to(config.data_root))
                        record_bytes += vq_path.stat().st_size
                    except Exception as e:
                        log.warning("vq_encode_failed", record_id=rec.id, error=str(e))

                # Branch 3 (was 4): Control Maps (Layout JSON from Stage 6 + Canny edges)
                try:
                    stage_cfg = config.get_stage("s08_encoding")
                    canny_low = stage_cfg.get("canny_low_threshold", 50)
                    canny_high = stage_cfg.get("canny_high_threshold", 150)

                    control_data: dict[str, Any] = {
                        "layout": rec.structure_output,
                        "record_id": rec.id,
                    }

                    edges_rel_path = None
                    try:
                        import cv2
                        import numpy as np

                        img_array = np.array(image.convert("L"))
                        edges = cv2.Canny(img_array, canny_low, canny_high)
                        edges_path = paths["control_tokens"] / f"{rec.id}_edges.png"
                        from PIL import Image as PILImage
                        PILImage.fromarray(edges).save(edges_path)
                        edges_rel_path = str(edges_path.relative_to(config.data_root))
                        record_bytes += edges_path.stat().st_size
                    except ImportError:
                        log.warning(
                            "canny_skipped_no_opencv",
                            record_id=rec.id,
                            note="opencv-python-headless not installed; control map will have no edges",
                        )
                    except Exception as e:
                        log.warning("canny_compute_failed", record_id=rec.id, error=str(e))

                    control_data["canny_edges_path"] = edges_rel_path
                    control_data["canny_thresholds"] = [canny_low, canny_high]

                    control_path = paths["control_tokens"] / f"{rec.id}.json"
                    control_path.write_text(
                        json.dumps(control_data, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    encoding_paths["control_map"] = str(control_path.relative_to(config.data_root))
                    if edges_rel_path:
                        encoding_paths["canny_edges"] = edges_rel_path
                    record_bytes += control_path.stat().st_size
                except Exception as e:
                    log.warning("control_map_failed", record_id=rec.id, error=str(e))

                if not encoding_paths:
                    # All four branches failed — this record has zero
                    # actual encoded artifacts. Do NOT mark it "encoded";
                    # that status is relied on by Stage 9 (heldout carve)
                    # and Stage 10 (audit) as proof training data exists.
                    manifest.update_record(rec.id, "encoding", new_status="excluded_failed",
                                           reason="All encoding branches failed — no artifacts produced",
                                           exclusion_reason="encoding_produced_nothing")
                    failed += 1
                    continue

                manifest.update_record(rec.id, "encoding", new_status="encoded",
                                       encoding_paths=encoding_paths)
                processed += 1
                total_bytes += record_bytes

            except Exception as e:
                log.error("encoding_failed", record_id=rec.id, error=str(e))
                manifest.update_record(rec.id, "encoding", new_status="excluded_failed",
                                       reason=f"Encoding error: {e}", exclusion_reason="encoding_error")
                failed += 1

        result.records_processed = processed
        result.records_failed = failed
        result.metadata = {"total_bytes_written": total_bytes}
        log.info("encoding_complete", processed=processed, failed=failed,
                 bytes_written=total_bytes, gb_written=round(total_bytes / 1e9, 3))
        return result
