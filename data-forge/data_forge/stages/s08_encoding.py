"""Stage 8: Tri-Path Encoding — encode to Z-Image, Qwen-Image, VQ, and control maps."""

from __future__ import annotations

import json
from typing import Any

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
    requires = ["s07_routing"]

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
                img_path = config.data_root / (rec.scrubbed_image_path or rec.image_path or "")
                if not img_path.exists():
                    manifest.update_record(rec.id, "encoding", new_status="excluded_failed",
                                           reason="Image missing", exclusion_reason="image_missing")
                    failed += 1; continue

                image = load_image(img_path)
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

                # Branch 2: Qwen-Image Continuous Latents
                try:
                    q_vae = engine.get_encoder("qwen_image_vae")
                    q_tensor = normalize_for_vae(image_to_tensor(image)).unsqueeze(0).to("cuda", dtype=torch.float16)
                    with torch.no_grad():
                        q_latent = q_vae.encode(q_tensor).latent_dist.sample()

                    q_path = paths["latents_qwenimage"] / f"{rec.id}.safetensors"
                    save_file({"latent": q_latent.cpu()}, str(q_path))
                    encoding_paths["qwen_image_latent"] = str(q_path.relative_to(config.data_root))
                    record_bytes += q_path.stat().st_size
                except Exception as e:
                    log.warning("qwen_image_encode_failed", record_id=rec.id, error=str(e))

                # Branch 3: Sketch Tier VQ Tokens
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

                # Branch 4: Control Maps (Layout JSON from Stage 6)
                try:
                    control_path = paths["control_tokens"] / f"{rec.id}.json"
                    control_data = {
                        "layout": rec.structure_output,
                        "record_id": rec.id,
                    }
                    control_path.write_text(
                        json.dumps(control_data, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    encoding_paths["control_map"] = str(control_path.relative_to(config.data_root))
                    record_bytes += control_path.stat().st_size
                except Exception as e:
                    log.warning("control_map_failed", record_id=rec.id, error=str(e))

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
