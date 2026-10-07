"""Stage 8: Encoding — encode to Z-Image latents, sketch-tier VQ tokens, and control maps.

(Formerly "Tri-Path" — a Qwen-Image-Edit-2511 latent branch was removed
here; that model now ships frozen. See the branch-removal comment below
and s08_5_dpo_encoding.py for where Z-Image-Turbo's DPO-specific encoding
lives instead.)
"""

from collections import defaultdict
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image as PILImage, UnidentifiedImageError
from safetensors.torch import save_file
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
    resize_for_model,
)
from data_forge.utils.path_safety import resolve_data_path

try:
    import cv2
    import numpy as np
    _HAS_OPENCV = True
except ImportError:
    cv2 = None
    np = None
    _HAS_OPENCV = False

log = get_logger("stages.s08")


@register_stage("s08_encoding")
class EncodingStage(Stage):
    name = "s08_encoding"
    requires: ClassVar[tuple[str, ...]] = ("s07_routing",)

    async def run(self, manifest: Manifest, config: PipelineConfig,
                  record_ids: list[str], engine: Any | None = None) -> StageResult:
        result = StageResult(stage_name=self.name)

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status in ("routed", "encoded")]
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

        paths = config.paths.resolve(config.data_root) if config.paths else config.resolved_paths
        paths["latents_zimage"].mkdir(parents=True, exist_ok=True)
        paths["control_tokens"].mkdir(parents=True, exist_ok=True)
        processed = failed = 0
        total_bytes = 0

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(config.data_root))
            except ValueError:
                return str(p)

        stage_cfg = config.get_stage("s08_encoding")
        canny_low = stage_cfg.get("canny_low_threshold", 50)
        canny_high = stage_cfg.get("canny_high_threshold", 150)
        max_enc_size = stage_cfg.get("max_encoding_resolution", 1024)

        # Pre-load to avoid race conditions
        try:
            z_vae = engine.get_encoder("z_image_vae")
            enc_dev = "cuda" if torch.cuda.is_available() else "cpu"
            enc_dtype = torch.float16 if enc_dev == "cuda" else torch.float32
        except Exception as e:
            log.error("vae_load_failed", error=str(e))
            return result

        vae_batch_size = stage_cfg.get("vae_batch_size", 16 if enc_dev == "cuda" else 4)

        def _preprocess_record(rec) -> dict:
            try:
                z_path = paths["latents_zimage"] / f"{rec.id}.safetensors"
                control_path = paths["control_tokens"] / f"{rec.id}.json"
                edges_path = paths["control_tokens"] / f"{rec.id}_edges.png"

                # Instant cache hit: skip image loading/resizing completely if already on disk
                if (z_path.exists() and z_path.stat().st_size > 0 and
                    control_path.exists() and control_path.stat().st_size > 0):
                    encoding_paths: dict[str, str] = {
                        "z_image_latent": _rel(z_path),
                        "control_map": _rel(control_path),
                    }
                    record_bytes = z_path.stat().st_size + control_path.stat().st_size
                    if edges_path.exists() and edges_path.stat().st_size > 0:
                        encoding_paths["canny_edges"] = _rel(edges_path)
                        record_bytes += edges_path.stat().st_size
                    return {
                        "type": "done",
                        "id": rec.id,
                        "status": "encoded",
                        "encoding_paths": encoding_paths,
                        "bytes": record_bytes,
                    }

                if not rec.scrubbed_image_path:
                    return {
                        "type": "done",
                        "id": rec.id,
                        "status": "excluded_failed",
                        "reason": "No scrubbed_image_path (PII scrub missing/failed)",
                        "exclusion_reason": "pii_scrub_missing",
                    }

                img_path = resolve_data_path(config.data_root, rec.scrubbed_image_path)
                if not img_path.is_file():
                    return {
                        "type": "done",
                        "id": rec.id,
                        "status": "excluded_failed",
                        "reason": "Image missing",
                        "exclusion_reason": "image_missing",
                    }

                try:
                    image = load_image(img_path)
                except UnidentifiedImageError:
                    return {
                        "type": "done",
                        "id": rec.id,
                        "status": "excluded_failed",
                        "reason": "Corrupt image",
                        "exclusion_reason": "image_corrupt",
                    }
                except Exception as e:
                    return {
                        "type": "done",
                        "id": rec.id,
                        "status": "excluded_failed",
                        "reason": f"Image load error: {e}",
                        "exclusion_reason": "image_error",
                    }

                w, h = image.size
                if max(w, h) / max(min(w, h), 1) > 4.0:
                    return {
                        "type": "done",
                        "id": rec.id,
                        "status": "excluded_low_quality",
                        "reason": "Extreme aspect ratio",
                        "exclusion_reason": "extreme_aspect_ratio",
                    }

                image = resize_for_model(image, max_size=max_enc_size)
                image = pad_to_multiple(image, 16)

                encoding_paths = {}
                record_bytes = 0

                # Branch 2: Control Maps (Canny edges + JSON tokens on CPU)
                if control_path.exists() and control_path.stat().st_size > 0:
                    encoding_paths["control_map"] = _rel(control_path)
                    if edges_path.exists() and edges_path.stat().st_size > 0:
                        encoding_paths["canny_edges"] = _rel(edges_path)
                        record_bytes += edges_path.stat().st_size
                    record_bytes += control_path.stat().st_size
                else:
                    try:
                        control_data: dict[str, Any] = {
                            "layout": rec.structure_output,
                            "record_id": rec.id,
                        }

                        edges_rel_path = None
                        if _HAS_OPENCV:
                            try:
                                img_array = np.array(image.convert("L"))
                                edges = cv2.Canny(img_array, canny_low, canny_high)
                                PILImage.fromarray(edges).save(edges_path)
                                edges_rel_path = _rel(edges_path)
                                record_bytes += edges_path.stat().st_size
                            except Exception as e:
                                log.warning("canny_compute_failed", record_id=rec.id, error=str(e))
                        else:
                            log.warning("canny_skipped_no_opencv", record_id=rec.id, note="opencv-python-headless not installed")

                        control_data["canny_edges_path"] = edges_rel_path
                        control_data["canny_thresholds"] = [canny_low, canny_high]

                        control_path.write_text(json.dumps(control_data, ensure_ascii=False), encoding="utf-8")
                        encoding_paths["control_map"] = _rel(control_path)
                        if edges_rel_path:
                            encoding_paths["canny_edges"] = edges_rel_path
                        record_bytes += control_path.stat().st_size
                    except Exception as e:
                        log.warning("control_map_failed", record_id=rec.id, error=str(e))

                # Branch 1: Z-Image Continuous Latents
                if z_path.exists() and z_path.stat().st_size > 0:
                    encoding_paths["z_image_latent"] = _rel(z_path)
                    record_bytes += z_path.stat().st_size
                    return {
                        "type": "done",
                        "id": rec.id,
                        "status": "encoded",
                        "encoding_paths": encoding_paths,
                        "bytes": record_bytes,
                    }

                # Needs batched VAE forward pass
                tensor = normalize_for_vae(image_to_tensor(image))
                return {
                    "type": "needs_vae",
                    "id": rec.id,
                    "tensor": tensor,
                    "resolution": (image.width, image.height),
                    "z_path": z_path,
                    "encoding_paths": encoding_paths,
                    "record_bytes": record_bytes,
                }

            except Exception as e:
                log.error("encoding_failed", record_id=rec.id, error=str(e))
                return {
                    "type": "done",
                    "id": rec.id,
                    "status": "excluded_failed",
                    "reason": f"Encoding error: {e}",
                    "exclusion_reason": "encoding_error",
                }

        updates: list[dict[str, Any]] = []
        window_size = 256
        max_workers = min(32, max(4, multiprocessing.cpu_count()))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for w_start in range(0, len(records), window_size):
                window_records = records[w_start : w_start + window_size]

                # 1. Parallel CPU preprocessing (IO, decoding, Canny, JSON)
                prep_results = list(executor.map(_preprocess_record, window_records))

                # 2. Group items needing VAE by resolution for batched GPU forward pass
                needs_vae_by_res: dict[tuple[int, int], list[dict]] = defaultdict(list)
                for res in prep_results:
                    if res["type"] == "done":
                        item: dict[str, Any] = {
                            "id": res["id"],
                            "new_status": res["status"],
                        }
                        if res["status"] == "encoded":
                            item["encoding_paths"] = res["encoding_paths"]
                            processed += 1
                            total_bytes += res["bytes"]
                        else:
                            item["reason"] = res.get("reason")
                            item["exclusion_reason"] = res.get("exclusion_reason")
                            failed += 1
                        updates.append(item)
                    else:
                        needs_vae_by_res[res["resolution"]].append(res)

                # 3. High-throughput batched VAE encoding per resolution bucket
                for _res_key, res_items in needs_vae_by_res.items():
                    for b_start in range(0, len(res_items), vae_batch_size):
                        batch = res_items[b_start : b_start + vae_batch_size]
                        batch_tensor = torch.stack([item["tensor"] for item in batch]).to(enc_dev, dtype=enc_dtype)
                        try:
                            with torch.inference_mode():
                                batch_latents = z_vae.encode(batch_tensor).latent_dist.sample().cpu()
                            for item, latent in zip(batch, batch_latents):
                                save_file({"latent": latent.unsqueeze(0)}, str(item["z_path"]))
                                item["encoding_paths"]["z_image_latent"] = _rel(item["z_path"])
                                item["record_bytes"] += item["z_path"].stat().st_size
                        except Exception as batch_err:
                            # Resilient fallback: if batched encode fails (e.g. OOM), encode one-by-one
                            if enc_dev == "cuda" and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            for item in batch:
                                try:
                                    single_tensor = item["tensor"].unsqueeze(0).to(enc_dev, dtype=enc_dtype)
                                    with torch.inference_mode():
                                        single_latent = z_vae.encode(single_tensor).latent_dist.sample().cpu()
                                    save_file({"latent": single_latent}, str(item["z_path"]))
                                    item["encoding_paths"]["z_image_latent"] = _rel(item["z_path"])
                                    item["record_bytes"] += item["z_path"].stat().st_size
                                except Exception as e:
                                    log.warning("z_image_encode_failed", record_id=item["id"], error=str(e))

                        for item in batch:
                            rec_item: dict[str, Any] = {"id": item["id"]}
                            if not item["encoding_paths"]:
                                rec_item["new_status"] = "excluded_failed"
                                rec_item["reason"] = "All encoding branches failed — no artifacts produced"
                                rec_item["exclusion_reason"] = "encoding_produced_nothing"
                                failed += 1
                            else:
                                rec_item["new_status"] = "encoded"
                                rec_item["encoding_paths"] = item["encoding_paths"]
                                processed += 1
                                total_bytes += item["record_bytes"]
                            updates.append(rec_item)

                if len(updates) >= 200:
                    manifest.bulk_update_records(updates, stage="encoding")
                    updates.clear()

        if updates:
            manifest.bulk_update_records(updates, stage="encoding")

        result.records_processed = processed
        result.records_failed = failed
        result.metadata = {"total_bytes_written": total_bytes}
        log.info("encoding_complete", processed=processed, failed=failed,
                 bytes_written=total_bytes, gb_written=round(total_bytes / 1e9, 3))
        return result
