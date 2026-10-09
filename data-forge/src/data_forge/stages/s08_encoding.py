"""Stage 8: Encoding — produce Z-Image latents and control maps.

(Formerly "Tri-Path" — a Qwen-Image-Edit-2511 latent branch was removed
here; that model now ships frozen. See the branch-removal comment below
and s08_5_dpo_encoding.py for where Z-Image-Turbo's DPO-specific encoding
lives instead. CPU preprocessing feeds a bounded queue while the VAE
encodes prior batches, keeping memory bounded without serializing CPU/GPU work.)
"""

import asyncio
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

import torch
from PIL import Image as PILImage, UnidentifiedImageError
from safetensors import safe_open
from safetensors.torch import save_file

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest, ManifestRecord
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
        started_at = time.monotonic()
        result = StageResult(stage_name=self.name)

        records = manifest.get_records_by_ids(record_ids)
        records = [r for r in records if r.status in ("routed", "encoded")]
        if not records:
            return result
        if engine is None:
            raise RuntimeError("s08_encoding requires an active encoder session")

        # Storage check before encoding
        from data_forge.data.storage import StorageManager
        storage = StorageManager(config)
        mid_check = storage.mid_flight_check()
        if not mid_check["safe"]:
            log.error("storage_unsafe_for_encoding", **mid_check)
            raise RuntimeError(f"Insufficient storage for encoding: {mid_check}")

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
        encoder_spec = config.encoders.get("z_image_vae")
        stage_fingerprint_data = {
            "stage_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "encoder": vars(encoder_spec) if encoder_spec is not None else None,
            "max_encoding_resolution": max_enc_size,
            "canny_low_threshold": canny_low,
            "canny_high_threshold": canny_high,
            "opencv_version": getattr(cv2, "__version__", None) if _HAS_OPENCV else None,
        }
        def _atomic_json(path: Path, data: dict[str, Any]) -> None:
            temp_path = path.with_name(f".{path.name}.tmp")
            try:
                temp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                os.replace(temp_path, path)
            finally:
                temp_path.unlink(missing_ok=True)

        def _atomic_latent(
            path: Path, tensors: dict[str, torch.Tensor], input_fingerprint: str
        ) -> None:
            temp_path = path.with_name(f".{path.name}.tmp")
            try:
                save_file(
                    tensors,
                    str(temp_path),
                    metadata={"input_fingerprint": input_fingerprint},
                )
                os.replace(temp_path, path)
            finally:
                temp_path.unlink(missing_ok=True)

        def _record_fingerprint(rec: ManifestRecord, img_path: Path) -> str:
            stat = img_path.stat()
            data = {
                "stage": stage_fingerprint,
                "image_path": _rel(img_path),
                "image_size": stat.st_size,
                "image_mtime_ns": stat.st_mtime_ns,
                "structure_output": rec.structure_output,
            }
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

        def _cache_is_valid(
            rec: ManifestRecord,
            z_path: Path,
            control_path: Path,
            edges_path: Path,
            fingerprint: str,
        ) -> bool:
            try:
                control_data = json.loads(control_path.read_text(encoding="utf-8"))
                if control_data.get("record_id") != rec.id:
                    return False
                edge_path = control_data.get("canny_edges_path")
                if edge_path and (
                    edge_path != _rel(edges_path) or not edges_path.is_file()
                ):
                    return False
                with safe_open(str(z_path), framework="pt", device="cpu") as latent_file:
                    metadata = latent_file.metadata() or {}
                    return (
                        metadata.get("input_fingerprint") == fingerprint
                        and "latent" in latent_file.keys()
                    )
            except Exception as error:
                log.warning("invalid_encoding_cache", record_id=rec.id, error=str(error))
                return False

        # Pre-load to avoid race conditions
        try:
            z_vae = engine.get_encoder("z_image_vae")
            enc_dev = "cuda" if torch.cuda.is_available() else "cpu"
            enc_dtype = torch.float16 if enc_dev == "cuda" else torch.float32
        except Exception as e:
            log.error("vae_load_failed", error=str(e))
            raise RuntimeError("Failed to load the required Z-Image VAE encoder") from e

        vae_batch_size = stage_cfg.get("vae_batch_size", 16 if enc_dev == "cuda" else 4)
        if vae_batch_size < 1:
            raise ValueError("s08_encoding.vae_batch_size must be at least 1")
        stage_fingerprint_data["vae_batch_size"] = vae_batch_size
        stage_fingerprint_data["device"] = enc_dev
        stage_fingerprint_data["dtype"] = str(enc_dtype)
        stage_fingerprint = hashlib.sha256(
            json.dumps(stage_fingerprint_data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        def _preprocess_record(rec: ManifestRecord) -> dict[str, Any]:
            try:
                z_path = paths["latents_zimage"] / f"{rec.id}.safetensors"
                control_path = paths["control_tokens"] / f"{rec.id}.json"
                edges_path = paths["control_tokens"] / f"{rec.id}_edges.png"
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
                fingerprint = _record_fingerprint(rec, img_path)
                if (
                    z_path.is_file()
                    and control_path.is_file()
                    and z_path.stat().st_size > 0
                    and control_path.stat().st_size > 0
                    and _cache_is_valid(rec, z_path, control_path, edges_path, fingerprint)
                ):
                    encoding_paths: dict[str, str] = {
                        "z_image_latent": _rel(z_path),
                        "control_map": _rel(control_path),
                    }
                    record_bytes = z_path.stat().st_size + control_path.stat().st_size
                    control_data = json.loads(control_path.read_text(encoding="utf-8"))
                    if control_data.get("canny_edges_path"):
                        encoding_paths["canny_edges"] = _rel(edges_path)
                        record_bytes += edges_path.stat().st_size
                    return {
                        "type": "done",
                        "id": rec.id,
                        "status": "encoded",
                        "encoding_paths": encoding_paths,
                        "bytes": record_bytes,
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
                            temp_edges_path = edges_path.with_name(f".{edges_path.name}.tmp")
                            try:
                                PILImage.fromarray(edges).save(temp_edges_path, format="PNG")
                                os.replace(temp_edges_path, edges_path)
                            finally:
                                temp_edges_path.unlink(missing_ok=True)
                            edges_rel_path = _rel(edges_path)
                            record_bytes += edges_path.stat().st_size
                        except Exception as e:
                            log.warning("canny_compute_failed", record_id=rec.id, error=str(e))
                    else:
                        log.warning("canny_skipped_no_opencv", record_id=rec.id, note="opencv-python-headless not installed")

                    control_data["canny_edges_path"] = edges_rel_path
                    control_data["canny_thresholds"] = [canny_low, canny_high]

                    _atomic_json(control_path, control_data)
                    encoding_paths["control_map"] = _rel(control_path)
                    if edges_rel_path:
                        encoding_paths["canny_edges"] = edges_rel_path
                    record_bytes += control_path.stat().st_size
                except Exception as e:
                    log.warning("control_map_failed", record_id=rec.id, error=str(e))

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
                    "input_fingerprint": fingerprint,
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

        window_size = max(64, min(512, vae_batch_size * 8))
        max_workers = min(32, max(4, multiprocessing.cpu_count()))

        def _encode_preprocessed_window(
            prep_results: list[dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], int, int, int]:
            window_updates: list[dict[str, Any]] = []
            window_processed = window_failed = window_bytes = 0
            needs_vae_by_res: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

            for res in prep_results:
                if res["type"] == "done":
                    item: dict[str, Any] = {
                        "id": res["id"],
                        "new_status": res["status"],
                    }
                    if res["status"] == "encoded":
                        item["encoding_paths"] = res["encoding_paths"]
                        window_processed += 1
                        window_bytes += res["bytes"]
                    else:
                        item["reason"] = res.get("reason")
                        item["exclusion_reason"] = res.get("exclusion_reason")
                        window_failed += 1
                    window_updates.append(item)
                else:
                    needs_vae_by_res[res["resolution"]].append(res)

            for res_items in needs_vae_by_res.values():
                for b_start in range(0, len(res_items), vae_batch_size):
                    batch = res_items[b_start : b_start + vae_batch_size]
                    batch_tensor = torch.stack(
                        [item["tensor"] for item in batch]
                    ).to(enc_dev, dtype=enc_dtype, non_blocking=True)
                    try:
                        with torch.inference_mode():
                            batch_latents = z_vae.encode(batch_tensor).latent_dist.sample().cpu()
                        for item, latent in zip(batch, batch_latents):
                            _atomic_latent(
                                item["z_path"],
                                {"latent": latent.unsqueeze(0).contiguous()},
                                item["input_fingerprint"],
                            )
                            item["encoding_paths"]["z_image_latent"] = _rel(item["z_path"])
                            item["record_bytes"] += item["z_path"].stat().st_size
                    except Exception as batch_error:
                        log.warning(
                            "z_image_batch_encode_fallback",
                            batch_size=len(batch),
                            error=str(batch_error),
                        )
                        if enc_dev == "cuda" and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        for item in batch:
                            try:
                                single_tensor = item["tensor"].unsqueeze(0).to(
                                    enc_dev, dtype=enc_dtype
                                )
                                with torch.inference_mode():
                                    single_latent = (
                                        z_vae.encode(single_tensor).latent_dist.sample().cpu()
                                    )
                                _atomic_latent(
                                    item["z_path"],
                                    {"latent": single_latent.contiguous()},
                                    item["input_fingerprint"],
                                )
                                item["encoding_paths"]["z_image_latent"] = _rel(item["z_path"])
                                item["record_bytes"] += item["z_path"].stat().st_size
                            except Exception as error:
                                log.warning(
                                    "z_image_encode_failed",
                                    record_id=item["id"],
                                    error=str(error),
                                )

                    for item in batch:
                        rec_item: dict[str, Any] = {"id": item["id"]}
                        if not {
                            "z_image_latent",
                            "control_map",
                        }.issubset(item["encoding_paths"]):
                            rec_item.update({
                                "new_status": "excluded_failed",
                                "reason": "Required latent or control-map artifact was not produced",
                                "exclusion_reason": "encoding_incomplete",
                            })
                            window_failed += 1
                        else:
                            rec_item["new_status"] = "encoded"
                            rec_item["encoding_paths"] = item["encoding_paths"]
                            window_processed += 1
                            window_bytes += item["record_bytes"]
                        window_updates.append(rec_item)

            return window_updates, window_processed, window_failed, window_bytes

        sentinel = object()
        input_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=window_size)
        output_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=window_size * 2)
        loop = asyncio.get_running_loop()
        updates: list[dict[str, Any]] = []

        async def _feed_records() -> None:
            for record in records:
                await input_queue.put(record)
            for _ in range(max_workers):
                await input_queue.put(sentinel)

        async def _preprocess_worker(
            executor: concurrent.futures.ThreadPoolExecutor,
        ) -> None:
            while True:
                record = await input_queue.get()
                if record is sentinel:
                    await output_queue.put(sentinel)
                    return
                try:
                    prepared = await loop.run_in_executor(executor, _preprocess_record, record)
                except Exception as error:
                    prepared = {
                        "type": "done",
                        "id": record.id,
                        "status": "excluded_failed",
                        "reason": f"Preprocessing worker failed: {error}",
                        "exclusion_reason": "encoding_preprocess_error",
                    }
                await output_queue.put(prepared)

        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as cpu_executor,
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as gpu_executor,
        ):
            feed_task = asyncio.create_task(_feed_records())
            worker_tasks = [
                asyncio.create_task(_preprocess_worker(cpu_executor))
                for _ in range(max_workers)
            ]
            finished_workers = 0
            prep_window: list[dict[str, Any]] = []
            try:
                while finished_workers < max_workers:
                    prepared = await output_queue.get()
                    if prepared is sentinel:
                        finished_workers += 1
                        continue
                    prep_window.append(prepared)
                    if len(prep_window) < window_size and finished_workers < max_workers:
                        continue
                    encoded, window_processed, window_failed, window_bytes = (
                        await loop.run_in_executor(
                            gpu_executor, _encode_preprocessed_window, prep_window
                        )
                    )
                    updates.extend(encoded)
                    processed += window_processed
                    failed += window_failed
                    total_bytes += window_bytes
                    prep_window.clear()
                    if len(updates) >= 200:
                        manifest.bulk_update_records(updates, stage="encoding")
                        updates.clear()
                if prep_window:
                    encoded, window_processed, window_failed, window_bytes = (
                        await loop.run_in_executor(
                            gpu_executor, _encode_preprocessed_window, prep_window
                        )
                    )
                    updates.extend(encoded)
                    processed += window_processed
                    failed += window_failed
                    total_bytes += window_bytes
                await feed_task
                await asyncio.gather(*worker_tasks)
            finally:
                for task in [feed_task, *worker_tasks]:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(feed_task, *worker_tasks, return_exceptions=True)

        if updates:
            manifest.bulk_update_records(updates, stage="encoding")

        result.records_processed = processed
        result.records_failed = failed
        elapsed_seconds = time.monotonic() - started_at
        records_per_second = processed / elapsed_seconds if elapsed_seconds else 0.0
        result.metadata = {
            "total_bytes_written": total_bytes,
            "elapsed_seconds": elapsed_seconds,
            "records_per_second": records_per_second,
            "preprocess_workers": max_workers,
            "preprocess_window_size": window_size,
            "vae_batch_size": vae_batch_size,
        }
        log.info("encoding_complete", processed=processed, failed=failed,
                 bytes_written=total_bytes, gb_written=round(total_bytes / 1e9, 3),
                 elapsed_seconds=round(elapsed_seconds, 2),
                 records_per_second=round(records_per_second, 2),
                 preprocess_workers=max_workers, preprocess_window_size=window_size,
                 vae_batch_size=vae_batch_size)
        return result
