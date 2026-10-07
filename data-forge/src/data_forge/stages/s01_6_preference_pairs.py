"""Stage 1.6: Preference-Pair Post-Processing (replaces the old synthetic
planner-conversation-synthesis stage).

Runs once, globally, after every preference-pair dataset (Pick-a-Pic v2,
HPDv2, DesignSense-10k, DesignPref — download_mode `"preference_pair"` or
HPDv2's dedicated `"hpdv2_ranked_list"`, see
fetcher.py::_fetch_huggingface_preference_pairs and
::_fetch_hpdv2_ranked_pairs) has written its raw (prompt, image_a,
image_b, preferred) triples into preference_pairs/{key}/.

Why this exists as its own stage rather than folding dedup/PII/safety into
the fetcher: the fetcher already does a *cheap* within-source exact-hash
guard inline (see `_fetch_huggingface_preference_pairs`'s `seen_hashes`
set), but three real gaps remain that need the same care the main image
pipeline gives every other record:

  1. Cross-source duplicates. Pick-a-Pic and HPDv2 both draw candidate
     images from overlapping public T2I-model output pools; the same
     generated image can legitimately appear in both datasets' pair
     sets. Left alone, that image's "quality" would be represented
     twice in the DPO training pool with no signal that it happened.
  2. PII. These are model-generated images, but T2I models can and do
     render photorealistic faces. The PRD's "never let un-blurred faces
     reach training encodings" rule (see s03_5_pii_scrub.py) is not
     specific to real photographs — it applies here too, via the same
     shared face-blur helper (utils/pii_faces.py) the main image
     pipeline uses, not a separate, drift-prone reimplementation.
  3. Content safety. BUG FIX (previously missing entirely): an earlier
     revision of this stage ran with no engine and never classified
     preference-pair images for NSFW/harmful content at all — a real
     gap, since these are T2I-model outputs, not manifest-vetted
     content, and nothing else in this pipeline screens them before
     they'd reach s08_5_dpo_encoding. Now uses the same Tier-1
     `classify_safety` path s04_safety.py uses for the main manifest.
     Any pair with either image at safety tier "unsafe" is dropped
     entirely (never marked `dedup_status: "unique"`, so
     s08_5_dpo_encoding's "only encode unique pairs" guard skips it too
     — no separate exclusion list to keep in sync). "borderline" pairs
     are kept but flagged (`safety_tier` in the metadata) rather than
     routed through a full Tier-2 escalation pass — this stage
     deliberately doesn't duplicate s04_5_escalation.py's machinery for
     what is, in practice, a much smaller and already largely-curated
     preference-pair volume; audit borderline pairs before a production
     DPO run rather than assuming this stage cleared them.

No AI-judge *labeling* happens here — these are real, already-published
human comparisons (see the removed s10_5_critic_preference.py and the
PRD's no-RLHF-loop revision). Content-safety classification is a
different, non-optional check that applies regardless of who labeled the
preference, not a step toward generating synthetic preference data.

If TASTE/PartiPrompts-style eval-only sources ever accidentally show up in
preference_pairs/ (they shouldn't — see fetcher.py::_fetch_eval_reference,
which never writes here), this stage's `eval_only` guard drops them
rather than silently processing eval data as if it were training data.
"""

import asyncio
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import threading
from typing import Any, ClassVar

from PIL import Image, UnidentifiedImageError

from data_forge.config import DatasetSpec, PipelineConfig
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.pii_faces import blur_faces, load_face_detector

log = get_logger("stages.s01_6")


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


@register_stage("s01_6_preference_pairs")
class PreferencePairsStage(Stage):
    name = "s01_6_preference_pairs"
    requires: ClassVar[tuple[str, ...]] = ("s01_fetch",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        stage_cfg = config.get_stage("s01_6_preference_pairs")
        face_conf = stage_cfg.get("face_detection_confidence", 0.5)
        blur_kernel = stage_cfg.get("blur_kernel_size", 99)
        safety_conf_threshold = stage_cfg.get("safety_confidence_threshold", 0.8)

        pref_root = config.resolved_paths["preference_pairs"]
        if not pref_root.exists():
            log.info("preference_pairs_root_missing", path=str(pref_root))
            return result

        preference_sources = {
            key: spec for key, spec in config.datasets.items()
            # BUG FIX: previously only matched "preference_pair" — see
            # DatasetSpec.PREFERENCE_PAIR_DOWNLOAD_MODES in config.py for
            # why this is now the single, centralized set (avoids this
            # filter and the storage-projection methods in config.py
            # silently drifting apart again the next time a new
            # dedicated adapter mode is added).
            if spec.fetch_config.get("download_mode") in DatasetSpec.PREFERENCE_PAIR_DOWNLOAD_MODES
        }
        if not preference_sources:
            log.info("no_preference_pair_sources_configured")
            return result

        if engine is None:
            # Hard-fail rather than silently skipping safety classification
            # — the whole point of the earlier bug fix is that this stage
            # must not process preference pairs without a safety pass. If
            # the orchestrator ever calls this stage without a Tier-1
            # session again, that's a wiring regression that should be
            # loud, not a quiet no-op that lets unscreened T2I images
            # reach dpo_alignment/.
            log.error(
                "preference_pairs_no_engine",
                note="s01_6_preference_pairs requires a Tier-1 engine for safety "
                     "classification — see orchestrator.py's ModelEngine.vllm_session"
                     "(self.config, \"tier1\") wiring for this stage. Refusing to "
                     "process any pairs without it rather than skipping safety checks.",
            )
            return result
        tier1 = Tier1Engine(engine, config)

        face_detector = load_face_detector(min_confidence=face_conf)
        if face_detector is not None:
            log.info("mediapipe_loaded_for_preference_pairs")
        else:
            log.warning(
                "mediapipe_not_available",
                note="Preference-pair images will not be face-blurred this run.",
            )

        seen_pair_hashes: set[str] = set()
        kept = 0
        dropped_duplicate = 0
        dropped_corrupt = 0
        dropped_unsafe = 0
        flagged_borderline = 0
        total_faces_blurred = 0

        semaphore = asyncio.Semaphore(128)
        max_workers = min(32, max(4, multiprocessing.cpu_count()))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        thread_local = threading.local()
        worker_detectors: list[Any] = []
        detector_lock = threading.Lock()

        has_detector = face_detector is not None
        hash_lock = threading.Lock()

        def _get_detector():
            if not has_detector:
                return None
            if not hasattr(thread_local, "detector"):
                thread_local.detector = load_face_detector(min_confidence=face_conf)
                if thread_local.detector is not None:
                    with detector_lock:
                        worker_detectors.append(thread_local.detector)
            return thread_local.detector

        def _cpu_tasks(meta_path):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                image_names = (meta.get("image_a"), meta.get("image_b"))
                if any(
                    not isinstance(name, str)
                    or not name
                    or name in {".", ".."}
                    or Path(name).is_absolute()
                    or Path(name).name != name
                    or "\\" in name
                    for name in image_names
                ):
                    return None, "corrupt", meta_path, "Unsafe preference-pair image path", None, None, None, None
                pair_dir = meta_path.parent.resolve()
                image_paths = [(meta_path.parent / name).resolve(strict=True) for name in image_names]
                if any(path.parent != pair_dir or not path.is_file() for path in image_paths):
                    return None, "corrupt", meta_path, "Preference-pair image escaped its source directory", None, None, None, None
                img_a_path, img_b_path = image_paths

                try:
                    with Image.open(img_a_path) as source_a:
                        format_a = source_a.format
                        img_a = source_a.convert("RGB")
                    with Image.open(img_b_path) as source_b:
                        format_b = source_b.format
                        img_b = source_b.convert("RGB")
                except (UnidentifiedImageError, FileNotFoundError, OSError):
                    return None, "corrupt", meta_path, None, None, None, None, None

                hash_a = hashlib.sha256(img_a.tobytes()).hexdigest()
                hash_b = hashlib.sha256(img_b.tobytes()).hexdigest()
                
                if hash_a == hash_b:
                    return None, "self_duplicate", meta_path, meta, None, None, None, None
                
                pair_hash = f"{min(hash_a, hash_b)}_{max(hash_a, hash_b)}"

                with hash_lock:
                    if pair_hash in seen_pair_hashes:
                        return pair_hash, "cross_duplicate", meta_path, meta, False, 0, None, None
                    seen_pair_hashes.add(pair_hash)

                detector = _get_detector()
                if has_detector and detector is None:
                    return None, "corrupt", meta_path, "Face detector failed to initialize", None, None, None, None
                img_a, mod_a, det_a = blur_faces(img_a, detector, blur_kernel)
                img_b, mod_b, det_b = blur_faces(img_b, detector, blur_kernel)
                
                if mod_a:
                    if format_a not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                        return None, "corrupt", meta_path, f"Unsupported image format for redaction: {format_a}", None, None, None, None
                    temp_path = img_a_path.with_name(f".{img_a_path.name}.redacting")
                    img_a.save(
                        temp_path,
                        format=format_a,
                        **({"quality": 95} if format_a == "JPEG" else {}),
                    )
                    os.replace(temp_path, img_a_path)
                if mod_b:
                    if format_b not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                        return None, "corrupt", meta_path, f"Unsupported image format for redaction: {format_b}", None, None, None, None
                    temp_path = img_b_path.with_name(f".{img_b_path.name}.redacting")
                    img_b.save(
                        temp_path,
                        format=format_b,
                        **({"quality": 95} if format_b == "JPEG" else {}),
                    )
                    os.replace(temp_path, img_b_path)
                
                faces_blurred = len(det_a) + len(det_b)
                return pair_hash, "ok", meta_path, meta, mod_a or mod_b, faces_blurred, img_a_path, img_b_path
            except Exception as e:
                return None, "error", meta_path, str(e), None, None, None, None

        async def _process_pair(meta_path):
            nonlocal kept, dropped_duplicate, dropped_corrupt, dropped_unsafe, flagged_borderline, total_faces_blurred
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(executor, _cpu_tasks, meta_path)
            pair_hash, status, m_path, m_data, pii_mod, f_blurred, a_path, b_path = res

            if status == "corrupt":
                dropped_corrupt += 1
                return
            if status == "error":
                log.error("preference_pair_processing_failed", pair=str(m_path), error=m_data)
                dropped_corrupt += 1
                return
            if status == "self_duplicate":
                log.warning("preference_pair_degenerate_self_pair", pair=m_data.get("pair_id"))
                dropped_duplicate += 1
                return
            if status == "cross_duplicate":
                dropped_duplicate += 1
                return

            async with semaphore:
                try:
                    safety_results = await tier1.batch_classify_safety([a_path, b_path])
                except Exception as e:
                    safety_results = None
            
            if not safety_results or len(safety_results) < 2 or safety_results[0] is None or safety_results[1] is None:
                log.warning("preference_pair_safety_inference_failed", pair=m_data.get("pair_id"))
                dropped_corrupt += 1
                return
            
            safety_a, safety_b = safety_results[0], safety_results[1]
            if safety_a.tier == "unsafe" or safety_b.tier == "unsafe":
                log.warning("preference_pair_dropped_unsafe", pair=m_data.get("pair_id"), tier_a=safety_a.tier, tier_b=safety_b.tier)
                dropped_unsafe += 1
                return

            is_borderline = (
                safety_a.tier == "borderline" or safety_b.tier == "borderline"
                or safety_a.confidence < safety_conf_threshold
                or safety_b.confidence < safety_conf_threshold
            )
            if is_borderline:
                flagged_borderline += 1
            
            total_faces_blurred += f_blurred

            m_data["dedup_status"] = "unique"
            m_data["pii_scrubbed"] = has_detector
            m_data["safety_tier"] = "borderline" if is_borderline else "safe"
            
            await asyncio.to_thread(_write_json_file, m_path, m_data)
            kept += 1

        try:
            for source_key, spec in preference_sources.items():
                if spec.eval_only:
                    log.warning("eval_only_source_in_preference_pairs_skipped", dataset=source_key)
                    continue

                src_dir = pref_root / source_key
                if not src_dir.exists():
                    log.warning("preference_pair_source_dir_missing", dataset=source_key, path=str(src_dir))
                    continue

                pair_meta_files = sorted(src_dir.glob("*.json"))
                log.info("preference_pairs_processing_source", dataset=source_key, pairs=len(pair_meta_files))

                # Batch run pairs in bounded chunks to prevent unbounded memory backlog
                chunk_size = 256
                for c_start in range(0, len(pair_meta_files), chunk_size):
                    chunk = pair_meta_files[c_start : c_start + chunk_size]
                    tasks = [_process_pair(p) for p in chunk]
                    if tasks:
                        await asyncio.gather(*tasks)

        finally:
            executor.shutdown(wait=True)
            for detector in worker_detectors:
                detector.close()
            if face_detector is not None:
                face_detector.close()

        result.records_processed = kept
        result.records_failed = dropped_corrupt
        result.records_excluded = dropped_duplicate + dropped_unsafe
        result.metadata = {
            "sources": list(preference_sources.keys()),
            "kept": kept,
            "dropped_duplicate": dropped_duplicate,
            "dropped_corrupt": dropped_corrupt,
            "dropped_unsafe": dropped_unsafe,
            "flagged_borderline": flagged_borderline,
            "faces_blurred": total_faces_blurred,
        }
        log.info(
            "preference_pairs_complete",
            kept=kept,
            dropped_duplicate=dropped_duplicate,
            dropped_corrupt=dropped_corrupt,
            dropped_unsafe=dropped_unsafe,
            flagged_borderline=flagged_borderline,
            faces_blurred=total_faces_blurred,
        )
        return result
