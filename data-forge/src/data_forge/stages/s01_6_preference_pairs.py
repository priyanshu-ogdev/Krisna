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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

from data_forge.config import DatasetSpec, PipelineConfig
from data_forge.inference.tier1 import Tier1Engine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.pii_faces import blur_faces, load_face_detector

log = get_logger("stages.s01_6")


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

        for source_key, spec in preference_sources.items():
            if spec.eval_only:
                # Should never happen (eval_reference sources never write
                # here) — guarded explicitly anyway, see module docstring.
                log.warning("eval_only_source_in_preference_pairs_skipped", dataset=source_key)
                continue

            src_dir = pref_root / source_key
            if not src_dir.exists():
                log.warning("preference_pair_source_dir_missing", dataset=source_key, path=str(src_dir))
                continue

            pair_meta_files = sorted(src_dir.glob("*.json"))
            log.info("preference_pairs_processing_source", dataset=source_key, pairs=len(pair_meta_files))

            for meta_path in pair_meta_files:
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    img_a_path = meta_path.parent / meta["image_a"]
                    img_b_path = meta_path.parent / meta["image_b"]

                    from PIL import Image, UnidentifiedImageError
                    try:
                        img_a = Image.open(img_a_path).convert("RGB")
                        img_b = Image.open(img_b_path).convert("RGB")
                    except (UnidentifiedImageError, FileNotFoundError, OSError):
                        dropped_corrupt += 1
                        continue

                    pair_hash = (
                        hashlib.sha256(img_a.tobytes()).hexdigest()
                        + hashlib.sha256(img_b.tobytes()).hexdigest()
                    )
                    if pair_hash in seen_pair_hashes:
                        dropped_duplicate += 1
                        continue
                    seen_pair_hashes.add(pair_hash)

                    img_a, mod_a, det_a = blur_faces(img_a, face_detector, blur_kernel)
                    img_b, mod_b, det_b = blur_faces(img_b, face_detector, blur_kernel)
                    if mod_a:
                        img_a.save(img_a_path, quality=95)
                    if mod_b:
                        img_b.save(img_b_path, quality=95)
                    total_faces_blurred += len(det_a) + len(det_b)

                    # Content-safety classification — the fix for the gap
                    # noted in the module docstring. Runs on the
                    # (already face-blurred) files on disk, same as
                    # s04_safety.py does for the main manifest.
                    safety_a, safety_b = await tier1.batch_classify_safety(
                        [img_a_path, img_b_path]
                    )
                    if safety_a is None or safety_b is None:
                        log.warning("preference_pair_safety_inference_failed", pair=meta.get("pair_id"))
                        dropped_corrupt += 1
                        continue
                    if safety_a.tier == "unsafe" or safety_b.tier == "unsafe":
                        log.warning(
                            "preference_pair_dropped_unsafe",
                            pair=meta.get("pair_id"),
                            tier_a=safety_a.tier, tier_b=safety_b.tier,
                        )
                        dropped_unsafe += 1
                        continue
                    is_borderline = (
                        safety_a.tier == "borderline" or safety_b.tier == "borderline"
                        or safety_a.confidence < safety_conf_threshold
                        or safety_b.confidence < safety_conf_threshold
                    )
                    if is_borderline:
                        flagged_borderline += 1

                    meta["dedup_status"] = "unique"
                    meta["pii_scrubbed"] = bool(mod_a or mod_b)
                    meta["safety_tier"] = "borderline" if is_borderline else "safe"
                    meta_path.write_text(json.dumps(meta), encoding="utf-8")
                    kept += 1

                except Exception as e:
                    log.error("preference_pair_processing_failed", pair=str(meta_path), error=str(e))
                    dropped_corrupt += 1

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
