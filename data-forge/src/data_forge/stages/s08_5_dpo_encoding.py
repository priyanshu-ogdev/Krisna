"""Stage 8.5: DPO Latent Encoding.

DISABLED BY DEFAULT (configs/pipeline.yaml `s08_5_dpo_encoding.enabled:
false`) — confirmed, this pass, to have zero real consumers anywhere in
training/ or inference/. `train_dpo.py` resolves chosen_ref/rejected_ref
through the shared BlobStore and re-encodes from raw images itself
(`vae.encode(batch["chosen_pixel_values"]...)`), the same way
sync_sketch_tier.py bypasses the Sketch tier's own data-forge encoding.
Every enabled run of this stage was spending real GPU-encode time and
disk space on `.safetensors` latents nothing reads — see
docs/review/02_polish_tier.md's original finding (same orphaned-producer
shape as the removed `maskgit_vq` stage) and
docs/review/15_data_forge_finalization.md for this pass's disposition.
Left in the codebase rather than deleted outright, unlike maskgit_vq,
because there's a real option worth deciding on deliberately rather than
foreclosing: wiring this into train_dpo.py as a fast path (skip live
VAE-encoding when precomputed latents already exist) would be a genuine
speed win if DPO training ever becomes GPU-time-constrained. Re-enable
only after that wiring exists, or for a specific one-off analysis that
needs the latents directly — not for a normal pipeline run.

Encodes the deduped, face-blurred preference pairs from
s01_6_preference_pairs.py into Z-Image-Turbo's latent space, ready for a
standard Diffusion-DPO training loop.

Deliberately Z-Image-Turbo only, not a "Tri-Path" encoding like
s08_encoding.py: Z-Image-Turbo is the only renderer this project actually
fine-tunes. Qwen-Image-Edit-2511 ships frozen (zero-shot ICL + SDEdit at
inference time — see PRD "no-RLHF-loop" revision), so it never needs
training latents, and the sketch tier doesn't participate in DPO at all
(preference data here is general/design-aesthetic, not sparse-token
layout). If Qwen-Image-Edit-2511 is ever un-frozen and given its own DPO
pass, add a second branch here the same way s08_encoding.py's Branch 2
used to exist — do not silently repurpose this stage's output for it.

Runs once, globally, independent of the main record manifest — preference
pairs never entered it in the first place (see fetcher.py's "not manifest
records — written directly to preference_pairs/" pattern).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

from safetensors.torch import save_file
import torch

from data_forge.config import PipelineConfig
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult
from data_forge.utils.image_utils import image_to_tensor, load_image, normalize_for_vae, pad_to_multiple

log = get_logger("stages.s08_5")


@register_stage("s08_5_dpo_encoding")
class DPOEncodingStage(Stage):
    name = "s08_5_dpo_encoding"
    requires: ClassVar[tuple[str, ...]] = ("s01_6_preference_pairs",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        if engine is None:
            log.warning("dpo_encoding_no_engine", note="Encoder session not provided — skipping.")
            return result

        pref_root = config.resolved_paths["preference_pairs"]
        dpo_root = config.resolved_paths["dpo_latents"]
        if not pref_root.exists():
            log.info("preference_pairs_root_missing", path=str(pref_root))
            return result

        try:
            z_vae = engine.get_encoder("z_image_vae")
        except Exception as e:
            log.error("dpo_encoding_no_zimage_encoder", error=str(e))
            return result

        processed = failed = skipped = 0
        enc_dev = "cuda" if torch.cuda.is_available() else "cpu"
        enc_dtype = torch.float16 if enc_dev == "cuda" else torch.float32

        def _process_dpo_pairs():
            p_processed = p_failed = p_skipped = 0
            for source_dir in sorted(p for p in pref_root.iterdir() if p.is_dir()):
                source_key = source_dir.name
                out_dir = dpo_root / source_key
                out_dir.mkdir(parents=True, exist_ok=True)

                for meta_path in sorted(source_dir.glob("*.json")):
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        log.warning("dpo_encoding_bad_metadata", path=str(meta_path), error=str(e))
                        p_failed += 1
                        continue

                    if (
                        meta.get("dedup_status") != "unique"
                        or meta.get("safety_tier") != "safe"
                        or meta.get("pii_scrubbed") is not True
                        or meta.get("text_pii_scrubbed") is not True
                    ):
                        p_skipped += 1
                        continue

                    pair_id = meta["pair_id"]
                    out_path = out_dir / f"{pair_id}.safetensors"
                    if out_path.exists():
                        p_processed += 1
                        continue  # idempotent re-run

                    try:
                        p_a = Path(meta["image_a"])
                        p_b = Path(meta["image_b"])
                        path_a = p_a if p_a.is_absolute() else (meta_path.parent / p_a)
                        path_b = p_b if p_b.is_absolute() else (meta_path.parent / p_b)
                        img_a = pad_to_multiple(load_image(path_a), 16)
                        img_b = pad_to_multiple(load_image(path_b), 16)

                        t_a = normalize_for_vae(image_to_tensor(img_a)).to(enc_dev, dtype=enc_dtype)
                        t_b = normalize_for_vae(image_to_tensor(img_b)).to(enc_dev, dtype=enc_dtype)
                        with torch.inference_mode():
                            if t_a.shape == t_b.shape:
                                batch_t = torch.stack([t_a, t_b], dim=0)
                                batch_lat = z_vae.encode(batch_t).latent_dist.sample()
                                lat_a = batch_lat[0:1]
                                lat_b = batch_lat[1:2]
                            else:
                                lat_a = z_vae.encode(t_a.unsqueeze(0)).latent_dist.sample()
                                lat_b = z_vae.encode(t_b.unsqueeze(0)).latent_dist.sample()

                        lat_a_cpu = lat_a.cpu()
                        lat_b_cpu = lat_b.cpu()
                        if not (torch.isfinite(lat_a_cpu).all() and torch.isfinite(lat_b_cpu).all()):
                            raise ValueError(f"Corrupted DPO latent in pair {pair_id}: contains NaN or Inf values")

                        tmp_out = out_path.with_name(f".{out_path.name}.tmp")
                        try:
                            save_file(
                                {"latent_a": lat_a_cpu, "latent_b": lat_b_cpu},
                                str(tmp_out),
                            )
                            import os
                            os.replace(tmp_out, out_path)
                        finally:
                            tmp_out.unlink(missing_ok=True)

                        meta_tmp = (out_dir / f".{pair_id}.meta.json.tmp")
                        try:
                            meta_tmp.write_text(
                                json.dumps({
                                    "pair_id": pair_id,
                                    "prompt": meta.get("prompt", ""),
                                    "preferred": meta.get("preferred"),  # "a" or "b"
                                    "origin": meta.get("origin", source_key),
                                    "label_source": meta.get("label_source", "human"),
                                }),
                                encoding="utf-8",
                            )
                            import os
                            os.replace(meta_tmp, out_dir / f"{pair_id}.meta.json")
                        finally:
                            meta_tmp.unlink(missing_ok=True)
                        p_processed += 1

                    except Exception as e:
                        log.warning("dpo_pair_encode_failed", pair=pair_id, error=str(e))
                        p_failed += 1
            return p_processed, p_failed, p_skipped
            
        loop = asyncio.get_running_loop()
        processed, failed, skipped = await loop.run_in_executor(None, _process_dpo_pairs)

        result.records_processed = processed
        result.records_failed = failed
        result.metadata = {"skipped_not_deduped": skipped}
        log.info("dpo_encoding_complete", processed=processed, failed=failed, skipped=skipped)
        return result
