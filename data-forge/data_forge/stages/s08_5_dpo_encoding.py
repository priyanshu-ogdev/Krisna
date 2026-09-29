"""Stage 8.5: DPO Latent Encoding.

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

import json
from typing import Any, ClassVar

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

        for source_dir in sorted(p for p in pref_root.iterdir() if p.is_dir()):
            source_key = source_dir.name
            out_dir = dpo_root / source_key
            out_dir.mkdir(parents=True, exist_ok=True)

            for meta_path in sorted(source_dir.glob("*.json")):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception as e:
                    log.warning("dpo_encoding_bad_metadata", path=str(meta_path), error=str(e))
                    failed += 1
                    continue

                if meta.get("dedup_status") != "unique":
                    # Never processed by s01_6 (duplicate/corrupt/not yet
                    # run) — do not encode it as if it were clean.
                    skipped += 1
                    continue

                pair_id = meta["pair_id"]
                out_path = out_dir / f"{pair_id}.safetensors"
                if out_path.exists():
                    continue  # idempotent re-run

                try:
                    img_a = pad_to_multiple(load_image(meta_path.parent / meta["image_a"]), 16)
                    img_b = pad_to_multiple(load_image(meta_path.parent / meta["image_b"]), 16)

                    t_a = normalize_for_vae(image_to_tensor(img_a)).unsqueeze(0).to("cuda", dtype=torch.float16)
                    t_b = normalize_for_vae(image_to_tensor(img_b)).unsqueeze(0).to("cuda", dtype=torch.float16)
                    with torch.no_grad():
                        lat_a = z_vae.encode(t_a).latent_dist.sample()
                        lat_b = z_vae.encode(t_b).latent_dist.sample()

                    from safetensors.torch import save_file
                    save_file(
                        {"latent_a": lat_a.cpu(), "latent_b": lat_b.cpu()},
                        str(out_path),
                    )
                    (out_dir / f"{pair_id}.meta.json").write_text(
                        json.dumps({
                            "pair_id": pair_id,
                            "prompt": meta.get("prompt", ""),
                            "preferred": meta.get("preferred"),  # "a" or "b"
                            "origin": meta.get("origin", source_key),
                            "label_source": meta.get("label_source", "human"),
                        }),
                        encoding="utf-8",
                    )
                    processed += 1

                except Exception as e:
                    log.warning("dpo_pair_encode_failed", pair=pair_id, error=str(e))
                    failed += 1

        result.records_processed = processed
        result.records_failed = failed
        result.metadata = {"skipped_not_deduped": skipped}
        log.info("dpo_encoding_complete", processed=processed, failed=failed, skipped=skipped)
        return result
