"""Bridges data-forge's `model_data/` export (see that project's
`s12_model_data_export.py`) into the exact input formats this project's
own training packages expect (`training/sketch/dataset.py`,
`training/polish/dataset_prep.py`, `dpo/preference_store.py`).

Two real format mismatches were found while building this, not papered
over:

1. **Sketch tier's VQ tokenizer identity didn't match, and data-forge's
   side has since been deleted rather than fixed.** data-forge used to
   produce `sketch_tier_maskgit/vq_tokens/*.pt` via a `maskgit_vq`
   encoder (TencentARC/Open-MAGVIT2) — but that encoder's loading code in
   data-forge's `engine.py` raised `RuntimeError` on purpose, on every
   single call, because Open-MAGVIT2 needs a custom `.encode()`/
   `.decode()` wrapper that was never built. THIS project's
   `training/sketch/vq_tokenizer.py` uses a different, real, verified-
   working tokenizer (`boris/vqgan_f16_16384`) — an incompatible token
   space even if the other encoder had worked.

   `sync_sketch_tier.py` resolves this the honest way: it does NOT
   consume data-forge's VQ tokens. It re-tokenizes data-forge's raw
   `images/` (linked by data-forge's export) through THIS project's own
   working `VQTokenizer`. Sync audit item #1 removed the dead
   `maskgit_vq` encoder, its config entry, and its only caller entirely
   from data-forge (see that project's `models.yaml`,
   `inference/engine.py`, `stages/s08_encoding.py`) rather than fixing
   the wrapper — nothing on this side ever depended on it, and the dead
   branch was additionally causing every `ui_first` record to fail
   data-forge's own completeness check (see
   `docs/review/01_sketch_tier.md` for the full history). This project's
   re-tokenize step is unchanged and still the source of truth.

2. **Z-Image-Turbo's precomputed latents have no consumer.** Same root
   cause as #1's fix on the data-forge side: the OFFICIAL diffusers
   training script (`train_dreambooth_lora_z_image.py`, which
   `training/polish/`'s wrapper actually calls) takes raw images via
   `--instance_data_dir` and computes its own latents internally — it has
   no documented flag for consuming externally precomputed latent files.
   `sync_polish_default.py` reads data-forge's `images/` (also newly
   linked) rather than `latents/`.

3. **DPO preference pairs are genuinely compatible, no encoder mismatch**
   — `sync_dpo_pairs.py` reads data-forge's RAW preference-pair images
   (`preference_pairs/{source}/*.json` + the `image_a`/`image_b` files
   next to them — the pre-latent-encoding stage, not
   `s08_5_dpo_encoding.py`'s `.safetensors` output, since this project has
   no DPO trainer yet that consumes Z-Image-Turbo latents directly — see
   `dpo/__init__.py`'s own "does NOT implement the actual DPO training
   loop" note). Imports them as real `PreferencePair` rows via
   `dpo/preference_store.py`, using data-forge's exact source keys
   (pickapic_v2, hpdv2, designsense_10k, designpref) as the `source`
   field — extended into `preference_store.VALID_SOURCES` for this.

4. **Planner RAG Corpus synchronization**
   — `sync_planner_rag.py` synchronizes human UICrit critiques into
   `data/planner_rag_corpus/uicrit_critiques.jsonl`.
"""

from krisna_training.data_forge_bridge import (
    sync_dpo_pairs,
    sync_planner_rag,
    sync_polish_default,
    sync_sketch_tier,
)

__all__ = [
    "sync_sketch_tier",
    "sync_polish_default",
    "sync_dpo_pairs",
    "sync_planner_rag",
]

