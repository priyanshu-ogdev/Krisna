"""Bridges data-forge's `model_data/` export (see that project's
`s12_model_data_export.py`) into the exact input formats this project's
own training packages expect (`training/sketch/dataset.py`,
`training/polish/dataset_prep.py`, `dpo/preference_store.py`).

Two real format mismatches were found while building this, not papered
over:

1. **Sketch tier's VQ tokenizer identity doesn't match.** data-forge's
   `sketch_tier_maskgit/vq_tokens/*.pt` files come from `maskgit_vq`
   (TencentARC/Open-MAGVIT2) — but that encoder's own loading code in
   data-forge's `engine.py` raises `RuntimeError` on purpose, because
   Open-MAGVIT2 needs a custom `.encode()`/`.decode()` wrapper that
   hasn't been built and verified there yet (see that file's own comment:
   "AutoModel.from_pretrained(..., trust_remote_code=True) is not
   confirmed to work against this repo"). Meanwhile, THIS project's
   `training/sketch/vq_tokenizer.py` uses a different, real, verified-
   working tokenizer (`boris/vqgan_f16_16384`). The two are incompatible
   token spaces — you cannot decode Open-MAGVIT2 tokens with the
   boris/vqgan_f16_16384 decoder or vice versa.

   `sync_sketch_tier.py` resolves this the honest way: it does NOT
   consume data-forge's `.pt` VQ tokens at all. It re-tokenizes
   data-forge's raw `images/` (linked by data-forge's export — see the
   data-forge-side fix that added this) through THIS project's own
   working `VQTokenizer`. This is real, extra encode time you wouldn't
   need if both sides agreed on one tokenizer — flagged here rather than
   silently accepted, since "why are we re-tokenizing images data-forge
   already tokenized" is a fair question whose answer is "because the
   other tokenizer doesn't actually work yet."

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
"""
