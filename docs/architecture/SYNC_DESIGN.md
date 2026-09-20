# data-forge ↔ training Sync Design

This is the verified contract between `data-forge`'s export stage
(`data_forge.stages.s12_model_data_export.ModelDataExportStage`) and
`krisna_training.data_forge_bridge`'s three sync scripts. Every field and
path below was checked against real, running code on both sides in the
same session this doc was written — not just cross-referenced as text.

## Directory contract

`data-forge` writes to `<DATA_ROOT>/model_data/`:

```
model_data/
├── sketch_tier_maskgit/
│   ├── images/                 <- scrubbed images, linked
│   ├── vq_tokens/               <- data-forge's own VQ tokens (NOT consumed —
│   │                                see "Why re-tokenize" below)
│   ├── captions.jsonl           <- {record_id, caption, image_filename}
│   └── manifest_summary.json
├── polish_zimage_turbo/
│   ├── images/
│   ├── latents/                 <- data-forge's own Z-Image latents (NOT
│   │                                consumed — official training script
│   │                                computes its own)
│   ├── captions.jsonl           <- same shape as above
│   └── manifest_summary.json
├── dpo_alignment/
│   ├── general/{pickapic_v2,hpdv2}/
│   └── domain/{designsense_10k,designpref}/
├── planner_rag_corpus/
│   └── uicrit_critiques.jsonl   <- {record_id, image_path, critique_output}
└── eval_external/
```

`data-forge` also writes to `<DATA_ROOT>/preference_pairs/<source>/` —
this is what `sync_dpo_pairs.py` actually reads (NOT `dpo_alignment/`,
which is data-forge's own already-encoded-latent export for its own
Diffusion-DPO training; the sync bridge re-derives everything from the
raw pair JSON + images instead, for the same "re-encode rather than trust
a possibly-mismatched pre-encoding" reason as the sketch/polish sync
below).

## Field 1: `image_filename` — the actual join key

**The bug this fixes, and why it matters**: `record_id` in `captions.jsonl`
is a random `uuid4` (see `data_forge.manifest.Manifest.create_record`),
completely unrelated to the linked image's on-disk filename (named after
the original fetch-time file, e.g. `rico_core_0000042.png`). An earlier
revision of both `sync_sketch_tier.py` and `sync_polish_default.py`
assumed `image_filename.stem == record_id` — true of neither in a real
export. That produced **empty captions for 100% of synced records**, with
no error anywhere, because the lookup just silently missed.

Fixed on both sides: data-forge's `captions.jsonl` entries now carry the
real linked filename directly (`image_filename`), and both sync scripts
join on that field instead of guessing. Verified end-to-end: a real
`ModelDataExportStage.run()` call producing a record with `record_id =
"e397461e-f3d3-42b9-bd0d-b39ba90addd2"` and `image_filename =
"rico_core_0000042.png"` was fed directly into real `sync_sketch_tier.py`/
`sync_polish_default.py` code, and the caption came through correctly.

Both sync scripts still handle a pre-fix (`image_filename`-less)
data-forge export gracefully — falls back to the old stem-matching
behavior with a loud warning, rather than crashing.

## Field 2: preference-pair metadata

`preference_pairs/<source>/<pair_id>.json`:
```json
{
  "pair_id": "pickapic_v2_0000001",
  "prompt": "a red button",
  "image_a": "pickapic_v2_0000001_a.png",
  "image_b": "pickapic_v2_0000001_b.png",
  "preferred": "a",
  "origin": "pickapic_v2",
  "label_source": "human",
  "dedup_status": "unique",
  "pii_scrubbed": false,
  "safety_tier": "safe"
}
```

`sync_dpo_pairs.py` only imports pairs with `dedup_status == "unique"` —
matches `s01_6_preference_pairs.py`'s guarantee that this field is only
set after dedup + face-blur + NSFW safety classification all pass.
`origin` maps directly to `krisna_training.dpo.preference_store.
VALID_SOURCES` — verified to contain the exact same four keys
(`pickapic_v2`, `hpdv2`, `designsense_10k`, `designpref`) data-forge's
`datasets.yaml` uses for its `preference_pair`/`hpdv2_ranked_list`
download modes.

## Field 3: RAG corpus

`planner_rag_corpus/uicrit_critiques.jsonl` — `{record_id, image_path,
critique_output}`, where `critique_output` has `critique_source`,
`overall_score`, four `<dimension>_score`/`<dimension>_note` pairs (all
four `_note` fields are identical duplicates of the same truncated
`critique_text` — UICrit's real rubric doesn't map 1:1 onto them),
`suggested_edits`, `raw_fields`. `krisna_inference.backends.planner_rag.
UICritRAGIndex` reads exactly this shape — verified directly against
`data_forge.data.uicrit_ingest.to_critique_output_dict`, the actual
function that produces it.

## Why re-tokenize/re-encode instead of trusting data-forge's precomputed artifacts

Two different, real reasons, one per tier:

1. **Sketch tier**: data-forge's VQ tokenizer (Open-MAGVIT2) has an
   unverified `.encode()`/`.decode()` wrapper — its own `engine.py` raises
   `RuntimeError` on purpose rather than silently loading a checkpoint
   that hasn't been confirmed to actually work. `krisna_training.sketch`
   uses a different, real, working tokenizer (`boris/vqgan_f16_16384`).
   Rather than pretend these are interchangeable, `sync_sketch_tier.py`
   re-tokenizes the raw images through the working tokenizer.
2. **Polish default (Z-Image-Turbo)**: the *official* diffusers training
   script (`train_dreambooth_lora_z_image.py`) takes raw images via
   `--instance_data_dir` and computes its own latents internally — it has
   no documented flag for consuming externally precomputed latents. So
   data-forge's `latents/` folder has no real consumer; `images/` does.

Both of these are the reason `data-forge`'s `s12_model_data_export.py`
links `images/` alongside its own already-processed artifacts, not just
the processed artifacts alone.
