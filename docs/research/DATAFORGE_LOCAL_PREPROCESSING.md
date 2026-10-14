# Data-Forge Local Preprocessing and Source-Synchronization Plan

**Status:** implemented local adapter path; governed model stages remain separate
**Date:** 2026-10-14

## Goal

Produce one canonical manifest from multiple released UI sources while
preserving source identity, app/design grouping, source revision, metadata,
structure, and image hashes. The local preprocessing path must not load a
model, call a hosted inference service, invent labels, or claim that a record
is legally or training eligible.

## Sources and evidence

### Figma2Code

- Dataset: `xcodemind/Figma2Code`
- Card: <https://huggingface.co/datasets/xcodemind/Figma2Code>
- Paper: Gui et al., ICLR 2026, <https://openreview.net/forum?id=CaXZB6bI31>
- Card-reported fields: `root`, `filekey`, `node_id`, `annotation`,
  `statistics`, `raw_metadata`, `processed_metadata`, `image_refs`, and
  `svg_assets`.
- Card-reported splits: `data_test/*.parquet` and `data_rest/*.parquet`.
- Use: high-quality community-Figma screenshot and design-structure source.
- Not assumed: a condition/target edit pair. A bitmap asset is not an
  independent screen and must remain metadata unless a source record proves it
  is the root screen.

### MobileViews

- Dataset: `mllmTeam/MobileViews`
- Card: <https://huggingface.co/datasets/mllmTeam/MobileViews>
- Paper: Gao et al., <https://arxiv.org/abs/2409.14337>
- Card-reported core fields: `image_content` and `json_content`.
- Card-reported app-trace fields: package-named app folders, state screenshots,
  view hierarchies, actions, and transitions.
- Use: mobile screenshot/view-hierarchy structure and app-distribution source.
- Rights limitation: the card says MIT metadata while describing screenshots
  captured from third-party Google Play apps. The registry therefore keeps the
  source disabled by default and does not promote it to training eligibility
  from the metadata license alone.

## Canonical record contract

The adapter writes the same record shape for both sources:

```text
source_dataset       registry namespace
source_file          source-relative artifact/shard path
source_record_id     lossless upstream node/screen/row identity
source_group_id      Figma file key or mobile app/package when known
source_revision      configured upstream revision or local receipt
canonical_id         SHA-256 of adapter + dataset + record/group/revision
source_metadata      JSON-safe upstream metadata, no embedded binary payloads
source_structure     decoded source hierarchy/metadata when available
source_caption       source-provided text when available
image_path           DATA_ROOT-confined canonical image
content_hash_sha256  hash of bytes actually written to image_path
status               fetched
```

Records from both sources can be combined because `source_dataset` remains the
registry key while `source_group_id` and `canonical_id` preserve their distinct
identity namespaces. Dataset-level split fallback remains available when a
source cannot prove an app/file group.

## Pipeline audit and decision

| Stage | Local preprocessing | Governed model pipeline | Synthetic output? |
|---|---|---|---|
| Acquire | User/downloaded local artifact directory or pinned HF snapshot | `s01_fetch` and license workflow | No |
| Decode | Bounded parquet batches; image decode; safe JSON decode | Same adapter after remote snapshot | No |
| Normalize | Canonical filename, RGB/PNG output, width/height, saved-byte hash | Same | No |
| Preserve | IDs, groups, splits, metadata, source structure | Same | No |
| PII | Not performed by `preprocess` | `s03_5_pii_scrub` | No |
| Safety | Not performed by `preprocess` | `s04_safety` and escalation | No |
| OCR/caption | Not performed by `preprocess` | `s05`/OCR stages if enabled | Model-generated only in governed mode |
| Structure | Source structure is retained, not invented | `s06_structure` can validate/extract in the governed run | No fallback invention |
| Split | No train/held-out claim is made | `s09_heldout`, group-aware and source-disjoint | No |
| Pair rendering | Not performed by `preprocess` | Future deterministic renderer only | Must be deterministic, not human-designed |
| Export | No training export | `s12_model_data_export` gates policy, audit, split, target, and renderer artifact | No fabricated pairs |

The local command intentionally stops before quality, PII, safety, OCR,
captioning, structure-model inference, encoding, split, audit, and export. This
is the safest way to combine datasets without silently assigning generated
labels or licensing claims.

## Synthetic analysis decision

For the current research paper, synthetic VLM analysis is **not required** to
make Figma2Code or MobileViews useful. Both sources provide a screenshot and
some form of structure. The better default is:

1. retain the upstream structure as provenance;
2. validate only what matches the canonical schema;
3. mark missing or non-normalizable structure as incomplete;
4. use a deterministic renderer only for records whose structure is valid;
5. exclude records without a condition image from the Qwen pair export.

Synthetic captions may be useful for a separate ablation, but they must be
stored with `synthetic_status=generated`, never represented as human labels,
and never be the only evidence for a training export. No such synthetic path is
enabled by the local preprocessor.

## Reproducibility and safety requirements

- Pin the upstream revision before a training run; mutable `main` is not a
  training receipt.
- Keep the exact local source directory and adapter mode in the preprocessing
  receipt.
- Use bounded parquet batches; do not load an entire MobileViews shard into a
  pandas frame.
- Hash canonical saved bytes, not the original embedded bytes after a decode or
  RGB conversion.
- Keep all output under `DATA_ROOT`; reject path traversal and symlink escapes
  when later stages resolve manifest paths.
- Preserve Figma `data_test` as an evaluation/source split rather than mixing
  it into training by row order.
- Use group-aware splitting for Figma file keys and MobileViews package names
  when present; fall back conservatively to dataset-level separation.
- Do not use TASTE, PartiPrompts, or any evaluation-only source as training
  input.

## Verification commands

```powershell
python -m py_compile data-forge/src/data_forge/data/source_adapters.py data-forge/src/data_forge/cli.py
pytest -q tests/data_forge/test_provenance_adapters.py tests/data_forge/test_manifest.py tests/data_forge/test_fetch_safety.py
data-forge preprocess --dataset figma2code --source-path D:\datasets\Figma2Code --limit 100
```

The first two checks are local code checks. The final command requires a local
artifact directory and produces a real preprocessing receipt; it does not
download or contact a model service.
