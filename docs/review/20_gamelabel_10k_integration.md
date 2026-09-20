# Phase 20 — GameLabel-10K Integration

Follow-up to `DATA_SOURCES.md`'s "vetted but not yet integrated" entry.
Confirmed the dataset is real via live search (`Jonathan-Zhou/GameLabel-10k`,
Apache 2.0, arXiv:2409.19830 — real crowdsourced mobile-game player
votes, not AI-judge/synthetic labeling), then confirmed the **actual**
column schema directly against the live dataset's own HF dataset-viewer
output rather than guessing from the paper's description:
`prompt`, `img0_votes`/`img1_votes` (int, 0-5 — a vote *count* from up
to 5 players per pair, not a fixed 0/1 label column like Pick-a-Pic's
format), and `img0_encoding`/`img1_encoding` (plain strings — a
base64-encoded JPEG wrapped in a stray Python `b'...'` bytes-repr,
confirmed from real sample values, not the standard HF `{"bytes": ...}`
image-feature dict the generic `preference_pair` fetch mode detects).

Neither existing fetch mode (`preference_pair`, `hpdv2_ranked_list`)
fits this shape — reusing either would have found zero usable pairs
silently. Also found HF auto-generates a parquet mirror
(`refs/convert/parquet`) for this CSV-formatted repo, letting the new
adapter reuse the existing parquet-reading infrastructure rather than
streaming the raw 2.26GB `data.csv` directly.

## What was built

- `_decode_gamelabel_image()` — strips the `b'...'`/`b"..."` wrapper,
  base64-decodes the real JPEG bytes.
- `_fetch_gamelabel_csv()` — reads the parquet mirror, derives a
  win/lose label from vote counts (dropping ties, including 0-0 — same
  convention `_fetch_huggingface_preference_pairs`/
  `_fetch_hpdv2_ranked_pairs` already use), preserves the vote margin
  per pair for potential downstream weighting, applies the same
  exact-duplicate SHA256 guard the other fetchers use, and writes pairs
  in the identical output format so `s01_6_preference_pairs.py`'s
  dedup/PII/safety pipeline picks them up with zero changes needed
  there.
- `datasets.yaml`'s `gamelabel_10k` entry, category
  `dpo_preference_general` — supplements Pick-a-Pic/HPDv2's Stage-1
  general-domain DPO signal, does not close the Stage-2 UI-domain gap
  (`designsense_10k`/`designpref` remain unreleased for that, per the
  earlier review).

## Two real bugs caught before shipping, both by running the code

1. `base64.b64decode()` is lenient by default — garbage input didn't
   raise, it silently produced corrupted bytes rather than a clean
   failure signal. Fixed with `validate=True`, then a follow-up test
   caught that `validate=True` *still* lets an empty string decode
   successfully to `b''`, so added an explicit empty-check before the
   decode attempt.
2. `import base64` was scoped locally inside `_fetch_gamelabel_csv`,
   invisible to the separate module-level `_decode_gamelabel_image`
   function that also needs it — the same class of `NameError` caught
   in `flows.py` during Phase 19's work. Moved to a proper module-level
   import.

## Verification

`pyarrow`/`huggingface_hub` unavailable in this environment, so real
`pandas`/`PIL` were used with just the I/O boundary
(`snapshot_download`, `pd.read_parquet`) mocked, exercising the actual
row-processing logic end to end: confirmed a tie is correctly dropped,
a row with genuinely undecodable image data is skipped without crashing
the whole fetch, and two rows with distinct images produce correctly
labeled pairs with correct vote margins and valid, openable PNG files.
Caught a flaw in the *test* itself mid-verification — two test rows
initially reused identical image bytes, triggering the pipeline's real
exact-duplicate dedup guard (correct behavior) and producing a
misleading "missing pair" result; fixed by using genuinely distinct
images per test row, not by changing the code. Formalized as
`tests/data_forge/test_fetch_gamelabel.py`, matching the sibling
`test_fetch_hf_parquet_images.py`'s established patching conventions,
including a schema-drift guard test that fails loudly rather than
silently misparsing if the live dataset's columns ever change.
