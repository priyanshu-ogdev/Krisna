# 33 — Model 2/5 data↔training connection review: Sketch tier

Follow-up to `docs/review/32_model_review_2_sketch.md`, which covered
the Planner→Sketch inference-side connection. This pass reviews the
OTHER connection: data-forge → training, specifically for the Sketch
tier — and found one genuinely significant, previously-missed
optimization opportunity, plus fixed an unrelated pre-existing test gap
encountered while validating.

## The finding: one `.npy` file per image was a real I/O bottleneck at scale

Traced the full data path again from scratch: data-forge's
`s12_model_data_export.py::_export_sketch_tier` exports raw images (not
pre-tokenized — data-forge's own VQ-encoding stage was removed
entirely, per that method's own comment, pushing tokenization to
sync-time instead). Both real consumers of those raw images —
`sync_sketch_tier.py` (the data-forge bridge) and `prepare_dataset.py`
(the manual/standalone path) — then tokenized each image through the
real VQGAN tokenizer and wrote **one `.npy` file per image**
(`tokens/{i:07d}.npy`).

This is correctly a ONE-TIME cost (tokenization happens once at sync
time, not re-run every epoch — confirmed `SketchTokenDataset` reads
pre-computed tokens, never raw images). But the FILE LAYOUT itself was
the problem: at PRD §8.3's target corpus scale (100K–500K images),
that's hundreds of thousands of individual small files (1–4KB each,
depending on 16×16 vs. 32×32 grid). Every epoch, every DataLoader worker
process independently pays a real `open()`/`read()`/`close()` syscall
per sample — a well-known, well-documented I/O bottleneck at this scale
(small-file overhead dominates over actual bytes transferred), worse on
network-mounted storage and with multiple parallel workers hitting the
filesystem simultaneously. This is exactly the kind of inefficiency
tools like WebDataset, LMDB-backed datasets, and HuggingFace `datasets`'
Arrow-backed storage exist specifically to avoid — and this project was
using none of those techniques.

## The fix: consolidated, memory-mapped shards

New shared module `training/sketch/token_shard_writer.py`
(`TokenShardWriter`), used identically by both producers:

- Buffers up to `DEFAULT_SHARD_SIZE=10,000` token grids in memory, then
  flushes one consolidated `.npy` array (`tokens/shard_NNNNN.npy`, shape
  `(shard_size, grid_h*grid_w)`, dtype int32) — sharded rather than one
  array for the whole corpus, so a single write failure doesn't require
  re-tokenizing everything, and so shard files stay a manageable size
  (a full 10K-image shard at the 32×32/Stage-2 grid is ≈41MB).
- Manifest records now carry `{"tokens_shard": "tokens/shard_00003.npy",
  "tokens_index": 42, ...}` instead of `{"tokens_path":
  "tokens/0000042.npy", ...}`.
- `SketchTokenDataset` opens each distinct shard file via
  `np.load(path, mmap_mode="r")` **once** in `__init__` (lazily, only
  for shards actually referenced by that manifest — e.g. a heldout
  split carved from a subset), not once per `__getitem__` call.
  Memory-mapped arrays are safe and standard practice to share read-only
  across forked DataLoader worker processes — each worker gets its own
  OS-level mapping to the same file via the page cache, with pages
  faulted in lazily on access rather than the whole shard being loaded
  into RAM upfront.
- **Full backward compatibility preserved**: the dataset still reads
  the old `tokens_path` (one-file-per-image) and inline `tokens` formats
  for any existing manifest — auto-detected per record, so nothing that
  already ran against the old format breaks.

Verified with a direct test that specifically proves the fix does what
it claims (`test_shard_opened_once_not_per_getitem_call`): patches
`np.load` to count calls, reads 5 samples from the same shard, and
asserts exactly **one** `np.load` call happened, not five. Also added a
full end-to-end integration test
(`TestEndToEndSyncSketchTierWithShardWriter::test_sync_then_read_round_trip`)
exercising `sync_sketch_tier.py`'s real `sync()` function through a fake
tokenizer and reading the result back through `SketchTokenDataset`, not
just the writer and reader tested in isolation.

## Updated the two existing tests that asserted the old format explicitly

`test_sync_writes_manifest_with_correct_captions`,
`test_sync_writes_real_token_files_from_tokenizer`, and
`test_prepare_dataset_with_json_dict` all explicitly asserted
`"tokens_path" in record` — correctly caught by the format change, not a
regression. Updated to assert the new `tokens_shard`/`tokens_index`
fields instead. Also fixed a stale docstring in `sync_sketch_tier.py`
that still described the old format as what gets written.

## Unrelated fix encountered along the way: a pre-existing test-fixture gap

While running the full training suite to validate the above, found 3
pre-existing failures in `test_dpo_dataset_integration.py` — unrelated
to this pass's Sketch-tier work, left over from an earlier session
(`flip_prob` was added to `PreferencePairDataset.__init__` in a later
review pass, but three tests construct the dataset via
`object.__new__()` to bypass `__init__`'s real SQLite/blob-store
requirements, and were never updated to also set `flip_prob` manually).
Fixed by adding `ds.flip_prob = 0.0` to all three construction sites,
matching `__init__`'s own default. Also confirmed two other failure
batches encountered during validation (`torchvision` and
`pyarrow`/`jsonschema`/`ruamel.yaml` missing) were pure sandbox package
gaps, not real bugs — installed each and confirmed the underlying code
was already correct.

## What was checked and found NOT to need changing

- **Tokenization timing**: confirmed this is correctly a one-time,
  sync-time cost, not re-run per epoch or per step — the actual
  bottleneck was file layout, not redundant computation.
- **Caption handling, `caption_mix_ratio`**: already reviewed and fixed
  in an earlier pass (docs/review/31 area) — re-confirmed still correct
  and unaffected by this change (the shard format only changes how
  *tokens* are stored/read; caption fields are untouched, still inline
  in the JSONL manifest).
- **VQ tokenizer itself, masking schedule, loss computation**: reviewed
  earlier in this project's history, no issues found, not re-litigated
  here.

## Test coverage

7 new tests in `tests/training/test_token_shard_writer.py` (writer
mechanics + full end-to-end round trip), 4 new tests in
`tests/training/test_sketch_dataset.py`'s `TestConsolidatedShardFormat`
(reader mechanics, including the "opened once" proof), 3 existing tests
updated to match the new format, 3 pre-existing unrelated failures
fixed. Full training suite (178 tests) and data-forge suite (143 tests)
both pass cleanly after installing the sandbox's missing optional
packages (`torchvision`, `pyarrow`, `jsonschema`, `ruamel.yaml`) to get
a true signal separate from environment gaps.
