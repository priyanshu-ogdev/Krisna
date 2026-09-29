# 30 — Model 1/5 deep review: Planner (Qwen3.5-9B)

Systematic review of the Planner tier: every training↔inference
connection point traced and verified, PRD alignment checked, and real
research into further optimization for generalization quality (for this
tier, "generalization" means retrieval/grounding quality, since the
model itself is frozen — see PRD §6).

## What "training" means for this tier — verified consistent with PRD

PRD §6 is explicit: Planner ships frozen, grounded via RAG over real
human critique data, not fine-tuned. Confirmed this is genuinely true in
the live code, not just stated:

- `training/src/krisna_training/planner/{train.py, lora_config.py,
  masking.py, dataset.py}` exist but are deliberately deprecated —
  `planner/__init__.py` fires a real `DeprecationWarning` (verified
  earlier in this project's review), and `scripts/training/
  train_planner_lora.sh` prints a loud, accurate deprecation banner
  before running, correctly stating `inference/planner_backend.py` "no
  longer has a lora_adapter_path parameter to load it with" — checked
  directly against `PlannerBackend.__init__`: confirmed true, no such
  parameter exists anywhere in either the transformers or vLLM backend.
- `training/configs/deprecated_planner_lora.yaml` vs.
  `planner_lora_train.yaml`: diffed — comment-only differences (one
  "canonical," one "verbose," same pattern as the sketch-tier config
  pairs), no value drift. Not a bug.

## The real training↔inference connection: the RAG corpus pipeline

Traced the full chain end-to-end, not assumed:

1. **data-forge** (`s12_model_data_export.py::_export_planner_rag`)
   reads `manifest.get_all_records_with_critique()`, filters to
   `critique_source == "uicrit_human"` **only** — real human critique
   text, explicitly excluding any AI-judge-labeled critique from ever
   entering this corpus (consistent with the project-wide no-RLHF-loop
   discipline, applied here even though nothing would obviously break if
   it weren't). Writes `{record_id, image_path, critique_output}` rows
   to `model_data/planner_rag_corpus/uicrit_critiques.jsonl`.
2. **`uicrit_ingest.py::to_critique_output_dict`** — verified the exact
   dict shape this produces (`overall_score`,
   `<dimension>_score`/`<dimension>_note` flattened fields) matches
   `planner_rag.py`'s own docstring claim about the real producer's
   schema, field for field. No drift.
3. **`sync_planner_rag.py`** copies this file to a destination
   `planner_rag_corpus/uicrit_critiques.jsonl` — verified the relative
   path matches `planner_backend.py`'s `_RAG_CORPUS_RELATIVE_PATH`
   constant exactly (`"planner_rag_corpus/uicrit_critiques.jsonl"`).
4. **`planner_rag.py::UICritRAGIndex.from_jsonl`** consumes this file at
   `PlannerBackend.load()` time. Re-read the full retrieval
   implementation (TF-IDF + cosine similarity, pure stdlib) line by
   line: correctly guards the empty-corpus case (`retrieve()` returns
   `[]`, never raises), correctly guards a would-be zero-division on a
   degenerate all-zero-weight document (the `if dot == 0.0: continue`
   check necessarily fires before the division that could otherwise
   divide by a zero `_doc_norms[i]`, since a zero numerator implies the
   same zero-weight document). No bugs found.

**Verdict: this connection is clean.** No schema drift, no silent
data-loss point, no unguarded edge case, across all four stages.

## Optimization research: TF-IDF's real, well-evidenced quality ceiling

TF-IDF is purely lexical — it scores by term overlap, not meaning. This
has a concrete, predictable failure mode for this exact use case: a
design-intent query like *"make the button pop more"* shares zero
tokens with a stored critique about *"insufficient contrast on the
primary CTA"* — TF-IDF's cosine similarity for that pair is near-zero
regardless of how relevant the critique actually is, because there's no
lexical overlap for the term-frequency vectors to agree on.

Researched real, current alternatives suited to this project's specific
constraints (small corpus, short texts, and — critically — **must not
add GPU/VRAM pressure**, since PRD §7.3's per-tier budget accounting is
already tight and adding a new GPU-resident model would need its own
ledger entry and swap-orchestration wiring):

- Small sentence-embedding models (`all-MiniLM-L6-v2`,
  `EmbeddingGemma-300M`, `gte-small`) are specifically designed for
  exactly this: short-text semantic similarity, CPU-capable, ~30–300MB.
  `all-MiniLM-L6-v2` in particular is a long-established, well-tested
  choice (384-dim, ~80MB) that runs perfectly well on CPU for a corpus
  this size (hundreds to low thousands of short critique entries) —
  meaning it can replace TF-IDF's lexical scoring with genuine semantic
  scoring **without ever touching the GPU VRAM ledger at all**, since it
  never needs to be GPU-resident for a corpus this small.
- This is a materially better fit than reusing the project's existing
  CLIP text embedder (already loaded for the verifier stack): CLIP's
  text encoder is trained for image-text alignment, not text-text
  semantic similarity — a dedicated sentence-embedding model is the
  correct tool for this specific job, not just the most convenient one
  already in memory.

## Recommendation — IMPLEMENTED, see docs/review/31

The recommendation below was written before implementation; it has
since been built as an opt-in feature (`KRISNA_PLANNER_RAG_EMBEDDINGS=1`)
with real empirical evidence of the TF-IDF failure mode (not just this
section's theoretical argument) and two further bugs found on
re-review. See `docs/review/31_planner_retrieval_upgrade.md` for the
full account, including a mid-implementation design correction (the
first draft's "gracefully degrade on load failure" behavior was wrong
for this project's actual build-time-baked-in deployment model, and was
fixed to fail loudly instead, matching every other checkpoint load in
this codebase).

Original recommendation text, preserved for context:

Replace `UICritRAGIndex`'s internal TF-IDF scoring with
`all-MiniLM-L6-v2` embeddings (CPU-only, `sentence-transformers` or a
minimal raw `transformers` CPU forward pass to avoid a new heavy
dependency), keeping the exact same `retrieve(query, k) ->
list[RetrievedCritique]` public interface so `PlannerBackend`/
`PlannerBackendVLLM` need zero changes. This is a genuine,
well-evidenced retrieval-quality upgrade — but per this project's own
"verify at adoption, not announcement" discipline (PRD Appendix C.4),
it should be evaluated against this project's actual UICrit corpus
before shipping (measuring retrieval precision/recall on a held-out set
of query→relevant-critique pairs), not swapped in on the strength of
general embedding-model reputation alone. Flagged here rather than
silently implemented, matching how the vLLM Planner backend (§29) was
also shipped as an explicitly-flagged, evaluate-before-trusting opt-in
rather than a claimed-verified default change.

## Cross-cutting note: why "image generation quality" doesn't apply directly here

Planner produces no pixels — its output quality surfaces downstream, as
`constraint_updates` quality feeding into Sketch/Polish generation (see
`constraint_merge.py`, reviewed in an earlier pass). The retrieval
upgrade above is the one lever this tier has for "better generalization"
in the PRD's sense: grounding the Planner's design-domain reasoning in
retrieved real critique examples that actually match the user's intent,
not just its vocabulary.

## PRD alignment check

No PRD text describes retrieval-scoring methodology (TF-IDF vs.
embeddings) at all — this was an implementation detail below the PRD's
level of abstraction, not a documented decision to reconcile against.
No PRD update needed for what's already implemented; the embedding-based
retrieval recommendation above would be a genuinely new addition worth a
line in PRD §5.3 or the RESEARCH_AND_CITATIONS doc if/when implemented,
since it would then be a real, citable design decision like the other
model choices in that document.
