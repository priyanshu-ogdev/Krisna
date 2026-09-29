# 31 — Planner RAG retrieval upgrade: TF-IDF → optional embeddings

Follow-up to `docs/review/30_model_review_1_planner.md`'s recommendation.
Records what was actually implemented, a correction made mid-implementation
based on real deployment feedback, and what's still genuinely unverified.

## The empirical case (not just theory)

Ran the real `UICritRAGIndex` code (not a hypothetical) against a
realistic corpus and realistic casual user queries. One result stood
out as a clean, unambiguous failure: the query *"the form feels too
tight, give it some breathing room"* should retrieve a critique about
cramped, poorly-spaced form fields — instead that document ranked
**last**, beaten by two unrelated documents, because TF-IDF scores
lexical overlap and "tight"/"breathing room" share no tokens with
"cramped"/"spacing". This is a real, reproducible failure on this
project's own retrieval code and corpus schema, confirmed with the exact
test harness already in `tests/inference/test_planner_rag.py`
(`test_query_with_no_overlapping_terms_returns_empty` already encodes
this exact limitation class as expected TF-IDF behavior).

## What was implemented

`UICritRAGIndex` now accepts an optional `embed_fn` — when provided,
retrieval uses cosine similarity over dense embeddings instead of
TF-IDF; when absent (unchanged default), behavior is byte-identical to
before. Gated behind `KRISNA_PLANNER_RAG_EMBEDDINGS=1`, resolved through
one shared function (`resolve_embed_fn_from_env`) used identically by
both `planner_backend.py` and `planner_backend_vllm.py`, so the two
backends cannot disagree with each other about retrieval behavior.

Recommended model: `sentence-transformers/all-MiniLM-L6-v2` — CPU-only
(~80MB), chosen specifically because the UICrit corpus this indexes is
far too small (~1,000 documents) to justify GPU residency or a new
VRAMLedger entry (§7.2). Added to `requirements-inference.txt`, verified
via a real `pip install --dry-run` to resolve cleanly alongside the
existing `transformers>=5.2.0`/`torch>=2.6.0` pins.

## Correction made mid-implementation — a real design mistake caught by feedback

The first draft of this feature was built around a "gracefully degrade
to TF-IDF if the embedding model fails to load" pattern — reasoning that
a first-run network download could fail. This was WRONG for this
project's actual deployment model, corrected after direct feedback: this
project's inference containers build model weights into the image at
build time — the same as every other tier's checkpoint (Qwen3.5,
Z-Image-Turbo, Gemma-4) — not lazily on first request. So a load failure
here, when the feature is explicitly enabled, means a real configuration
bug (missing requirement, broken build), not an expected runtime
condition. Silently falling back to a degraded retrieval mode would hide
that bug with no signal anything was wrong — the opposite of this
project's own established discipline (the safety gate must run and must
be visible if it doesn't, the VQ-token handoff must not silently no-op,
etc.). Fixed: `load_default_embedder()` and `resolve_embed_fn_from_env()`
now raise on failure, exactly like every other checkpoint load in this
codebase, wrapped into the same `BackendLoadError` other tiers use. The
one thing that's still handled gracefully (and correctly so, a different
failure class): a single malformed document's text crashing the encoder
DURING index build, after the model itself loaded fine — that's a
per-row data-quality issue, not a missing-model configuration bug, and
is skipped-and-logged rather than taking down the whole index, matching
data-forge's own established skip-and-count pattern for malformed rows
elsewhere in this project.

## On the test suite's "fake embedder" — clarified, not just renamed

A fair question was raised about what the deterministic stand-in
function used in tests actually proves. Answer, stated precisely rather
than glossed over: it proves the retrieval MECHANISM is correctly wired
(index building over dense vectors, cosine similarity math, per-call
error handling for a single bad document/query) — it does NOT prove the
real `all-MiniLM-L6-v2` model retrieves better results than TF-IDF on
the real UICrit corpus. Renamed from `_fake_embedder` to
`_stand_in_embedder` and the test class from `TestEmbeddingRetrievalMode`
to `TestEmbeddingRetrievalMechanism`, with an explicit docstring stating
both what the tests do and do not prove. This is a standard testing
technique (substituting a cheap, deterministic function for an expensive
external dependency to test the code that calls it) — legitimate for
verifying the surrounding logic, not a substitute for evaluating the
real model.

## What remains genuinely unverified

The real `all-MiniLM-L6-v2` model could not be downloaded in this
project's own development sandbox specifically — that environment's
network allowlist covers `pypi.org`/`github.com` but not
`huggingface.co`. This is a constraint of the development/review
environment, not the real deployment (where the model is built into the
image and has full registry access at build time). Before trusting this
in production: run the same before/after comparison technique used to
find the TF-IDF failure above — real queries against the real UICrit
corpus, inspecting top-k results by hand — in the real deployment
environment, where the model is genuinely present. This is flagged, not
assumed, per PRD Appendix C.4's "verify at adoption, not announcement"
discipline.

## Test coverage

19 tests in `tests/inference/test_planner_rag.py` (up from 9), covering:
TF-IDF mode unchanged (existing 9), embedding-mode index/retrieval
mechanics via the stand-in embedder, per-document/per-query embedding
failure resilience during build/retrieve, empty-query symmetry between
both modes, and the corrected fail-loudly-on-load-failure behavior for
`load_default_embedder()`/`resolve_embed_fn_from_env()`. Plus 2 new
tests in `tests/inference/test_planner_load_cleanup.py`. Full inference
suite (205 tests) passes.

## Two more real bugs found on a careful re-review pass

Went back through `planner_rag.py`/`planner_backend.py` a second time
after finishing the above, rather than treating "tests pass" as the end
of the review. Found two more genuine, if smaller, issues:

1. **Empty-query asymmetry between the two retrieval modes.**
   `_retrieve_tfidf` already short-circuits on an empty/untokenizable
   query (`if not q_tokens: return []`), but `_retrieve_embeddings` had
   no equivalent check — it would call the embedding model on an empty
   string and return whatever essentially-arbitrary top-k the model
   happened to produce. This is a genuinely reachable path, not just a
   defensive edge case: `PlannerBackend.run()`'s `message` parameter
   defaults to `""`. Fixed by mirroring the same short-circuit in
   `_retrieve_embeddings`, with a regression test asserting the
   embedding model is never even called for an empty or whitespace-only
   query.

2. **A real GPU-memory leak on a specific load() failure path.** Traced
   `swap_orchestrator.py`'s `_load_one` to check what happens when a
   backend's `load()` raises, and found it only rolls back its OWN
   ledger bookkeeping (`self.ledger.release(tier)`) — it never calls the
   backend's `unload()`. This means every backend is individually
   responsible for cleaning up any partial state it accumulated before
   re-raising (confirmed this is already the established, deliberate
   pattern: `critic_backend.py`'s own load() failure paths already call
   `await self._kill()` before raising, for exactly this reason).
   `PlannerBackend.load()`'s RAG-embedding step is a SECOND
   failure-prone step, downstream of the model/tokenizer already
   loading successfully onto the GPU — and it was missing this cleanup:
   a RAG-embedding load failure would leak the already-resident model
   while the ledger incorrectly believed that budget was free again.
   `planner_backend_vllm.py`'s equivalent failure path already had this
   right (`await self._kill()`, added in the prior implementation pass)
   — only the transformers-based backend was missing it. Fixed by
   calling `self.unload()` before raising, with two new regression
   tests in `tests/inference/test_planner_load_cleanup.py` (one
   exercising the real `PlannerBackend.load()` lifecycle with only
   `AutoModelForCausalLM.from_pretrained`/`AutoTokenizer.from_pretrained`
   mocked, confirming `self._model`/`self._tokenizer` are actually
   cleared after the induced failure; one confirming the ordinary,
   embeddings-disabled path is unaffected).

Neither of these was caught by the first round of tests, because the
first round tested the RAG module in isolation — these needed tracing
the actual call chain between `swap_orchestrator.py` and the backend's
full `load()` lifecycle to find. Worth noting as a general lesson for
the remaining four models still to review: testing a module in
isolation is necessary but not sufficient — the cross-file connection
points are where this class of bug tends to hide.
