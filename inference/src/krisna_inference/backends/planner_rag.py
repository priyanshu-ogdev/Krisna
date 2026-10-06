"""Retrieval index over data-forge's `model_data/planner_rag_corpus/
uicrit_critiques.jsonl` — the Planner's real, frozen-model replacement for
the LoRA fine-tune this project's earlier (v10 PRD) revision used.

Schema note (verified directly against data-forge's real producer —
data_forge/data/uicrit_ingest.py::to_critique_output_dict — not assumed):
each line is `{"record_id": str, "image_path": str, "critique_output":
{...}}`, where critique_output has `critique_source`, `overall_score`,
four `<dimension>_score`/`<dimension>_note` pairs (visual_hierarchy,
readability, layout_consistency, brand_alignment — UICrit's real rubric
doesn't map 1:1 onto these four, so all four `_note` fields are
duplicates of the same truncated `critique_text`, not independent
per-dimension commentary), `suggested_edits`, and `raw_fields` (the
original UICrit row, preserved in case the normalization above needs
revisiting).

Retrieval strategy: TF-IDF + cosine similarity by default, with an
OPTIONAL embedding-based mode (see `embed_fn` below) — UPGRADED this
review pass (docs/review/30_model_review_1_planner.md /
31_planner_retrieval_upgrade.md) after finding a real, empirical failure
mode, not a theoretical one: a realistic casual query ("the form feels
too tight, give it some breathing room") against this corpus's real
schema ranked the actually-relevant document (about cramped, poorly-
spaced form fields) BELOW two unrelated ones, because TF-IDF scores
lexical overlap and "tight"/"breathing room" share no tokens with
"cramped"/"spacing". Confirmed with the exact test harness in
tests/inference/test_planner_rag.py. TF-IDF is purely lexical by
construction — this isn't a bug in the implementation, it's the
technique's inherent ceiling.

The fix is ADDITIVE, not a replacement: `UICritRAGIndex` accepts an
optional `embed_fn: Callable[[str], Sequence[float]] | None`. When
provided, retrieval uses cosine similarity over dense embeddings
instead of TF-IDF. When absent (the default — nothing changes for any
existing caller), behavior is byte-identical to before this pass.

MODEL AVAILABILITY: this project's inference containers build model
weights into the image at build time (the same as every other tier's
checkpoint — Qwen3.5, Z-Image-Turbo, Gemma-4, etc.), not lazily on first
request. So unlike a public web service pulling weights over the
network at runtime, `sentence-transformers/all-MiniLM-L6-v2` being
unavailable when `KRISNA_PLANNER_RAG_EMBEDDINGS=1` is set is a real
configuration bug (bad build, missing requirement), not an expected
degraded-mode trigger — `load_default_embedder()` therefore does NOT
silently fall back to TF-IDF on failure. It raises, exactly like a
missing/broken checkpoint would for any other tier, so the failure is
visible at load() time instead of silently producing worse retrieval
with no signal that anything is wrong.

HONEST LIMITATION, stated plainly rather than glossed over: the real
`all-MiniLM-L6-v2` model could not be downloaded and evaluated end-to-end
in THIS project's own development sandbox specifically, because that
environment's network allowlist covers pypi.org/github.com but not
huggingface.co — a sandbox-specific restriction, not a statement about
the real deployment target. What IS verified here: the retrieval
MECHANISM (index building over dense vectors, cosine similarity, error
propagation) has real test coverage using a deterministic stand-in
function in place of the real model — a standard testing technique
(dependency injection) for verifying logic that surrounds an expensive
external model, NOT a substitute for evaluating the real model's actual
output quality. What is NOT verified by this project: the real model's
actual retrieval quality on the real UICrit corpus. Per this project's
own "verify at adoption, not announcement" discipline (PRD Appendix
C.4), run the SAME before/after comparison technique used to find the
TF-IDF failure above — real queries against the real corpus, inspecting
top-k results by hand — in the real deployment environment, where the
model is genuinely present, before trusting this in production.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("krisna_inference.backends.planner_rag")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def load_default_embedder() -> Callable[[str], Sequence[float]]:
    """Builds a CPU-only sentence-embedding function using
    `all-MiniLM-L6-v2` (~80MB, 384-dim — small enough to never need GPU
    residency or a VRAMLedger entry for a corpus this size).

    Does NOT catch import/load failures — model weights are built into
    this project's inference image at build time (same as every other
    tier's checkpoint), so a failure here means a real configuration bug
    (missing requirement, broken build), not an expected runtime
    condition to degrade around. Raises, and the caller (load()) wraps it
    into the same BackendLoadError every other checkpoint-load failure
    uses, so it's visible immediately rather than silently producing
    worse retrieval with no signal anything is wrong.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _embed(text: str) -> Sequence[float]:
        return model.encode(text, show_progress_bar=False).tolist()

    return _embed


def resolve_embed_fn_from_env() -> Callable[[str], Sequence[float]] | None:
    """Single shared entry point both planner_backend.py and
    planner_backend_vllm.py call, so the opt-in env-var check can't drift
    between the two backends — same "one implementation, not two that
    could disagree" discipline as _generate_raw()'s extraction (see
    planner_backend_vllm.py's module docstring). Callers should run this
    inside asyncio.to_thread — model loading is real, blocking work, even
    though it's a local, already-downloaded checkpoint (not a network
    call) in the real deployment.

    Returns None only when the feature is simply not enabled
    (KRISNA_PLANNER_RAG_EMBEDDINGS unset) — that is the one legitimate
    "nothing to do here" case. If the feature IS enabled,
    load_default_embedder()'s exception (if any) propagates up
    unmodified; this function does not swallow it."""
    import os

    if os.environ.get("KRISNA_PLANNER_RAG_EMBEDDINGS", "0") != "1":
        return None
    return load_default_embedder()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


@dataclass
class RetrievedCritique:
    record_id: str
    image_path: str
    note: str
    overall_score: float
    score: float  # retrieval relevance score, not overall_score — deliberately named
    # differently so callers can't accidentally conflate "how relevant is
    # this to the query" with "how good was the rated design"


@dataclass
class UICritRAGIndex:
    """Build once at PlannerBackend.load() time, query per-turn in run().

    Deliberately NOT built at import time or module scope — the corpus
    path is only known once a real data_root/rag_corpus_dir is configured
    (see PlannerBackend.__init__'s rag_corpus_dir param), and building an
    index is real (if small) work that shouldn't happen as an import
    side-effect.
    """

    embed_fn: Callable[[str], Sequence[float]] | None = None
    _entries: list[dict] = field(default_factory=list)
    _doc_tokens: list[list[str]] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)  # document frequency per term
    _idf: dict[str, float] = field(default_factory=dict)
    _doc_tfidf: list[dict[str, float]] = field(default_factory=list)
    _doc_norms: list[float] = field(default_factory=list)
    _doc_embeddings: list[Sequence[float]] = field(default_factory=list)

    @classmethod
    def from_jsonl(
        cls, path: str | Path, embed_fn: Callable[[str], Sequence[float]] | None = None
    ) -> "UICritRAGIndex":
        path = Path(path)
        idx = cls(embed_fn=embed_fn)
        if not path.exists():
            log.warning(
                "uicrit_rag_corpus_missing",
                extra={
                    "path": str(path),
                    "note": "Planner will run without retrieval context — run "
                             "data-forge's s01_5_uicrit_join + s12_model_data_export "
                             "stages to produce this file.",
                },
            )
            return idx

        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                idx._entries.append(entry)

        idx._build()
        log.info(
            "uicrit_rag_index_built",
            extra={"documents": len(idx._entries),
                   "mode": "embeddings" if embed_fn else "tfidf"},
        )
        return idx

    def _note_text(self, entry: dict) -> str:
        co = entry.get("critique_output") or {}
        # All four *_note fields are the same truncated critique_text per
        # data-forge's real normalization (see module docstring) — read
        # just one rather than concatenating four identical copies, which
        # would silently quadruple that term's TF-IDF weight for no
        # reason.
        return co.get("visual_hierarchy_note") or co.get("readability_note") or ""

    def _build(self) -> None:
        if self.embed_fn is not None:
            for entry in self._entries:
                note = self._note_text(entry)
                try:
                    self._doc_embeddings.append(self.embed_fn(note) if note else [])
                except Exception as e:
                    # A single document's embedding failing must not take
                    # down the whole index — log and treat it as
                    # unretrievable (empty vector never wins a cosine
                    # comparison against anything with real magnitude).
                    log.warning("planner_rag_doc_embed_failed", extra={"error": str(e)})
                    self._doc_embeddings.append([])
            return

        for entry in self._entries:
            tokens = _tokenize(self._note_text(entry))
            self._doc_tokens.append(tokens)
            for term in set(tokens):
                self._df[term] += 1

        n_docs = max(len(self._entries), 1)
        self._idf = {
            term: math.log((n_docs + 1) / (df + 1)) + 1.0  # smoothed idf, always positive
            for term, df in self._df.items()
        }

        for tokens in self._doc_tokens:
            tf = Counter(tokens)
            weights = {term: count * self._idf.get(term, 0.0) for term, count in tf.items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            self._doc_tfidf.append(weights)
            self._doc_norms.append(norm)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedCritique]:
        """Top-k critiques by cosine similarity to `query`. Empty list if
        the index has no documents (missing corpus, or a query that
        tokenizes to nothing under TF-IDF mode) — callers must treat
        retrieval as optional context, never a hard requirement (a frozen
        model with zero retrieved context should still degrade to a
        reasonable generic system prompt, not fail outright).
        """
        if not self._entries:
            return []

        if self.embed_fn is not None:
            return self._retrieve_embeddings(query, k)
        return self._retrieve_tfidf(query, k)

    def _retrieve_embeddings(self, query: str, k: int) -> list[RetrievedCritique]:
        # Mirrors _retrieve_tfidf's empty-query short-circuit — message
        # defaults to "" in PlannerBackend.run(), a genuinely reachable
        # path, not just a defensive check. Without this, an empty query
        # would still call the embedding model (wastefully, though
        # harmlessly for a well-behaved model) and likely return
        # essentially-arbitrary top-k results, rather than the same
        # "nothing to retrieve" signal TF-IDF mode gives for the same input.
        if not query.strip():
            return []
        try:
            q_vec = self.embed_fn(query)
        except Exception as e:
            log.warning("planner_rag_query_embed_failed", extra={"error": str(e)})
            return []
        if not q_vec:
            return []

        scored: list[tuple[float, int]] = []
        for i, doc_vec in enumerate(self._doc_embeddings):
            if not doc_vec:
                continue
            scored.append((_cosine(q_vec, doc_vec), i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._to_result(i, score) for score, i in scored[:k]]

    def _retrieve_tfidf(self, query: str, k: int) -> list[RetrievedCritique]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        q_weights = {term: count * self._idf.get(term, 0.0) for term, count in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0

        scored: list[tuple[float, int]] = []
        for i, doc_weights in enumerate(self._doc_tfidf):
            # Sparse dot product over the smaller of the two term sets.
            shared_terms = q_weights.keys() & doc_weights.keys()
            dot = sum(q_weights[t] * doc_weights[t] for t in shared_terms)
            if dot == 0.0:
                continue
            cos_sim = dot / (q_norm * self._doc_norms[i])
            scored.append((cos_sim, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._to_result(i, score) for score, i in scored[:k]]

    def _to_result(self, i: int, score: float) -> RetrievedCritique:
        entry = self._entries[i]
        co = entry.get("critique_output") or {}
        return RetrievedCritique(
            record_id=entry.get("record_id", ""),
            image_path=entry.get("image_path", ""),
            note=self._note_text(entry),
            overall_score=co.get("overall_score", 0.5),
            score=score,
        )

    def __len__(self) -> int:
        return len(self._entries)
