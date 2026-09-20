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

Retrieval strategy: plain TF-IDF + cosine similarity, stdlib only. This is
a small corpus (UICrit is ~1,000 real human-annotated screens, not a
web-scale collection) — a heavier embedding-model retriever is a
reasonable future upgrade, but pulling in a whole new sentence-transformer
dependency for ~1K short documents is disproportionate to the corpus size,
and stdlib TF-IDF is a real, correct, well-understood technique rather
than an approximation of one.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("krisna_inference.backends.planner_rag")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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
    (see PlannerBackend.__init__'s rag_corpus_dir param), and building a
    TF-IDF index is real (if small) work that shouldn't happen as an
    import side-effect.
    """

    _entries: list[dict] = field(default_factory=list)
    _doc_tokens: list[list[str]] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)  # document frequency per term
    _idf: dict[str, float] = field(default_factory=dict)
    _doc_tfidf: list[dict[str, float]] = field(default_factory=list)
    _doc_norms: list[float] = field(default_factory=list)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "UICritRAGIndex":
        path = Path(path)
        idx = cls()
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
        log.info("uicrit_rag_index_built", extra={"documents": len(idx._entries)})
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
        tokenizes to nothing) — callers must treat retrieval as optional
        context, never a hard requirement (a frozen model with zero
        retrieved context should still degrade to a reasonable generic
        system prompt, not fail outright).
        """
        if not self._entries:
            return []

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

        results = []
        for cos_sim, i in scored[:k]:
            entry = self._entries[i]
            co = entry.get("critique_output") or {}
            results.append(RetrievedCritique(
                record_id=entry.get("record_id", ""),
                image_path=entry.get("image_path", ""),
                note=self._note_text(entry),
                overall_score=co.get("overall_score", 0.5),
                score=cos_sim,
            ))
        return results

    def __len__(self) -> int:
        return len(self._entries)
