from __future__ import annotations

import json

from krisna_inference.backends.planner_rag import UICritRAGIndex


def _write_corpus(path, entries):
    """entries: list of (record_id, image_path, note_text, overall_score).

    Mirrors data-forge's REAL exported shape exactly (verified directly
    against data_forge/data/uicrit_ingest.py::to_critique_output_dict, not
    assumed) — critique_output.visual_hierarchy_note/readability_note are
    identical duplicates of the same truncated critique_text, not
    independent per-dimension text.
    """
    lines = []
    for record_id, image_path, note, score in entries:
        lines.append(json.dumps({
            "record_id": record_id,
            "image_path": image_path,
            "critique_output": {
                "critique_source": "uicrit_human",
                "overall_score": score,
                "visual_hierarchy_score": score, "visual_hierarchy_note": note,
                "readability_score": score, "readability_note": note,
                "layout_consistency_score": score, "layout_consistency_note": note,
                "brand_alignment_score": score, "brand_alignment_note": note,
                "suggested_edits": [],
                "raw_fields": {},
            },
        }))
    path.write_text("\n".join(lines))


class TestUICritRAGIndex:
    def test_missing_corpus_file_degrades_to_empty_index(self, tmp_path):
        idx = UICritRAGIndex.from_jsonl(tmp_path / "does_not_exist.jsonl")
        assert len(idx) == 0
        assert idx.retrieve("anything") == []

    def test_retrieves_most_relevant_document_first(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [
            ("rec1", "img1.png", "The signup form has poor contrast between text and background", 0.4),
            ("rec2", "img2.png", "This settings screen uses a clean modern navigation pattern", 0.9),
            ("rec3", "img3.png", "Login button placement is inconsistent with platform conventions", 0.5),
        ])
        idx = UICritRAGIndex.from_jsonl(corpus)
        assert len(idx) == 3

        results = idx.retrieve("designing a signup form with contrast issues", k=2)
        assert len(results) > 0
        assert results[0].record_id == "rec1"

    def test_retrieve_respects_k(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [
            (f"rec{i}", f"img{i}.png", "a login screen with a signup button", 0.5)
            for i in range(10)
        ])
        idx = UICritRAGIndex.from_jsonl(corpus)
        results = idx.retrieve("login signup screen", k=3)
        assert len(results) == 3

    def test_query_with_no_overlapping_terms_returns_empty(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [("rec1", "img1.png", "onboarding flow uses skeuomorphic icons", 0.6)])
        idx = UICritRAGIndex.from_jsonl(corpus)
        results = idx.retrieve("zzz qqq xyzabc", k=3)
        assert results == []

    def test_empty_query_returns_empty(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [("rec1", "img1.png", "a settings panel", 0.5)])
        idx = UICritRAGIndex.from_jsonl(corpus)
        assert idx.retrieve("", k=3) == []
        assert idx.retrieve("   ", k=3) == []

    def test_retrieved_critique_exposes_note_and_overall_score(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [("rec1", "img1.png", "dashboard widget spacing is inconsistent", 0.7)])
        idx = UICritRAGIndex.from_jsonl(corpus)
        results = idx.retrieve("dashboard widget spacing", k=1)
        assert results[0].note == "dashboard widget spacing is inconsistent"
        assert results[0].overall_score == 0.7
        assert results[0].image_path == "img1.png"

    def test_reads_only_one_of_the_four_duplicate_note_fields(self, tmp_path):
        """Regression guard for the specific schema quirk documented in
        planner_rag.py: all four *_note fields are identical duplicates.
        Reading more than one would silently quadruple that content's
        TF-IDF term weight relative to a hypothetical corpus entry with
        genuinely distinct per-dimension notes."""
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [("rec1", "img1.png", "consistent spacing throughout", 0.8)])
        idx = UICritRAGIndex.from_jsonl(corpus)
        results = idx.retrieve("consistent spacing", k=1)
        # If all four fields were concatenated, this would contain the
        # phrase four times; assert it appears in the note exactly once.
        assert results[0].note.count("consistent spacing") == 1

    def test_malformed_jsonl_lines_are_skipped_not_fatal(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "not valid json at all\n"
            + json.dumps({
                "record_id": "rec1", "image_path": "img1.png",
                "critique_output": {"visual_hierarchy_note": "a valid entry"},
            })
        )
        idx = UICritRAGIndex.from_jsonl(corpus)
        assert len(idx) == 1

    def test_blank_lines_skipped(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "\n\n" + json.dumps({
                "record_id": "rec1", "image_path": "img1.png",
                "critique_output": {"visual_hierarchy_note": "some note"},
            }) + "\n\n"
        )
        idx = UICritRAGIndex.from_jsonl(corpus)
        assert len(idx) == 1


class TestEmbeddingRetrievalMechanism:
    """Tests the embedding-based retrieval MECHANISM (index building over
    dense vectors, cosine similarity math, per-call error handling) using
    a deterministic stand-in function in place of the real
    all-MiniLM-L6-v2 model.

    WHAT THIS DOES prove: `UICritRAGIndex`'s embedding code path is
    correctly wired — it builds dense-vector document representations,
    scores queries by cosine similarity, and handles a malformed
    document/query without crashing the whole index.

    WHAT THIS DOES NOT prove: that the real `all-MiniLM-L6-v2` model
    actually retrieves better results than TF-IDF on the real UICrit
    corpus. That model could not be downloaded in this project's own
    development sandbox specifically (its network allowlist covers
    pypi.org/github.com, not huggingface.co) — a constraint of this
    development environment, not a statement about the real deployment,
    where the model is built into the image. This is a standard
    testing technique (substituting a cheap, deterministic stand-in for
    an expensive external dependency to test the code that CALLS it),
    not a claim about real-world retrieval quality — see
    planner_rag.py's module docstring for the honest statement of what
    still needs validating in the real deployment before trusting this
    in production.
    """

    @staticmethod
    def _stand_in_embedder(text: str):
        """A deterministic, hand-built function standing in for the real
        model in these tests — NOT a claim that this represents how
        all-MiniLM-L6-v2 actually embeds text, only a tool for proving
        the surrounding index/retrieval code correctly uses WHATEVER
        embed_fn it's given. Maps a small vocabulary of related UI-critique
        terms onto shared dimensions, so a query and a document using
        different surface words for the same concept get non-trivial
        cosine similarity — the property TF-IDF structurally lacks and
        that this test suite needs to exercise to prove the embedding
        code path (not TF-IDF) is what's actually running."""
        vocab = {
            "tight": [1.0, 0.0, 0.0], "cramped": [0.9, 0.1, 0.0],
            "spacing": [0.8, 0.2, 0.0], "breathing": [0.85, 0.15, 0.0],
            "room": [0.0, 0.0, 0.0],
            "contrast": [0.0, 1.0, 0.0], "button": [0.0, 0.9, 0.1],
            "modern": [0.0, 0.0, 1.0], "premium": [0.0, 0.1, 0.9],
        }
        tokens = text.lower().split()
        vecs = [vocab.get(t, [0.0, 0.0, 0.0]) for t in tokens]
        if not vecs:
            return [0.0, 0.0, 0.0]
        return [sum(v[i] for v in vecs) for i in range(3)]

    def test_embedding_mode_finds_synonym_match_tfidf_would_miss(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [
            ("rec_contrast", "img1.png", "contrast button issue", 0.4),
            ("rec_spacing", "img2.png", "cramped spacing problem", 0.5),
            ("rec_modern", "img3.png", "modern premium feel", 0.6),
        ])
        idx = UICritRAGIndex.from_jsonl(corpus, embed_fn=self._stand_in_embedder)
        # "tight" and "breathing room" never appear in any document —
        # under TF-IDF this would score zero everywhere. Under the fake
        # embedder, "tight"/"breathing" share dimension 0 with
        # "cramped"/"spacing", so rec_spacing should win.
        results = idx.retrieve("tight breathing room", k=1)
        assert len(results) == 1
        assert results[0].record_id == "rec_spacing"

    def test_embedding_mode_used_when_embed_fn_provided(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [("rec1", "img1.png", "contrast button", 0.5)])
        idx = UICritRAGIndex.from_jsonl(corpus, embed_fn=self._stand_in_embedder)
        assert idx.embed_fn is not None
        assert idx._doc_embeddings  # embeddings computed, not TF-IDF weights
        assert not idx._doc_tfidf

    def test_tfidf_mode_used_when_embed_fn_is_none(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [("rec1", "img1.png", "contrast button", 0.5)])
        idx = UICritRAGIndex.from_jsonl(corpus)  # no embed_fn — default, unchanged behavior
        assert idx.embed_fn is None
        assert idx._doc_tfidf
        assert not idx._doc_embeddings

    def test_doc_embedding_failure_does_not_crash_index_build(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [
            ("rec_ok", "img1.png", "a fine note", 0.5),
            ("rec_bad", "img2.png", "BOOM", 0.5),
        ])

        def flaky_embedder(text):
            if text == "BOOM":
                raise RuntimeError("simulated embedding failure")
            return self._stand_in_embedder(text)

        idx = UICritRAGIndex.from_jsonl(corpus, embed_fn=flaky_embedder)
        assert len(idx) == 2  # both entries still indexed
        results = idx.retrieve("a fine note", k=2)
        assert any(r.record_id == "rec_ok" for r in results)
        assert not any(r.record_id == "rec_bad" for r in results)  # empty vector never matches

    def test_query_embedding_failure_returns_empty_not_raises(self, tmp_path):
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [("rec1", "img1.png", "a note", 0.5)])

        def always_fails(text):
            raise RuntimeError("simulated failure")

        idx = UICritRAGIndex.from_jsonl(corpus, embed_fn=self._stand_in_embedder)
        idx.embed_fn = always_fails  # simulate the model breaking after index build
        assert idx.retrieve("anything", k=1) == []

    def test_empty_query_short_circuits_in_embedding_mode_too(self, tmp_path):
        """message defaults to "" in PlannerBackend.run() — a genuinely
        reachable path, not just a defensive edge case. Must behave the
        same as TF-IDF mode's own empty-query short-circuit
        (test_empty_query_returns_empty above), not waste a model call
        on empty input and return arbitrary top-k results."""
        corpus = tmp_path / "corpus.jsonl"
        _write_corpus(corpus, [("rec1", "img1.png", "contrast button", 0.5)])

        calls = []

        def counting_embedder(text):
            calls.append(text)
            return self._stand_in_embedder(text)

        idx = UICritRAGIndex.from_jsonl(corpus, embed_fn=counting_embedder)
        calls.clear()  # discard the index-build-time call for the document itself

        assert idx.retrieve("", k=1) == []
        assert idx.retrieve("   ", k=1) == []  # whitespace-only, same as TF-IDF's .strip() via _tokenize
        assert calls == []  # the model must never be called for either


class TestResolveEmbedFnFromEnv:
    def test_disabled_by_default(self, monkeypatch):
        from krisna_inference.backends.planner_rag import resolve_embed_fn_from_env

        monkeypatch.delenv("KRISNA_PLANNER_RAG_EMBEDDINGS", raising=False)
        assert resolve_embed_fn_from_env() is None

    def test_disabled_for_any_non_1_value(self, monkeypatch):
        from krisna_inference.backends.planner_rag import resolve_embed_fn_from_env

        monkeypatch.setenv("KRISNA_PLANNER_RAG_EMBEDDINGS", "true")
        assert resolve_embed_fn_from_env() is None

    def test_load_default_embedder_raises_loudly_when_package_missing(self, monkeypatch):
        """UPDATED behavior: model weights are built into this project's
        image at build time, so a missing/broken sentence-transformers
        install when the feature is explicitly enabled is a real
        configuration bug — it must raise, matching every other
        checkpoint-load failure in this codebase (see
        planner_backend.py's load() wrapping this into BackendLoadError),
        not silently degrade to a worse retrieval mode with no signal
        anything is wrong."""
        import builtins

        import pytest

        from krisna_inference.backends.planner_rag import load_default_embedder

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("simulated: package not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError):
            load_default_embedder()

    def test_resolve_embed_fn_from_env_propagates_load_failure(self, monkeypatch):
        """resolve_embed_fn_from_env() must not swallow
        load_default_embedder()'s exception when the feature is enabled —
        only the "feature not enabled" case returns None."""
        import pytest

        from krisna_inference.backends import planner_rag

        monkeypatch.setenv("KRISNA_PLANNER_RAG_EMBEDDINGS", "1")

        def raises(*args, **kwargs):
            raise RuntimeError("simulated load failure")

        monkeypatch.setattr(planner_rag, "load_default_embedder", raises)
        with pytest.raises(RuntimeError, match="simulated load failure"):
            planner_rag.resolve_embed_fn_from_env()
