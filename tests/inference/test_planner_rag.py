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
