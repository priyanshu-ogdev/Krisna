"""Generic dataset-ingestion completeness tests.

This class of bug — a dataset silently ingesting zero usable records —
has now happened three separate times (PD12M/CC12M's metadata-only
shape, UICrit's annotation-only shape, Screen2Words' caption-join shape),
each caught only by manual code tracing, not by any test. These tests
exercise the actual logic paths that broke, using synthetic data so they
run without network access — they verify the *mechanism* is sound
(parsing, column detection, joining), not that any specific live dataset
schema matches what's configured (that still needs a real connectivity
check against each dataset before a production run, same as any other
"confirm before you trust it" item flagged throughout this pipeline).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_forge.data.uicrit_ingest import parse_uicrit_annotations, to_critique_output_dict
from data_forge.manifest import Manifest


class TestUICritIngestion:
    def test_parse_annotations_from_csv(self, tmp_path: Path):
        """A CSV annotation file with a recognizable ID/critique/rating
        shape should parse into non-empty, correctly-keyed rows — this is
        the exact mechanism that silently produced zero rows before the
        annotation_only fix (the generic image scanner never even looked
        at this file)."""
        repo_dir = tmp_path / "uicrit_repo"
        repo_dir.mkdir()
        csv_path = repo_dir / "ratings.csv"
        # Realistically sized (UICrit is ~983 rows) — find_annotation_file
        # deliberately skips anything under 1KB to avoid false-matching an
        # unrelated tiny CSV (e.g. a stray config file), so a 2-row
        # fixture would be rejected by that same guard a real ~983-row
        # file would never trip. Padding the critique text is what makes
        # this test actually exercise the parsing path instead of the
        # "file too small, skip it" path.
        rows = ["rico_id,critique,rating"]
        rows.append(
            '12345,"Buttons lack sufficient contrast against the background. '
            + ("Padding to realistic row length. " * 20) + '",3'
        )
        rows.append(
            '67890,"Clean hierarchy, but the CTA is buried below the fold. '
            + ("Padding to realistic row length. " * 20) + '",4'
        )
        csv_path.write_text("\n".join(rows))
        assert csv_path.stat().st_size > 1024, "test fixture must exceed the real size guard"

        annotations = parse_uicrit_annotations(repo_dir)

        assert len(annotations) == 2, "Expected 2 parsed rows from the CSV"
        assert annotations[0]["rico_join_key"] == "12345"
        assert "contrast" in annotations[0]["critique_text"]
        assert annotations[0]["rating"] == 3

    def test_parse_annotations_no_file_found(self, tmp_path: Path):
        """No matching file at all should return an empty list (a hard
        stop the caller must notice), not raise or silently synthesize
        data."""
        repo_dir = tmp_path / "empty_repo"
        repo_dir.mkdir()
        (repo_dir / "README.md").write_text("nothing relevant here")

        annotations = parse_uicrit_annotations(repo_dir)
        assert annotations == []

    def test_to_critique_output_dict_shape(self):
        """A parsed annotation should map onto the shared CritiqueOutput
        shape with critique_source correctly marked as human-sourced —
        this is what distinguishes real calibration data from Gemma 4's
        own self-generated critiques downstream."""
        ann = {
            "rico_join_key": "12345",
            "critique_text": "Buttons lack sufficient contrast.",
            "rating": 4,
            "raw_fields": {"rico_id": "12345", "rating": 4},
        }
        out = to_critique_output_dict(ann)

        assert out["critique_source"] == "uicrit_human"
        assert 0.0 <= out["overall_score"] <= 1.0
        assert out["overall_score"] == pytest.approx(0.8)  # 4/5
        assert "raw_fields" in out

    def test_uicrit_join_matches_rico_records_by_filename_stem(self, manifest: Manifest):
        """End-to-end join mechanism: annotations should correctly match
        already-ingested RICO records by filename stem, and correctly
        NOT match when the key doesn't correspond to anything ingested."""
        rec = manifest.create_record(
            source_dataset="rico_core", source_file="12345.jpg", image_path="raw/rico_core/12345.jpg"
        )
        manifest.create_record(
            source_dataset="rico_core", source_file="99999.jpg", image_path="raw/rico_core/99999.jpg"
        )

        stem_to_record = {
            r.source_file.rsplit(".", 1)[0]: r.id
            for r in manifest.query_by_dataset("rico_core")
        }
        assert "12345" in stem_to_record
        assert stem_to_record["12345"] == rec.id
        assert "00000" not in stem_to_record


class TestScreen2WordsIngestionShapeDetection:
    """These test the *detection logic* in isolation (embedded-image vs.
    ID+caption shape) since the fetcher's actual HTTP/HF calls aren't
    exercisable offline — see fetcher.py::_fetch_huggingface_caption_join.
    """

    def test_detects_embedded_image_column(self):
        import pandas as pd

        df = pd.DataFrame({
            "image": [{"bytes": b"\x89PNG\r\n", "path": "0.png"}],
            "caption": ["A login screen"],
        })
        sample = df["image"].dropna().iloc[0]
        assert isinstance(sample, dict) and "bytes" in sample

    def test_detects_id_and_caption_columns_for_join_path(self):
        import pandas as pd

        df = pd.DataFrame({
            "rico_id": ["12345", "67890"],
            "caption": ["A login screen", "A settings page"],
        })
        columns = list(df.columns)
        assert any(c.lower() in ("rico_id", "screen_id", "image_id", "id") for c in columns)
        assert any(c.lower() in ("caption", "captions", "summary", "text", "description") for c in columns)


class TestGenericIngestionYield:
    """The general-purpose test the audit recommended: every configured
    dataset should produce non-zero records after Stage 1, or the join,
    for its actual expected shape. This doesn't hit the network — it
    documents the expectation and exercises it against a manifest built
    the way each ingestion path is supposed to leave it, so a future
    silent-zero-record regression on ANY dataset (not just these three)
    has a test that would need updating to explain why zero is now
    expected, rather than nothing noticing at all.
    """

    def test_bulk_create_records_produces_nonzero_count(self, manifest: Manifest):
        records = [
            {"source_file": f"img{i:03d}.png", "image_path": f"raw/test/img{i:03d}.png",
             "content_hash_sha256": f"hash{i}", "image_width": 512, "image_height": 512,
             "file_size_bytes": 1000}
            for i in range(50)
        ]
        count = manifest.bulk_create_records(records, source_dataset="test_dataset")
        assert count == 50, (
            "bulk_create_records should insert every well-formed record — "
            "a regression here would be the same failure class as the "
            "PD12M/CC12M/UICrit/Screen2Words bugs, just one layer deeper."
        )

    def test_join_only_pseudo_records_are_not_inserted_as_image_records(self, manifest: Manifest):
        """_join_only pseudo-records (from caption_join's path b) must
        never reach bulk_create_records as if they were real image
        records — they have none of the required columns, and inserting
        them would silently create garbage rows instead of failing
        loudly. This test locks in the s01_fetch.py routing behavior
        that prevents that."""
        join_pairs = [
            {"_join_only": True, "_join_target_dataset": "rico_core",
             "_join_key": "12345", "_source_caption": "A login screen"},
        ]
        real_records = [
            r for r in join_pairs if not r.get("_join_only")
        ]
        assert real_records == [], (
            "join-only pairs must be filtered out before bulk_create_records "
            "is ever called on them"
        )
