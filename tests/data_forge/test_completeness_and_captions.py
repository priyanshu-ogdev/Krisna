"""Tests for domain-aware encoding completeness and the manifest's
critique_output / source_caption fields — both added/fixed this revision.
"""

from __future__ import annotations

from data_forge.manifest import Manifest, ManifestRecord
from data_forge.utils.completeness import (
    is_encoding_complete,
    missing_artifacts,
    required_artifacts_for,
)


def _mk_record(domain: str, encoding_paths: dict | None) -> ManifestRecord:
    return ManifestRecord(
        id="test-id", source_dataset="test", domain=domain,
        status="encoded", encoding_paths=encoding_paths,
    )


class TestCompletenessPredicate:
    def test_ui_first_requires_three_artifacts(self):
        """qwen_image_latent dropped from the required set: Qwen-Image-
        Edit-2511 ships frozen now (no data-forge training branch for it —
        see s08_encoding.py's Branch 2 removal)."""
        assert required_artifacts_for("ui_first") == {
            "z_image_latent", "control_map", "vq_tokens",
        }

    def test_general_design_requires_two_artifacts_not_three(self):
        """The core fix: a general_design record correctly lacking
        vq_tokens must NOT be flagged incomplete — Stage 8's VQ branch is
        deliberately domain-gated to ui_first only."""
        assert required_artifacts_for("general_design") == {
            "z_image_latent", "control_map",
        }

    def test_ui_first_complete_record(self):
        rec = _mk_record("ui_first", {
            "z_image_latent": "a", "control_map": "c", "vq_tokens": "d",
        })
        assert is_encoding_complete(rec) is True

    def test_ui_first_missing_vq_tokens_is_incomplete(self):
        rec = _mk_record("ui_first", {
            "z_image_latent": "a", "control_map": "c",
        })
        assert is_encoding_complete(rec) is False
        assert missing_artifacts(rec) == {"vq_tokens"}

    def test_general_design_without_vq_tokens_is_complete(self):
        """This is the exact case that would have been a false positive
        before this fix — s09_heldout.py would have needed to flag every
        correctly-processed general_design record as broken under a
        blanket rule that required an artifact Stage 8 never produces for
        this domain."""
        rec = _mk_record("general_design", {
            "z_image_latent": "a", "control_map": "c",
        })
        assert is_encoding_complete(rec) is True
        assert missing_artifacts(rec) == set()

    def test_record_with_no_encoding_paths_is_incomplete(self):
        rec = _mk_record("ui_first", None)
        assert is_encoding_complete(rec) is False
        assert missing_artifacts(rec) == required_artifacts_for("ui_first")


class TestSourceCaptionField:
    """source_caption was captured by the fetcher but silently dropped by
    bulk_create_records prior to this fix — never reached the database at
    all despite being extracted."""

    def test_source_caption_stored_on_bulk_create(self, manifest: Manifest):
        records = [{
            "source_file": "img001.png", "image_path": "raw/test/img001.png",
            "content_hash_sha256": "abc", "image_width": 512, "image_height": 512,
            "file_size_bytes": 1000, "source_caption": "A settings screen with toggles",
        }]
        manifest.bulk_create_records(records, source_dataset="test_dataset")
        rec = manifest.query_by_status("fetched")[0]
        assert rec.source_caption == "A settings screen with toggles"

    def test_source_caption_none_when_absent(self, manifest: Manifest):
        records = [{
            "source_file": "img002.png", "image_path": "raw/test/img002.png",
            "content_hash_sha256": "def", "image_width": 512, "image_height": 512,
            "file_size_bytes": 1000,
        }]
        manifest.bulk_create_records(records, source_dataset="test_dataset")
        rec = manifest.query_by_status("fetched")[0]
        assert rec.source_caption is None


class TestCritiqueOutputField:
    def test_critique_output_round_trips(self, manifest: Manifest):
        rec = manifest.create_record(source_dataset="rico_core", source_file="1.jpg")
        critique = {
            "critique_source": "uicrit_human",
            "overall_score": 0.8,
            "visual_hierarchy_score": 0.8,
            "visual_hierarchy_note": "Clear primary action.",
        }
        manifest.update_record(rec.id, "test", critique_output=critique)
        updated = manifest.get_records_by_ids([rec.id])[0]
        assert updated.critique_output == critique

    def test_get_all_records_with_critique_filters_correctly(self, manifest: Manifest):
        rec1 = manifest.create_record(source_dataset="rico_core", source_file="1.jpg")
        rec2 = manifest.create_record(source_dataset="rico_core", source_file="2.jpg")
        manifest.update_record(rec1.id, "test", critique_output={"critique_source": "uicrit_human"})
        # rec2 gets no critique_output — should not appear in the results

        results = manifest.get_all_records_with_critique()
        result_ids = {r.id for r in results}
        assert rec1.id in result_ids
        assert rec2.id not in result_ids

    def test_query_by_dataset_ignores_status(self, manifest: Manifest):
        rec = manifest.create_record(source_dataset="rico_core", source_file="1.jpg")
        manifest.update_record(rec.id, "test", new_status="structured")

        results = manifest.query_by_dataset("rico_core")
        assert len(results) == 1
        assert results[0].status == "structured"
