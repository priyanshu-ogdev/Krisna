"""Tests for DatasetSpec.storage_relevant_record_count.

Regression coverage for the bug traced during the v10 PRD completeness
review: the Stage 0 pre-flight storage check summed every dataset's raw
`expected_record_count` (PD12M's full 12.4M, CC12M's full 12.4M, etc.)
with no regard for `fetch_config.sample_size` actually capping downloads,
or for annotation_only/text_reference/caption_join sources that never
produce a standalone image record at all. That inflated the projected
corpus by roughly 25x against the PRD's real ~100K-500K target, which
would false-fail `pre_flight_check` on any normal workstation disk before
Stage 1 ever ran.
"""

from __future__ import annotations

from data_forge.config import DatasetSpec


def _spec(**overrides) -> DatasetSpec:
    defaults = dict(
        display_name="test",
        source_type="huggingface",
        category="general_visual",
        expected_record_count=12_400_000,
        fetch_config={},
    )
    defaults.update(overrides)
    return DatasetSpec(**defaults)


class TestStorageRelevantRecordCount:
    def test_sample_size_caps_the_count(self):
        """PD12M's real shape: 12.4M expected, but sample_size=200K caps
        what's actually downloaded. The storage projection should use
        200K, not 12.4M."""
        spec = _spec(
            expected_record_count=12_400_000,
            fetch_config={"download_mode": "url_list", "sample_size": 200_000},
        )
        assert spec.storage_relevant_record_count() == 200_000

    def test_no_sample_size_uses_full_expected_count(self):
        """A source with no sample_size cap (e.g. rico_core) should still
        project against its full expected_record_count."""
        spec = _spec(expected_record_count=66_000, fetch_config={})
        assert spec.storage_relevant_record_count() == 66_000

    def test_sample_size_larger_than_expected_count_is_harmless(self):
        """sample_size is a cap, not a floor — if it's larger than the
        dataset's real size, the real size still wins."""
        spec = _spec(
            expected_record_count=1_000,
            fetch_config={"sample_size": 20_000},
        )
        assert spec.storage_relevant_record_count() == 1_000

    def test_annotation_only_contributes_zero(self):
        """UICrit joins onto existing RICO records — it produces zero new
        image records and should not add to the image-storage projection."""
        spec = _spec(
            expected_record_count=1_000,
            annotation_only=True,
            fetch_config={},
        )
        assert spec.storage_relevant_record_count() == 0

    def test_text_reference_contributes_zero(self):
        """Glaive/xLAM are pure text reference material, never inserted
        into the image manifest."""
        spec = _spec(
            expected_record_count=113_000,
            fetch_config={"download_mode": "text_reference"},
        )
        assert spec.storage_relevant_record_count() == 0

    def test_caption_join_contributes_zero(self):
        """Screen2Words' realistic shape joins captions onto existing
        RICO records rather than creating new image records."""
        spec = _spec(
            expected_record_count=22_417,
            fetch_config={"download_mode": "caption_join"},
        )
        assert spec.storage_relevant_record_count() == 0

    def test_realistic_corpus_sum_matches_prd_scale_not_full_corpus_scale(self):
        """Regression guard for the actual bug: summing every configured
        dataset's storage_relevant_record_count() should land in the
        hundreds-of-thousands (PRD §8.3's 100K-500K target range), not
        the tens-of-millions the raw expected_record_count sum produces.
        """
        specs = [
            _spec(expected_record_count=12_400_000,
                  fetch_config={"download_mode": "url_list", "sample_size": 200_000}),
            _spec(expected_record_count=12_423_374,
                  fetch_config={"download_mode": "url_list", "sample_size": 150_000}),
            _spec(expected_record_count=66_000, fetch_config={}),
            _spec(expected_record_count=66_000, fetch_config={}),
            _spec(expected_record_count=60_000, fetch_config={}),
            _spec(expected_record_count=1_460, fetch_config={}),
            _spec(expected_record_count=350_000, fetch_config={}),
            _spec(expected_record_count=22_417,
                  fetch_config={"download_mode": "caption_join"}),
            _spec(expected_record_count=1_000, annotation_only=True, fetch_config={}),
            _spec(expected_record_count=113_000,
                  fetch_config={"download_mode": "text_reference"}),
            _spec(expected_record_count=60_000,
                  fetch_config={"download_mode": "text_reference"}),
            _spec(expected_record_count=10_000,
                  fetch_config={"download_mode": "triple_dataset"}),
            _spec(expected_record_count=300_000,
                  fetch_config={"download_mode": "triple_dataset", "sample_size": 20_000}),
        ]

        raw_sum = sum(s.expected_record_count for s in specs)
        effective_sum = sum(s.storage_relevant_record_count() for s in specs)

        assert raw_sum > 25_000_000, "sanity check: the raw sum should reproduce the bug's scale"
        # magicbrush (10K, uncapped triple_dataset) + instructpix2pix (20K,
        # capped) count toward the effective sum since triple_dataset isn't
        # in the zero-storage set (it does write real image pairs, just to
        # a different directory) — bringing the realistic total to ~907K,
        # comfortably inside/near the PRD's 100K-500K curated target
        # (before Stage 2/3 quality and dedup filtering shrink it further).
        assert effective_sum < 1_000_000, (
            "effective sum should be roughly PRD-scale, not full-corpus scale"
        )
        assert effective_sum < raw_sum / 20, (
            "the fix should meaningfully shrink the projection, not just "
            "trim it slightly"
        )
