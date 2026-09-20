"""Tests for the storage manager."""


import pytest

from data_forge.config import PipelineConfig, StorageConfig
from data_forge.data.storage import StorageManager, StorageQuotaExceeded


class TestStorageManager:
    def test_preflight_passes(self, config: PipelineConfig):
        config.storage = StorageConfig(
            safety_margin=0.10,
            per_record_estimates={"raw_image_bytes": 1000, "metadata_bytes": 100},
        )
        sm = StorageManager(config)
        report = sm.pre_flight_check(100)  # 100 records * 1100 bytes = tiny
        assert report["pass"] is True

    def test_preflight_fails_on_huge(self, config: PipelineConfig):
        config.storage = StorageConfig(
            safety_margin=0.10,
            per_record_estimates={"raw_image_bytes": 10_000_000_000},  # 10GB per record
        )
        sm = StorageManager(config)
        with pytest.raises(StorageQuotaExceeded):
            sm.pre_flight_check(1_000_000)  # Absurd amount

    def test_mid_flight_check(self, config: PipelineConfig):
        sm = StorageManager(config)
        report = sm.mid_flight_check()
        assert "free_gb" in report
        assert "used_pct" in report

    def test_calculate_projected(self, config: PipelineConfig):
        """Uses real per-image estimate keys, not arbitrary placeholders —
        calculate_projected_size() only sums keys it recognizes as
        per-image or per-preference-pair (see StorageManager's
        _PER_IMAGE_ESTIMATE_KEYS / _PER_PREFERENCE_PAIR_ESTIMATE_KEYS), a
        deliberate split added so preference-pair storage (materially
        larger per item: two source images + two latents) can't get
        multiplied against the image-record count instead of its own.
        """
        config.storage = StorageConfig(
            per_record_estimates={"raw_image_bytes": 1000, "metadata_bytes": 2000},
        )
        sm = StorageManager(config)
        assert sm.calculate_projected_size(10) == 30000

    def test_calculate_projected_preference_pairs_use_their_own_count(self, config: PipelineConfig):
        config.storage = StorageConfig(
            per_record_estimates={
                "raw_image_bytes": 1000,          # per-image estimate — ignored for the pair count
                "preference_pair_source_bytes": 500,
                "preference_pair_latent_bytes": 250,
            },
        )
        sm = StorageManager(config)
        # 10 image records * 1000 + 5 preference pairs * 750 (500+250)
        assert sm.calculate_projected_size(10, preference_pair_count=5) == 10_000 + 3_750
