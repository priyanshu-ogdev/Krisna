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
        config.storage = StorageConfig(
            per_record_estimates={"a": 1000, "b": 2000},
        )
        sm = StorageManager(config)
        assert sm.calculate_projected_size(10) == 30000
