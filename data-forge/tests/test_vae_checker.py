"""Tests for the VAE config checker."""

import json
import pytest
from pathlib import Path
from data_forge.agents.vae_checker import check_vae_config, VAEConfigError


class TestVAEChecker:
    def test_valid_config(self, sample_vae_config: Path):
        result = check_vae_config(sample_vae_config, expected_channels=64, expected_downsample=16)
        assert result["verified"] is True
        assert result["channels"] == 64

    def test_wrong_channels(self, tmp_path: Path):
        config_dir = tmp_path / "vae"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"latent_channels": 32}))

        with pytest.raises(VAEConfigError, match="channel count mismatch"):
            check_vae_config(config_dir, expected_channels=64)

    def test_missing_config(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(VAEConfigError, match="No VAE config found"):
            check_vae_config(empty_dir)

"""Tests for the VAE config checker."""

import json
from pathlib import Path

import pytest

from data_forge.agents.vae_checker import VAEConfigError, check_vae_config


class TestVAEChecker:
    def test_valid_config(self, sample_vae_config: Path):
        result = check_vae_config(sample_vae_config, expected_channels=64, expected_downsample=16)
        assert result["verified"] is True
        assert result["channels"] == 64

    def test_wrong_channels(self, tmp_path: Path):
        config_dir = tmp_path / "vae"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"latent_channels": 32}))

        with pytest.raises(VAEConfigError, match="channel count mismatch"):
            check_vae_config(config_dir, expected_channels=64)

    def test_missing_config(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(VAEConfigError, match="No VAE config found"):
            check_vae_config(empty_dir)
