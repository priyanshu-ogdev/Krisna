from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from data_forge.config import EncoderSpec, PipelineConfig, StageConfig
from data_forge.data.storage import StorageManager
from data_forge.stages.s08_encoding import EncodingStage


class _Manifest:
    def __init__(self, records):
        self.records = records
        self.updates = []

    def get_records_by_ids(self, record_ids):
        by_id = {record.id: record for record in self.records}
        return [by_id[record_id] for record_id in record_ids if record_id in by_id]

    def bulk_update_records(self, updates, stage):
        self.updates.extend(updates)


class _Vae:
    def __init__(self):
        self.batch_sizes = []

    def encode(self, batch):
        self.batch_sizes.append(batch.shape[0])
        latents = torch.zeros((batch.shape[0], 4, 8, 8), dtype=batch.dtype)
        return SimpleNamespace(
            latent_dist=SimpleNamespace(sample=lambda: latents)
        )


class _Engine:
    def __init__(self, vae):
        self.vae = vae

    def get_encoder(self, name):
        assert name == "z_image_vae"
        return self.vae


@pytest.mark.asyncio
async def test_encoding_pipeline_batches_and_invalidates_stale_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(StorageManager, "mid_flight_check", lambda self: {"safe": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = PipelineConfig()
    config.data_root = tmp_path
    config.encoders = {
        "z_image_vae": EncoderSpec(model_id="test-vae", revision="test-revision")
    }
    config.stages["s08_encoding"] = StageConfig(
        enabled=True,
        params={"vae_batch_size": 2, "max_encoding_resolution": 256},
    )

    records = []
    image_paths = []
    for index in range(3):
        image_path = tmp_path / f"input-{index}.png"
        Image.new("RGB", (64, 64), (index * 20, 40, 80)).save(image_path)
        image_paths.append(image_path)
        records.append(
            SimpleNamespace(
                id=f"record-{index}",
                status="routed",
                scrubbed_image_path=image_path.name,
                structure_output={"layout_type": "test"},
            )
        )

    manifest = _Manifest(records)
    vae = _Vae()
    stage = EncodingStage()

    first = await stage.run(
        manifest, config, [record.id for record in records], _Engine(vae)
    )
    assert first.records_processed == len(records)
    assert sorted(vae.batch_sizes) == [1, 2]
    assert all(
        {"z_image_latent", "control_map"}.issubset(update["encoding_paths"])
        for update in manifest.updates
        if update.get("new_status") == "encoded"
    )

    calls_after_first_run = len(vae.batch_sizes)
    second = await stage.run(
        manifest, config, [record.id for record in records], _Engine(vae)
    )
    assert second.records_processed == len(records)
    assert len(vae.batch_sizes) == calls_after_first_run

    Image.new("RGB", (64, 64), (255, 0, 0)).save(image_paths[0])
    os_timestamp_ns = 2_000_000_000_000_000_000
    os.utime(image_paths[0], ns=(os_timestamp_ns, os_timestamp_ns))
    await stage.run(manifest, config, [records[0].id], _Engine(vae))
    assert len(vae.batch_sizes) == calls_after_first_run + 1
