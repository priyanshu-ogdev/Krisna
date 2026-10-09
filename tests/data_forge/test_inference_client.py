from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import httpx
import pytest

from data_forge.inference.client import InferenceClient


def test_image_encoding_reads_mutated_file_contents(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"before redaction")
    _, before = InferenceClient._encode_image(image_path)

    image_path.write_bytes(b"after redaction")
    _, after = InferenceClient._encode_image(image_path)

    assert base64.b64decode(before) == b"before redaction"
    assert base64.b64decode(after) == b"after redaction"


@pytest.mark.asyncio
async def test_complete_batch_bounds_workers_and_preserves_input_order() -> None:
    client = InferenceClient(httpx.AsyncClient(), "test-model", max_concurrent=3)
    active = 0
    peak_active = 0

    async def fake_complete(
        prompt, image_path=None, schema=None, max_tokens=2048, temperature=0.1
    ):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.001)
        active -= 1
        return {"prompt": prompt}

    client.complete = fake_complete
    try:
        results = await client.complete_batch(
            items=[{"value": str(index)} for index in range(17)],
            prompt_template="{value}",
        )
    finally:
        await client._client.aclose()

    assert peak_active == 3
    assert results == [{"prompt": str(index)} for index in range(17)]
