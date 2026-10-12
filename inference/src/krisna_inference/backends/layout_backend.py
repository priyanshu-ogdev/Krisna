"""Structured Sketch backend baseline.

This backend is deterministic until a LayoutFlow-trained checkpoint is
available. It establishes the production IR/rendering contract without
pretending that a layout model has already been trained.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from krisna_inference.layout_baseline import layout_from_constraints
from krisna_inference.layout_renderer import render_wireframe
from krisna_inference.backends.blob_store_singleton import get_blob_store
from krisna_inference.orchestrator.model_registry import ModelBackend


class LayoutBackend(ModelBackend):
    def __init__(self, spec, output_dir: str | None = None):
        super().__init__(spec)
        self.output_dir = Path(output_dir or os.environ.get("KRISNA_LAYOUT_OUTPUT_DIR", "models/sketch/layout_runs"))

    async def load(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def run(self, planner_output: dict | None = None, **kwargs):
        if not self._loaded:
            raise RuntimeError("Layout backend not loaded")
        planner_output = planner_output or {}
        delta = planner_output.get("design_state_delta") or {}
        constraints = dict(kwargs.get("constraints") or {})
        constraints.update(delta.get("constraint_updates") or {})
        layout = layout_from_constraints(constraints, text=kwargs.get("message", ""))
        session_id = str(kwargs.get("session_id") or "anonymous")
        turn_id = str(kwargs.get("turn_id") or kwargs.get("message") or "layout")
        stem = "layout_" + hashlib.sha256(f"{session_id}:{turn_id}".encode()).hexdigest()[:16]
        json_path = self.output_dir / f"{stem}.json"
        png_path = self.output_dir / f"{stem}.png"

        def _write():
            json_path.write_text(json.dumps(layout.model_dump(), indent=2) + "\n")
            render_wireframe(layout, png_path, style=os.environ.get("KRISNA_LAYOUT_WIREFRAME_STYLE", "gray"))

        await asyncio.to_thread(_write)
        def _save_blob():
            from PIL import Image
            with Image.open(png_path) as image:
                return get_blob_store().save_image(image.convert("RGB"), prefix="layout_wireframe")
        wireframe_ref = await asyncio.to_thread(_save_blob)
        return {
            "tier": self.spec.tier.value,
            "mode": "layout_baseline",
            "layout_ir": layout.model_dump(),
            "layout_ref": str(json_path),
            "wireframe_ref": wireframe_ref,
            "projection": {"displacement_mean": 0.0, "displacement_max": 0.0},
        }
