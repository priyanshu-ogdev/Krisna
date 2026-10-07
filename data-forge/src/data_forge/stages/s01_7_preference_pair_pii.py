"""OCR and text-PII redaction for preference-pair images."""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import re
import uuid
from typing import Any, ClassVar

from PIL import Image, ImageDraw

from data_forge.config import PipelineConfig
from data_forge.inference.ocr import OCREngine
from data_forge.logging_setup import get_logger
from data_forge.manifest import Manifest
from data_forge.orchestrator import register_stage
from data_forge.stages.base import Stage, StageResult

log = get_logger("stages.s01_7")


def _redact_image(path: Path, regions: list[dict[str, Any]], patterns: dict[str, re.Pattern]) -> tuple[bool, list[str]]:
    matched: list[str] = []
    boxes: list[list[float]] = []
    for region in regions:
        text = region.get("text", "")
        if not isinstance(text, str):
            continue
        for name, pattern in patterns.items():
            if not pattern.search(text):
                continue
            matched.append(name)
            bbox = region.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return False, matched
            try:
                coordinates = [float(value) for value in bbox]
            except (TypeError, ValueError):
                return False, matched
            if not all(math.isfinite(value) for value in coordinates):
                return False, matched
            boxes.append(coordinates)

    if not boxes:
        return True, matched

    with Image.open(path) as source:
        image_format = source.format
        image = source.convert("RGB")
    if image_format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
        return False, matched

    width, height = image.size
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in boxes:
        if all(0.0 <= coordinate <= 1.0 for coordinate in (x1, y1, x2, y2)):
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return False, matched
        draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))

    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.redacting")
    try:
        image.save(temp_path, format=image_format, **({"quality": 95} if image_format == "JPEG" else {}))
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return True, matched


@register_stage("s01_7_preference_pair_pii")
class PreferencePairPIIStage(Stage):
    name = "s01_7_preference_pair_pii"
    requires: ClassVar[tuple[str, ...]] = ("s01_6_preference_pairs",)

    async def run(
        self,
        manifest: Manifest,
        config: PipelineConfig,
        record_ids: list[str],
        engine: Any | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        if engine is None:
            raise RuntimeError("Preference-pair PII scrub requires the dedicated OCR engine")

        root = config.resolved_paths["preference_pairs"]
        stage_cfg = config.get_stage("s01_7_preference_pair_pii")
        token = config.get_stage("s05_5_pii_text_redact").get("redaction_token", "[REDACTED]")
        patterns: dict[str, re.Pattern] = {}
        for name, pattern in config.get_stage("s05_5_pii_text_redact").get("regex_patterns", {}).items():
            try:
                patterns[name] = re.compile(pattern)
            except re.error as error:
                raise ValueError(f"Invalid configured PII pattern {name!r}: {error}") from error
        if not patterns:
            raise ValueError("Preference-pair PII scrub requires at least one configured detection pattern")

        ocr = OCREngine(engine, config)
        concurrency = max(1, min(int(stage_cfg.get("max_concurrent_pairs", 8)), 16))
        semaphore = asyncio.Semaphore(concurrency)
        image_locks = [asyncio.Lock() for _ in range(256)]
        processed = failed = 0

        def _safe_path(directory: Path, name: str) -> Path | None:
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or Path(name).is_absolute()
                or Path(name).name != name
                or "\\" in name
            ):
                return None
            try:
                base = directory.resolve()
                path = (directory / name).resolve(strict=True)
            except OSError:
                return None
            return path if path.parent == base and path.is_file() else None

        def _sanitize(value: Any) -> Any:
            if isinstance(value, str):
                for pattern in patterns.values():
                    value = pattern.sub(token, value)
                return value
            if isinstance(value, list):
                return [_sanitize(item) for item in value]
            if isinstance(value, dict):
                return {key: _sanitize(item) for key, item in value.items()}
            return value

        async def _process_pair(meta_path: Path) -> tuple[bool, str | None]:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    meta.get("dedup_status") != "unique"
                    or meta.get("safety_tier") != "safe"
                    or meta.get("pii_scrubbed") is not True
                ):
                    return False, None
                paths = [
                    _safe_path(meta_path.parent, meta.get("image_a")),
                    _safe_path(meta_path.parent, meta.get("image_b")),
                ]
                if any(path is None for path in paths):
                    raise ValueError("Preference pair image path is missing or unsafe")

                detections: list[str] = []
                async with semaphore:
                    for image_path in paths:
                        assert image_path is not None
                        lock = image_locks[hash(str(image_path)) % len(image_locks)]
                        async with lock:
                            ocr_result = await ocr.extract_text(image_path)
                            if ocr_result is None:
                                raise RuntimeError("OCR returned no structured result")
                            success, found = await asyncio.to_thread(
                                _redact_image,
                                image_path,
                                ocr_result.model_dump().get("text_regions", []),
                                patterns,
                            )
                            if not success:
                                raise RuntimeError("PII was detected without a valid redaction region")
                            detections.extend(found)

                meta = _sanitize(meta)
                meta["pii_detections"] = sorted(set(detections))
                meta["text_pii_scrubbed"] = True
                temp_meta = meta_path.with_name(f".{meta_path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    temp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                    os.replace(temp_meta, meta_path)
                finally:
                    if temp_meta.exists():
                        temp_meta.unlink()
                return True, None
            except Exception as error:
                log.error("preference_pair_pii_failed", metadata=str(meta_path), error=str(error))
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["text_pii_scrubbed"] = False
                    temp_meta = meta_path.with_name(f".{meta_path.name}.{uuid.uuid4().hex}.tmp")
                    try:
                        temp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                        os.replace(temp_meta, meta_path)
                    finally:
                        if temp_meta.exists():
                            temp_meta.unlink()
                except Exception as write_error:
                    log.error("preference_pair_pii_status_write_failed", metadata=str(meta_path), error=str(write_error))
                return False, str(error)

        batch_size = max(concurrency, int(stage_cfg.get("batch_size", 64)))
        meta_paths = sorted(root.glob("*/*.json"))
        for start in range(0, len(meta_paths), batch_size):
            results = await asyncio.gather(*(
                _process_pair(path) for path in meta_paths[start : start + batch_size]
            ))
            for was_processed, error in results:
                if was_processed:
                    processed += 1
                elif error:
                    failed += 1

        result.records_processed = processed
        result.records_failed = failed
        result.metadata = {"text_pii_scrubbed": processed, "failed": failed}
        return result
