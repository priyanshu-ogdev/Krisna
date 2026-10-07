"""Shared face-detection/blur helper.

Extracted from s03_5_pii_scrub.py so the same face-blur behavior (and the
same MediaPipe availability handling) can be reused by
s01_6_preference_pairs.py without duplicating the detector setup and blur
logic in two places that could silently drift apart.
"""

from __future__ import annotations

import numpy as np
from PIL import ImageFilter
from typing import Any


def load_face_detector(min_confidence: float = 0.5) -> Any | None:
    """Return a MediaPipe FaceDetection or FaceDetector instance, or None if unavailable.

    Callers must call `.close()` on the returned detector when done.
    """
    # 1. Modern MediaPipe 1.x Tasks API
    try:
        from pathlib import Path
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_candidates = [
            Path(__file__).resolve().parents[3] / "configs" / "models" / "blaze_face_short_range.tflite",
            Path("data-forge/configs/models/blaze_face_short_range.tflite"),
            Path.cwd() / "configs" / "models" / "blaze_face_short_range.tflite",
        ]
        model_path = next((p for p in model_candidates if p.exists()), None)
        if model_path:
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=min_confidence)
            return vision.FaceDetector.create_from_options(options)
    except Exception:
        pass

    # 2. Legacy MediaPipe solutions API
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            return mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=min_confidence
            )
    except (ImportError, AttributeError):
        pass

    return None


def blur_faces(image: Any, face_detector: Any, blur_kernel_size: int = 99) -> tuple[Any, bool, list[str]]:
    """Blur any detected faces in a PIL Image in place (returns a copy).

    Returns (possibly-modified image, whether anything was blurred, list of
    detection descriptors for the manifest's `pii_detections` field).
    """
    img = image.copy()
    img_array = np.array(img)
    detections: list[str] = []
    modified = False

    if face_detector is None:
        return img, modified, detections

    # Modern MediaPipe 1.x Tasks FaceDetector
    if hasattr(face_detector, "detect"):
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_array))
        results = face_detector.detect(mp_image)
        if not results.detections:
            return img, modified, detections

        h_img, w_img = img_array.shape[:2]
        for detection in results.detections:
            bbox = detection.bounding_box
            x1 = max(0, int(bbox.left))
            y1 = max(0, int(bbox.top))
            x2 = min(w_img, int(bbox.right))
            y2 = min(h_img, int(bbox.bottom))
            if x2 <= x1 or y2 <= y1:
                continue

            face_region = img.crop((x1, y1, x2, y2))
            blurred = face_region.filter(ImageFilter.GaussianBlur(radius=blur_kernel_size // 2))
            img.paste(blurred, (x1, y1))
            modified = True
            detections.append(f"face_detected_at_{x1}_{y1}")

        return img, modified, detections

    # Legacy MediaPipe solutions FaceDetection
    if hasattr(face_detector, "process"):
        mp_results = face_detector.process(img_array)
        if not mp_results.detections:
            return img, modified, detections

        for detection in mp_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h_img, w_img = img_array.shape[:2]
            x1 = max(0, int(bbox.xmin * w_img))
            y1 = max(0, int(bbox.ymin * h_img))
            x2 = min(w_img, int((bbox.xmin + bbox.width) * w_img))
            y2 = min(h_img, int((bbox.ymin + bbox.height) * h_img))
            if x2 <= x1 or y2 <= y1:
                continue

            face_region = img.crop((x1, y1, x2, y2))
            blurred = face_region.filter(ImageFilter.GaussianBlur(radius=blur_kernel_size // 2))
            img.paste(blurred, (x1, y1))
            modified = True
            detections.append(f"face_detected_at_{x1}_{y1}")

        return img, modified, detections

    return img, modified, detections
