"""Shared face-detection/blur helper.

Extracted from s03_5_pii_scrub.py so the same face-blur behavior (and the
same MediaPipe availability handling) can be reused by
s01_6_preference_pairs.py without duplicating the detector setup and blur
logic in two places that could silently drift apart.
"""

from __future__ import annotations

from typing import Any


def load_face_detector(min_confidence: float = 0.5) -> Any | None:
    """Return a MediaPipe FaceDetection instance, or None if unavailable.

    Callers must call `.close()` on the returned detector when done.
    """
    try:
        import mediapipe as mp

        return mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence
        )
    except ImportError:
        return None


def blur_faces(image: Any, face_detector: Any, blur_kernel_size: int = 99) -> tuple[Any, bool, list[str]]:
    """Blur any detected faces in a PIL Image in place (returns a copy).

    Returns (possibly-modified image, whether anything was blurred, list of
    detection descriptors for the manifest's `pii_detections` field).
    """
    import numpy as np
    from PIL import ImageFilter

    img = image.copy()
    img_array = np.array(img)
    detections: list[str] = []
    modified = False

    if face_detector is None:
        return img, modified, detections

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
