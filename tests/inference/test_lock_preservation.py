from PIL import Image

from krisna_inference.verifiers.lock_preservation import LockPreservationVerifier


def test_lock_preservation_is_exact_for_unchanged_region():
    source = Image.new("RGB", (20, 20), (20, 20, 20))
    rendered = source.copy()
    score = LockPreservationVerifier().score(
        source,
        rendered,
        [{"bbox": [0.0, 0.0, 0.5, 0.5]}],
    )
    assert score == 1.0


def test_lock_preservation_penalizes_changed_pixels():
    source = Image.new("RGB", (20, 20), (0, 0, 0))
    rendered = Image.new("RGB", (20, 20), (255, 255, 255))
    score = LockPreservationVerifier().score(
        source,
        rendered,
        [{"bbox": [0.0, 0.0, 0.5, 0.5]}],
    )
    assert score == 0.0
