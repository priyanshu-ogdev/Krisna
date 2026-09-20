from __future__ import annotations

import pytest

pytest.importorskip("PIL")

from krisna_inference.backends.common import BlobStore


def test_save_and_load_roundtrip(tmp_path):
    from PIL import Image

    store = BlobStore(root=tmp_path / "blobs")
    img = Image.new("RGB", (4, 4), color="red")
    ref = store.save_image(img, prefix="test")

    assert ref.startswith("blob://test_")
    loaded = store.load_image(ref)
    assert loaded.size == (4, 4)


def test_load_missing_blob_raises(tmp_path):
    store = BlobStore(root=tmp_path / "blobs")
    with pytest.raises(FileNotFoundError):
        store.load_image("blob://does-not-exist.png")


def test_load_non_blob_ref_raises(tmp_path):
    store = BlobStore(root=tmp_path / "blobs")
    with pytest.raises(ValueError):
        store.load_image("not-a-blob-ref")


def test_path_for(tmp_path):
    store = BlobStore(root=tmp_path / "blobs")
    p = store.path_for("blob://foo.png")
    assert p == (tmp_path / "blobs" / "foo.png")


def test_delete_blob_lifecycle(tmp_path):
    from PIL import Image

    store = BlobStore(root=tmp_path / "blobs")
    img = Image.new("RGB", (4, 4), color="blue")
    ref = store.save_image(img, prefix="to_delete")

    assert (tmp_path / "blobs" / ref.removeprefix("blob://")).exists()
    deleted = store.delete(ref)
    assert deleted is True
    assert not (tmp_path / "blobs" / ref.removeprefix("blob://")).exists()

    # Deleting again returns False
    assert store.delete(ref) is False

    # Deleting invalid ref raises ValueError
    with pytest.raises(ValueError):
        store.delete("invalid_ref")
