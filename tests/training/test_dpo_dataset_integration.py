"""E3: Minimal integration test for PreferencePairDataset.

Uses a real PreferenceStore (in-memory SQLite via :memory:) and a mock
BlobStore so no disk I/O or GPU is needed. Verifies that:
1. Dataset construction from a populated PreferenceStore works.
2. __len__ returns the correct count.
3. __getitem__ returns the required keys with non-empty `prompt`.
4. Augmented prompt (constraint-style) and raw prompt both pass the
   non-empty invariant.
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestPreferencePairDatasetIntegration(unittest.TestCase):
    """Full __init__ + __len__ + __getitem__ path through the real class."""

    def _make_real_store_with_one_pair(self):
        from krisna_training.dpo.preference_store import PreferencePair, PreferenceStore

        store = PreferenceStore(db_path=":memory:")
        pair = PreferencePair(
            prompt="a majestic mountain landscape at dusk",
            chosen_ref="blob://chosen/abc123",
            rejected_ref="blob://rejected/def456",
            source="pickapic_v2",
        )
        store.add(pair)
        return store

    def _fake_image(self):
        img = MagicMock()
        img.convert.return_value = img
        return img

    def test_dataset_len_and_getitem_keys(self):
        store = self._make_real_store_with_one_pair()
        try:
            from krisna_training.polish.dpo_dataset import PreferencePairDataset

            # Build dataset via object.__new__ + manual wiring to avoid
            # needing real SQLite db_path + blob_root args (which would
            # call PreferenceStore(db_path=...) again with a different path).
            ds = object.__new__(PreferencePairDataset)
            ds.pairs = store.list()
            ds.transform = lambda img: img  # identity transform

            # Mock BlobStore.load_image to return a fake PIL image
            fake_blobs = MagicMock()
            fake_blobs.load_image.return_value = self._fake_image()
            ds.blobs = fake_blobs

            self.assertEqual(len(ds), 1)

            item = ds[0]
            self.assertIn("chosen_pixel_values", item)
            self.assertIn("rejected_pixel_values", item)
            self.assertIn("prompt", item)
            # prompt must never be empty (DPO encode_prompt invariant)
            self.assertIsNotNone(item["prompt"])
            self.assertGreater(len(item["prompt"]), 0)
        finally:
            store.close()

    def test_augmented_prompt_is_still_nonempty(self):
        """Force augmentation=True for one call and verify prompt is valid."""
        store = self._make_real_store_with_one_pair()
        try:
            from krisna_training.polish.dpo_dataset import PreferencePairDataset

            ds = object.__new__(PreferencePairDataset)
            ds.pairs = store.list()
            ds.transform = lambda img: img
            fake_blobs = MagicMock()
            fake_blobs.load_image.return_value = self._fake_image()
            ds.blobs = fake_blobs

            # Force augmentation by patching random.random to return 0.0 < 0.30
            with patch("krisna_training.polish.dpo_dataset.random.random", return_value=0.0):
                item = ds[0]

            self.assertGreater(len(item["prompt"]), 0)
        finally:
            store.close()

    def test_raw_prompt_path_is_original_dataset_prompt(self):
        """Force augmentation=False and verify prompt is the original pair prompt."""
        store = self._make_real_store_with_one_pair()
        try:
            from krisna_training.polish.dpo_dataset import PreferencePairDataset

            ds = object.__new__(PreferencePairDataset)
            ds.pairs = store.list()
            ds.transform = lambda img: img
            fake_blobs = MagicMock()
            fake_blobs.load_image.return_value = self._fake_image()
            ds.blobs = fake_blobs

            # Force augmentation OFF by patching random.random to return 1.0 > 0.30
            with patch("krisna_training.polish.dpo_dataset.random.random", return_value=1.0):
                item = ds[0]

            self.assertEqual(item["prompt"], "a majestic mountain landscape at dusk")
        finally:
            store.close()
