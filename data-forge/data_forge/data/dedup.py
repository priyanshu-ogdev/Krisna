"""FAISS-based deduplication — exact hash + semantic near-duplicate removal.

Two-phase dedup:
1. Exact: SHA-256 hash match (from manifest)
2. Semantic: CLIP embedding cosine similarity via FAISS index
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from data_forge.logging_setup import get_logger

log = get_logger("data.dedup")


class DedupEngine:
    """FAISS-powered deduplication engine."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        index_type: str = "IVF4096,Flat",
        nprobe: int = 64,
    ) -> None:
        self._threshold = similarity_threshold
        self._index_type = index_type
        self._nprobe = nprobe
        self._index: Any = None
        self._id_map: list[str] = []

    def build_index(
        self,
        embeddings: np.ndarray,
        record_ids: list[str],
    ) -> None:
        """Build a FAISS index from normalized embeddings.

        Args:
            embeddings: (N, D) float32 array, L2-normalized.
            record_ids: Corresponding record IDs.
        """
        import faiss

        n, d = embeddings.shape
        self._id_map = list(record_ids)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        if n < 10000:
            # Small dataset — flat index
            log.info("building_flat_index", n=n, d=d)
            self._index = faiss.IndexFlatIP(d)
        else:
            # Large dataset — IVF index
            nlist = min(int(np.sqrt(n)), 4096)
            log.info("building_ivf_index", n=n, d=d, nlist=nlist)
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.nprobe = self._nprobe

            # Train on all embeddings
            self._index.train(embeddings)

        self._index.add(embeddings)
        log.info("index_built", total_vectors=self._index.ntotal)

    def find_duplicates(
        self,
        embeddings: np.ndarray,
        record_ids: list[str],
        k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """Find near-duplicate pairs above the similarity threshold.

        Returns:
            List of (record_id_a, record_id_b, similarity) tuples.
        """
        import faiss

        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        faiss.normalize_L2(embeddings)
        scores, indices = self._index.search(embeddings, k)

        duplicates: list[tuple[str, str, float]] = []
        seen: set[frozenset[str]] = set()

        for i, (score_row, idx_row) in enumerate(zip(scores, indices)):
            query_id = record_ids[i]
            for sim, j in zip(score_row, idx_row):
                if j < 0 or j >= len(self._id_map):
                    continue
                match_id = self._id_map[j]
                if match_id == query_id:
                    continue
                if sim < self._threshold:
                    continue

                pair = frozenset([query_id, match_id])
                if pair not in seen:
                    seen.add(pair)
                    duplicates.append((query_id, match_id, float(sim)))

        log.info("duplicates_found", count=len(duplicates), threshold=self._threshold)
        return duplicates

    def save_index(self, path: Path) -> None:
        """Persist FAISS index to disk."""
        import json

        import faiss

        if self._index is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

        # Save ID map alongside
        id_map_path = path.with_suffix(".ids.json")
        id_map_path.write_text(
            json.dumps(self._id_map), encoding="utf-8"
        )
        log.info("index_saved", path=str(path))

    def load_index(self, path: Path) -> None:
        """Load a previously saved FAISS index."""
        import json

        import faiss

        self._index = faiss.read_index(str(path))
        id_map_path = path.with_suffix(".ids.json")
        if id_map_path.exists():
            self._id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
        log.info("index_loaded", path=str(path), vectors=self._index.ntotal)

    @staticmethod
    def generate_embeddings(
        image_paths: list[Path],
        clip_model: Any,
        clip_processor: Any,
        batch_size: int = 256,
        device: str = "cuda",
    ) -> np.ndarray:
        """Generate CLIP embeddings for a list of images.

        Returns:
            (N, D) float32 numpy array, L2-normalized.
        """
        import torch
        from PIL import Image

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                except Exception as e:
                    log.warning("image_load_failed", path=str(p), error=str(e))
                    # Use a blank image as placeholder
                    images.append(Image.new("RGB", (224, 224)))

            inputs = clip_processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = clip_model.get_image_features(**inputs)
                embeddings = outputs.cpu().numpy().astype(np.float32)
                all_embeddings.append(embeddings)

            log.debug(
                "embeddings_batch",
                batch=i // batch_size,
                count=len(batch_paths),
            )

        result = np.concatenate(all_embeddings, axis=0)
        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        result = result / norms

        log.info("embeddings_generated", total=result.shape[0], dim=result.shape[1])
        return result
