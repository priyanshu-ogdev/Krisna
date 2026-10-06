"""FAISS-based deduplication — exact hash + semantic near-duplicate removal.

Two-phase dedup:
1. Exact: SHA-256 hash match (from manifest or computed on disk)
2. Semantic: CLIP embedding cosine similarity via FAISS index
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from data_forge.logging_setup import get_logger

log = get_logger("data.dedup")


class DedupEngine:
    """FAISS-powered deduplication engine with GPU/multi-core acceleration and atomic persistence."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        index_type: str = "Flat",
        nprobe: int = 64,
        use_gpu: bool = True,
    ) -> None:
        self._threshold = similarity_threshold
        self._index_type = index_type
        self._nprobe = nprobe
        self._use_gpu = use_gpu
        self._index: Any = None
        self._gpu_res: Any = None
        self._is_gpu_index: bool = False
        self._id_map: list[str] = []

        # Optimize OpenMP thread count for CPU FAISS
        try:
            import faiss
            import psutil

            num_threads = max(1, psutil.cpu_count(logical=False) or 4)
            faiss.omp_set_num_threads(num_threads)
        except Exception:
            pass

    def _maybe_move_to_gpu(self) -> None:
        """Transfer FAISS index to GPU if CUDA and faiss-gpu are available."""
        if not self._use_gpu or self._index is None or self._is_gpu_index:
            return

        try:
            import faiss
            import torch

            if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
                if self._gpu_res is None:
                    self._gpu_res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._index)
                self._is_gpu_index = True
                log.info("faiss_index_moved_to_gpu", device="cuda:0")
        except Exception as e:
            log.debug("faiss_gpu_transfer_skipped", error=str(e))
            self._is_gpu_index = False

    def _ensure_cpu_index(self) -> Any:
        """Return a CPU copy of the index for serialization."""
        if self._index is None:
            return None
        if self._is_gpu_index:
            try:
                import faiss
                if hasattr(faiss, "index_gpu_to_cpu"):
                    return faiss.index_gpu_to_cpu(self._index)
            except Exception as e:
                log.warning("faiss_gpu_to_cpu_failed", error=str(e))
        return self._index

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

        if len(record_ids) == 0:
            return

        n, d = embeddings.shape
        self._id_map = list(record_ids)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # Determine index architecture
        idx_type = self._index_type.strip().lower()
        if idx_type.startswith("hnsw"):
            # Graph-based HNSW: ultra-fast incremental addition, zero training needed
            m = 32
            if idx_type.startswith("hnsw") and len(idx_type) > 4 and idx_type[4:].isdigit():
                m = int(idx_type[4:])
            log.info("building_hnsw_index", n=n, d=d, m=m)
            self._index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
        elif idx_type.startswith("ivf"):
            # IVF index requires enough training vectors
            if n < 256:
                log.info("building_flat_index_small_corpus", n=n, d=d)
                self._index = faiss.IndexFlatIP(d)
            else:
                nlist = min(int(np.sqrt(n)), 4096)
                log.info("building_ivf_index", n=n, d=d, nlist=nlist)
                quantizer = faiss.IndexFlatIP(d)
                self._index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                self._index.nprobe = self._nprobe
                self._index.train(embeddings)
        else:
            # Default: Flat index (exact cosine similarity)
            log.info("building_flat_index", n=n, d=d)
            self._index = faiss.IndexFlatIP(d)

        self._index.add(embeddings)
        self._maybe_move_to_gpu()
        log.info("index_built", total_vectors=self._index.ntotal)

    def add_records(
        self,
        embeddings: np.ndarray,
        record_ids: list[str],
    ) -> None:
        """Incrementally add records to the persistent index."""
        if len(record_ids) == 0:
            return

        if self._index is None:
            self.build_index(embeddings, record_ids)
            return

        import faiss

        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._id_map.extend(record_ids)
        log.info("records_added_to_index", added=len(record_ids), total=self._index.ntotal)

    def dedup_batch(
        self,
        embeddings: np.ndarray,
        record_ids: list[str],
        threshold: float | None = None,
    ) -> tuple[list[tuple[str, str, float]], list[int]]:
        """Intra-batch deduplication: identify pairs within the batch that are semantic duplicates.

        Returns:
            duplicates: list of (duplicate_id, canonical_id, similarity) where duplicate_id is
                        the subsequent record and canonical_id is the earlier record.
            survivor_indices: list of indices in record_ids that are not duplicates.
        """
        import faiss

        n = len(record_ids)
        if n <= 1:
            return [], list(range(n))

        thresh = threshold if threshold is not None else self._threshold
        emb_norm = np.copy(embeddings)
        faiss.normalize_L2(emb_norm)
        d = emb_norm.shape[1]

        # Use temporary flat index for fast intra-batch search
        temp_index = faiss.IndexFlatIP(d)
        temp_index.add(emb_norm)

        k = min(50, n)
        scores, indices = temp_index.search(emb_norm, k)

        duplicates: list[tuple[str, str, float]] = []
        excluded_indices: set[int] = set()

        for i in range(n):
            if i in excluded_indices:
                continue

            query_id = record_ids[i]
            for sim, j in zip(scores[i], indices[i]):
                # j <= i means self or previous record
                if j <= i or j in excluded_indices or j >= n:
                    continue

                if float(sim) >= thresh:
                    match_id = record_ids[j]
                    excluded_indices.add(j)
                    duplicates.append((match_id, query_id, float(sim)))

        survivor_indices = [i for i in range(n) if i not in excluded_indices]
        log.info(
            "intra_batch_dedup_done",
            total=n,
            duplicates=len(duplicates),
            survivors=len(survivor_indices),
            threshold=thresh,
        )
        return duplicates, survivor_indices

    def search_existing(
        self,
        embeddings: np.ndarray,
        record_ids: list[str],
        threshold: float | None = None,
        k: int = 5,
    ) -> tuple[list[tuple[str, str, float]], list[int]]:
        """Search surviving candidates against the persistent FAISS index.

        Returns:
            historical_duplicates: list of (incoming_id, historical_id, similarity)
            survivor_indices: list of indices in record_ids that have no historical duplicate match
        """
        if self._index is None or len(self._id_map) == 0 or len(record_ids) == 0:
            return [], list(range(len(record_ids)))

        import faiss

        thresh = threshold if threshold is not None else self._threshold
        emb_norm = np.copy(embeddings)
        faiss.normalize_L2(emb_norm)

        k_search = min(k, self._index.ntotal)
        scores, indices = self._index.search(emb_norm, k_search)

        historical_duplicates: list[tuple[str, str, float]] = []
        survivor_indices: list[int] = []

        for i, q_id in enumerate(record_ids):
            matched = False
            for sim, j in zip(scores[i], indices[i]):
                if j < 0 or j >= len(self._id_map):
                    continue
                if float(sim) >= thresh:
                    match_id = self._id_map[j]
                    historical_duplicates.append((q_id, match_id, float(sim)))
                    matched = True
                    break

            if not matched:
                survivor_indices.append(i)

        log.info(
            "inter_batch_dedup_done",
            candidates=len(record_ids),
            historical_duplicates=len(historical_duplicates),
            survivors=len(survivor_indices),
            threshold=thresh,
        )
        return historical_duplicates, survivor_indices

    def find_duplicates(
        self,
        embeddings: np.ndarray,
        record_ids: list[str],
        k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """Find near-duplicate pairs above the similarity threshold (legacy compatibility).

        Returns:
            List of (query_id, match_id, similarity) tuples.
        """
        import faiss

        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        faiss.normalize_L2(embeddings)
        k_search = min(k, self._index.ntotal)
        scores, indices = self._index.search(embeddings, k_search)

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
                if float(sim) < self._threshold:
                    continue

                pair = frozenset([query_id, match_id])
                if pair not in seen:
                    seen.add(pair)
                    duplicates.append((query_id, match_id, float(sim)))

        log.info("duplicates_found", count=len(duplicates), threshold=self._threshold)
        return duplicates

    def save_index(self, path: Path) -> None:
        """Persist FAISS index and ID map to disk atomically."""
        if self._index is None:
            return

        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_index = self._ensure_cpu_index()

        tmp_index_path = path.with_suffix(".bin.tmp")
        tmp_id_path = path.with_suffix(".ids.json.tmp")
        final_id_path = path.with_suffix(".ids.json")

        faiss.write_index(cpu_index, str(tmp_index_path))
        tmp_id_path.write_text(json.dumps(self._id_map), encoding="utf-8")

        # Atomic replacement to prevent corruption
        os.replace(tmp_index_path, path)
        os.replace(tmp_id_path, final_id_path)
        log.info("index_saved_atomically", path=str(path), vectors=len(self._id_map))

    def load_index(self, path: Path) -> None:
        """Load a previously saved FAISS index and ID map."""
        import faiss

        self._index = faiss.read_index(str(path))
        id_map_path = path.with_suffix(".ids.json")
        if id_map_path.exists():
            self._id_map = json.loads(id_map_path.read_text(encoding="utf-8"))

        if self._index.ntotal != len(self._id_map):
            log.warning(
                "index_id_map_length_mismatch",
                index_ntotal=self._index.ntotal,
                id_map_len=len(self._id_map),
            )

        self._maybe_move_to_gpu()
        log.info("index_loaded", path=str(path), vectors=self._index.ntotal)

    def cleanup(self) -> None:
        """Release FAISS GPU and CPU memory."""
        self._index = None
        self._gpu_res = None
        self._is_gpu_index = False

        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("dedup_engine_cleaned_up")

    @staticmethod
    def generate_embeddings(
        image_paths: list[Path],
        clip_model: Any,
        clip_processor: Any,
        batch_size: int = 256,
        device: str | None = None,
    ) -> tuple[np.ndarray, list[int], list[int]]:
        """Generate CLIP embeddings for a list of images.

        Returns:
            embeddings: (M, D) float32 numpy array, L2-normalized.
            valid_indices: indices in image_paths that were successfully embedded.
            failed_indices: indices in image_paths that failed to load/decode.
        """
        import torch
        from PIL import Image

        target_device = device or getattr(clip_model, "device", None) or next(clip_model.parameters()).device
        model_dtype = getattr(clip_model, "dtype", torch.float32)

        all_embeddings: list[np.ndarray] = []
        valid_indices: list[int] = []
        failed_indices: list[int] = []

        for batch_start in range(0, len(image_paths), batch_size):
            batch_slice = image_paths[batch_start : batch_start + batch_size]
            batch_images: list[Image.Image] = []
            batch_valid_idx: list[int] = []

            for offset, p in enumerate(batch_slice):
                orig_idx = batch_start + offset
                try:
                    with Image.open(p) as img:
                        img.load()
                        batch_images.append(img.convert("RGB"))
                    batch_valid_idx.append(orig_idx)
                except Exception as e:
                    log.warning("image_load_failed", path=str(p), error=str(e))
                    failed_indices.append(orig_idx)

            if not batch_images:
                continue

            inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {
                k: v.to(device=target_device, dtype=model_dtype if v.is_floating_point() else v.dtype)
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                outputs = clip_model.get_image_features(**inputs)
                embeddings = outputs.detach().cpu().numpy().astype(np.float32)
                all_embeddings.append(embeddings)
                valid_indices.extend(batch_valid_idx)

            log.debug(
                "embeddings_batch",
                batch=batch_start // batch_size,
                valid_count=len(batch_images),
            )

        if not all_embeddings:
            empty_dim = getattr(clip_model.config, "projection_dim", 768)
            return np.empty((0, empty_dim), dtype=np.float32), [], failed_indices

        result = np.concatenate(all_embeddings, axis=0)
        # L2 normalize
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        result = result / norms

        log.info(
            "embeddings_generated",
            total_valid=result.shape[0],
            total_failed=len(failed_indices),
            dim=result.shape[1],
        )
        return result, valid_indices, failed_indices
