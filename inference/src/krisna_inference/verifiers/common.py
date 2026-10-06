"""Shared CLIP model — loaded ONCE and reused by clip_alignment.py,
aesthetic_safety.py (aesthetic half), and handoff_consistency.py, rather
than each verifier loading its own copy of the same weights.
"""

from __future__ import annotations


import threading


class ClipEmbedder:
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14") -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self.model_id)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device).eval()

    def unload(self) -> None:
        import gc

        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed_image(self, image):
        import torch

        with self._lock:
            inputs = self._processor(images=image, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
            return feats / feats.norm(dim=-1, keepdim=True)

    def embed_text(self, text: str):
        import torch

        with self._lock:
            inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(
                self._model.device
            )
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
            return feats / feats.norm(dim=-1, keepdim=True)

    def image_text_similarity(self, image, text: str) -> float:
        img_feat = self.embed_image(image)
        txt_feat = self.embed_text(text)
        return float((img_feat @ txt_feat.T).item())

    def image_image_similarity(self, image_a, image_b) -> float:
        a = self.embed_image(image_a)
        b = self.embed_image(image_b)
        return float((a @ b.T).item())


_singleton: ClipEmbedder | None = None


def get_clip_embedder() -> ClipEmbedder:
    global _singleton
    if _singleton is None:
        _singleton = ClipEmbedder()
    return _singleton
