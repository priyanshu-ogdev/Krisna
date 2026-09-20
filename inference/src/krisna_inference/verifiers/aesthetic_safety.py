"""Aesthetic verifier — LAION-AI/aesthetic-predictor: a linear regressor on
top of CLIP ViT-L/14 image embeddings (768-dim), trained on SAC + LAION-Logos
+ AVA (see github.com/LAION-AI/aesthetic-predictor, laion.ai/blog/laion-aesthetics).
Predicts a 1-10 aesthetic score; rescaled to 0-1 here.

Safety verifier — Falconsai/nsfw_image_detection: a fine-tuned ViT
(google/vit-base-patch16-224-in21k base) binary "normal"/"nsfw" classifier,
~80M downloads on Hugging Face. Returns P(normal) as the 0-1 safety score.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

AESTHETIC_WEIGHTS_URL = (
    "https://github.com/LAION-AI/aesthetic-predictor/blob/main/"
    "sa_0_4_vit_l_14_linear.pth?raw=true"
)


class AestheticVerifier:
    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/krisna_aesthetic"))
        self._linear = None

    def _load(self) -> None:
        if self._linear is not None:
            return
        import torch
        import torch.nn as nn

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        weights_path = self.cache_dir / "sa_0_4_vit_l_14_linear.pth"
        if not weights_path.exists():
            urlretrieve(AESTHETIC_WEIGHTS_URL, weights_path)

        linear = nn.Linear(768, 1)
        state = torch.load(weights_path, map_location="cpu")
        linear.load_state_dict(state)
        linear.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._linear = linear.to(device)

    def score(self, image) -> float:
        from krisna_inference.verifiers.common import get_clip_embedder

        embedder = get_clip_embedder()
        embedder.load()
        self._load()

        import torch

        feat = embedder.embed_image(image)  # already L2-normalized, CLIP ViT-L/14 -> 768-dim
        with torch.no_grad():
            raw_score = self._linear(feat.to(self._linear.weight.device)).item()
        # Model predicts on a 1-10 scale (AVA/SAC/LAION-Logos convention).
        return max(0.0, min(1.0, raw_score / 10.0))


class SafetyVerifier:
    def __init__(self, model_id: str = "Falconsai/nsfw_image_detection") -> None:
        self.model_id = model_id
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline

        self._pipe = pipeline("image-classification", model=self.model_id)

    def score(self, image) -> float:
        """Returns P(normal) in [0, 1] — 1.0 = confidently safe."""
        self._load()
        results = self._pipe(image)  # [{"label": "normal"|"nsfw", "score": float}, ...]
        for r in results:
            if r["label"] == "normal":
                return float(r["score"])
        # Model returned only "nsfw" in its top result set — treat as unsafe.
        return 0.0
