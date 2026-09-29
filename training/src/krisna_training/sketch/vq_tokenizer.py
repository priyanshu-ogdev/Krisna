"""VQ tokenizer — encode images to discrete tokens, decode tokens back to
pixels. The sketch tier operates entirely in this token space.

Default checkpoint: boris/vqgan_f16_16384 — a real, publicly downloadable
PyTorch VQGAN (taming-transformers architecture: github.com/CompVis/
taming-transformers), f=16 spatial downsample, 16384-entry codebook,
pretrained on ImageNet then fine-tuned on CC3M + a YFCC100M subset for
better faces/people/text coverage. A 256x256 image tokenizes to a 16x16
(256-token) grid; 512x512 to 32x32 (1024 tokens).

This is NOT a UI-domain-tuned tokenizer — it's a general-purpose one used
here because it's real and grounded, versus fabricating a "UI VQGAN" that
doesn't exist. If the eventual training data pipeline produces its own
domain-tuned VQ tokenizer (e.g. via data-forge's separately-flagged, still
-unverified Open-MAGVIT2 wrapper — see that project's own audit notes on
why that integration isn't ready yet), swap it in here: this class is the
seam, `VQTokenizer.encode`/`.decode` is the contract the rest of this
package depends on, not the specific checkpoint.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG_URL = (
    "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/"
    "?p=/configs/model.yaml&dl=1"
)
DEFAULT_CKPT_URL = (
    "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/"
    "?p=/ckpts/last.ckpt&dl=1"
)


class VQTokenizer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.device = device
        self._model = None
        self._codebook_size = None
        self._downsample = None

    @property
    def codebook_size(self) -> int:
        if self._codebook_size is None:
            self.load()
        return self._codebook_size

    @property
    def downsample_factor(self) -> int:
        if self._downsample is None:
            self.load()
        return self._downsample

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.exists() or not self.config_path.exists():
            raise FileNotFoundError(
                f"VQGAN checkpoint/config not found at {self.checkpoint_path} / "
                f"{self.config_path}. Run scripts/download_vqgan.sh first, or pass "
                "your own taming-transformers-compatible checkpoint+config."
            )

        import torch
        from omegaconf import OmegaConf

        try:
            from taming.models.vqgan import VQModel
        except ImportError as e:
            raise ImportError(
                "VQTokenizer requires the 'taming-transformers' package "
                "(see requirements-training.txt) — pip install "
                "'taming-transformers @ git+https://github.com/CompVis/taming-transformers.git'"
            ) from e

        config = OmegaConf.load(self.config_path)
        model = VQModel(**config.model.params)
        state = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        self._model = model
        self._codebook_size = config.model.params.n_embed
        # f16 in the checkpoint name = 2^4 spatial downsample; derived from
        # config rather than hard-coded, in case a different-f checkpoint
        # is swapped in.
        self._downsample = 2 ** (len(config.model.params.ddconfig.ch_mult) - 1)

    def encode(self, image) -> "list[int]":
        """image: PIL.Image. Returns a flat list of token ids, row-major
        over the downsampled grid (grid_h * grid_w entries)."""
        import numpy as np
        import torch

        self.load()
        arr = np.array(image.convert("RGB")).astype("float32") / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(next(self._model.parameters()).device)
        with torch.no_grad():
            _, _, (_, _, indices) = self._model.encode(tensor)
        return indices.reshape(-1).cpu().tolist()

    def decode(self, tokens: "list[int]", grid_h: int, grid_w: int):
        """tokens: flat list of token ids (grid_h * grid_w). Returns a PIL.Image."""
        import torch
        from PIL import Image

        self.load()
        device = next(self._model.parameters()).device
        indices = torch.tensor(tokens, device=device).reshape(1, grid_h, grid_w)
        quantized = self._model.quantize.get_codebook_entry(
            indices.reshape(-1), shape=(1, grid_h, grid_w, -1)
        )
        with torch.no_grad():
            pixels = self._model.decode(quantized)
        pixels = ((pixels.clamp(-1, 1) + 1.0) * 127.5).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        return Image.fromarray(pixels)

    def grid_shape_for(self, image_size: int) -> tuple[int, int]:
        g = image_size // self.downsample_factor
        return g, g
