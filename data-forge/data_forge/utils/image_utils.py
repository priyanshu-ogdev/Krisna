"""Image loading, resizing, and format conversion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    import torch

from data_forge.logging_setup import get_logger

log = get_logger("utils.image_utils")


def load_image(path: Path | str) -> Image.Image:
    """Load an image file and convert to RGB."""
    return Image.open(str(path)).convert("RGB")


def resize_for_model(
    image: Image.Image,
    max_size: int = 2048,
    min_size: int = 256,
) -> Image.Image:
    """Resize image to fit within model input constraints.

    Preserves aspect ratio. Only downscales if > max_size.
    Upscales only if both dimensions < min_size.
    """
    w, h = image.size

    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return image.resize((new_w, new_h), Image.LANCZOS)

    if w < min_size and h < min_size:
        ratio = min_size / min(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return image.resize((new_w, new_h), Image.LANCZOS)

    return image


def image_to_tensor(image: Image.Image) -> "torch.Tensor":
    """Convert PIL image to normalized torch tensor (C, H, W), float32."""
    import torchvision.transforms.functional as TF

    tensor = TF.to_tensor(image)  # (C, H, W), [0, 1]
    return tensor


def normalize_for_vae(
    tensor: "torch.Tensor",
    mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    std: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> "torch.Tensor":
    """Normalize tensor from [0, 1] to [-1, 1] for VAE input."""
    import torchvision.transforms.functional as TF

    return TF.normalize(tensor, mean, std)


def pad_to_multiple(
    image: Image.Image, multiple: int = 16
) -> Image.Image:
    """Pad image to nearest multiple of `multiple` on both dimensions."""
    w, h = image.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple

    if new_w == w and new_h == h:
        return image

    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded.paste(image, (0, 0))
    return padded


def get_image_info(path: Path) -> dict[str, Any]:
    """Get image metadata without fully loading it."""
    try:
        with Image.open(path) as img:
            return {
                "width": img.size[0],
                "height": img.size[1],
                "format": img.format,
                "mode": img.mode,
                "file_size_bytes": path.stat().st_size,
            }
    except Exception as e:
        return {"error": str(e)}
