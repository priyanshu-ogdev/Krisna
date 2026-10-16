"""Model artifact discovery, verification, and hardware-calibrated loader.

Provides production-grade discovery, SHA256 manifest integrity verification,
CUDA capability probing, and optimized llama.cpp runtime configuration.

References & Citations:
- Gerganov, G., et al. (2023). llama.cpp: Port of Facebook's LLaMA model in C/C++.
  https://github.com/ggml-org/llama.cpp (Demonstrating zero-copy mmap and unified GGUF).
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (NeurIPS 2022).
  FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
  arXiv:2205.14135.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
  arXiv:2307.08691.
- Lin, J., et al. (MLSys 2024). AWQ: Activation-aware Weight Quantization for On-Device LLM Compression.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("krisna_inference.backends.model_loader")

PLANNER_GGUF_FILENAME = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
PLANNER_MMPROJ_FILENAME = "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"

DEFAULT_CANDIDATE_DIRS = [
    Path("models/planner"),
    Path("models"),
    Path.home() / ".cache" / "krisna" / "models" / "planner",
    Path("custom_data_root/models/planner"),
]


@dataclass
class LlamaCppConfig:
    """Hardware-calibrated configuration for llama.cpp runtime.
    
    Attributes:
        model_path: Absolute or relative path to target GGUF file.
        mmproj_path: Optional path to multimodal projector GGUF.
        n_ctx: Context window size (tokens).
        n_gpu_layers: GPU layer offload count (-1 for full offload).
        use_mmap: Memory mapping flag for zero-copy startup (Gerganov et al., 2023).
        use_mlock: Memory locking flag (False on consumer hosts to prevent thrashing).
        flash_attn: FlashAttention flag (Dao et al., 2022, 2023).
        n_batch: Batch size for prompt evaluation.
        n_threads: CPU worker threads for compute kernels.
    """
    model_path: str
    mmproj_path: str | None = None
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    use_mmap: bool = True
    use_mlock: bool = False
    flash_attn: bool = True
    n_batch: int = 512
    n_threads: int = field(default_factory=lambda: max(1, min(8, (os.cpu_count() or 4) // 2)))

    def to_llama_kwargs(self) -> dict[str, Any]:
        """Convert configuration to kwargs accepted by llama_cpp.Llama."""
        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "n_batch": self.n_batch,
            "n_threads": self.n_threads,
            "verbose": False,
        }
        if self.mmproj_path:
            kwargs["clip_model_path"] = self.mmproj_path
        # FlashAttention parameter in llama-cpp-python
        kwargs["flash_attn"] = self.flash_attn
        return kwargs


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 checksum of file in buffered chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_planner_model(explicit_path: str | None = None) -> Path | None:
    """Discover verified Planner GGUF model artifact across priority locations.
    
    Priority order:
    1. Explicit path parameter.
    2. KRISNA_PLANNER_GGUF environment variable.
    3. Auto-discovery in candidate model directories (matching *.gguf).
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p.resolve()
        if p.is_dir():
            ggufs = sorted(p.glob("*.gguf"))
            if ggufs:
                return ggufs[0].resolve()

    env_path = os.environ.get("KRISNA_PLANNER_GGUF")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p.resolve()
        if p.is_dir():
            ggufs = sorted(p.glob("*.gguf"))
            if ggufs:
                return ggufs[0].resolve()

    for candidate_dir in DEFAULT_CANDIDATE_DIRS:
        if candidate_dir.is_dir():
            target = candidate_dir / PLANNER_GGUF_FILENAME
            if target.is_file():
                log.info("Auto-discovered target Planner GGUF at %s", target)
                return target.resolve()
            # Exclude mmproj from planner model match
            ggufs = [f for f in sorted(candidate_dir.glob("*.gguf")) if "mmproj" not in f.name.lower()]
            if ggufs:
                log.info("Auto-discovered Planner GGUF at %s", ggufs[0])
                return ggufs[0].resolve()

    return None


def discover_mmproj_model(explicit_path: str | None = None) -> Path | None:
    """Discover multimodal projector (mmproj) GGUF for vision-language planning."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p.resolve()

    env_path = os.environ.get("KRISNA_PLANNER_MMPROJ")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p.resolve()

    for candidate_dir in DEFAULT_CANDIDATE_DIRS:
        if candidate_dir.is_dir():
            target = candidate_dir / PLANNER_MMPROJ_FILENAME
            if target.is_file():
                log.info("Auto-discovered target mmproj GGUF at %s", target)
                return target.resolve()
            mmprojs = [f for f in sorted(candidate_dir.glob("*.gguf")) if "mmproj" in f.name.lower()]
            if mmprojs:
                log.info("Auto-discovered mmproj GGUF at %s", mmprojs[0])
                return mmprojs[0].resolve()

    return None


def verify_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """Verify SHA256 integrity of all artifacts declared in manifest.json."""
    m_path = Path(manifest_path)
    if not m_path.is_file():
        return {"valid": False, "error": f"Manifest not found: {m_path}"}

    try:
        data = json.loads(m_path.read_text())
        artifacts = data.get("artifacts", [])
        verified_entries = []
        for entry in artifacts:
            filename = entry.get("filename")
            expected_hash = entry.get("sha256")
            target = m_path.parent / filename
            if not target.exists():
                return {"valid": False, "error": f"Artifact missing on disk: {target}"}
            actual_hash = sha256_file(target)
            if actual_hash.lower() != expected_hash.lower():
                return {
                    "valid": False,
                    "error": f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}",
                }
            verified_entries.append({"filename": filename, "verified": True})
        return {"valid": True, "repo_id": data.get("repo_id"), "artifacts": verified_entries}
    except Exception as e:
        return {"valid": False, "error": f"Manifest parsing error: {e}"}


def probe_cuda_hardware() -> dict[str, Any]:
    """Probe live hardware capabilities for calibrating llama.cpp and Diffusers."""
    result: dict[str, Any] = {
        "cuda_available": False,
        "device_name": None,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "recommended_gpu_layers": 0,
        "supports_flash_attn": False,
    }

    try:
        import torch
        if torch.cuda.is_available():
            result["cuda_available"] = True
            dev = torch.cuda.current_device()
            result["device_name"] = torch.cuda.get_device_name(dev)
            free_b, total_b = torch.cuda.mem_get_info(dev)
            result["vram_free_gb"] = round(free_b / (1024 ** 3), 2)
            result["vram_total_gb"] = round(total_b / (1024 ** 3), 2)
            cap = torch.cuda.get_device_capability(dev)
            result["compute_capability"] = f"{cap[0]}.{cap[1]}"
            # Compute capability >= 8.0 (Ampere, Ada, Hopper) supports FlashAttention
            result["supports_flash_attn"] = cap[0] >= 8
            # If >= 8GB VRAM, offload all layers (-1)
            result["recommended_gpu_layers"] = -1 if result["vram_total_gb"] >= 8.0 else 24
    except Exception as e:
        log.debug("CUDA hardware probe exception: %s", e)

    return result


def build_calibrated_planner_config() -> LlamaCppConfig | None:
    """Build production-ready, hardware-tuned configuration for Planner runtime."""
    model_path = discover_planner_model()
    if not model_path:
        return None

    mmproj_path = discover_mmproj_model()
    hw = probe_cuda_hardware()

    ctx = int(os.environ.get("KRISNA_PLANNER_CTX", "8192"))
    # In low-VRAM mode, default to 4096 or 8192
    if os.environ.get("KRISNA_LOW_VRAM_MODE") == "1":
        ctx = min(ctx, 4096)

    gpu_layers = int(os.environ.get("KRISNA_PLANNER_GPU_LAYERS", str(hw["recommended_gpu_layers"])))
    flash_attn = os.environ.get("KRISNA_PLANNER_FLASH_ATTN", "1") == "1" and hw["supports_flash_attn"]

    return LlamaCppConfig(
        model_path=str(model_path),
        mmproj_path=str(mmproj_path) if mmproj_path else None,
        n_ctx=ctx,
        n_gpu_layers=gpu_layers,
        use_mmap=True,  # Fast zero-copy memory mapping
        use_mlock=False,  # Avoid pinning all memory in consumer 16GB RAM
        flash_attn=flash_attn,
        n_batch=512,
    )
