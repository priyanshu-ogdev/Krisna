"""Production-grade Planner runtime powered by llama.cpp and GGUF.

Provides zero-copy memory-mapped loading, hardware-calibrated GPU layer offloading,
FlashAttention KV cache optimization, grammar-guided JSON schema decoding,
and hybrid UICrit RAG retrieval.

References & Citations:
- Gerganov, G., et al. (2023). llama.cpp: High-performance inference of GGUF models with k-quants.
  https://github.com/ggml-org/llama.cpp (Demonstrating zero-copy mmap, k-quants, and unified GGUF execution).
- Geng, S., Josifoski, M., Peyrard, M., & West, R. (EMNLP 2023).
  Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning.
  In Proceedings of EMNLP 2023, pp. 10932–10952. DOI: 10.18653/v1/2023.emnlp-main.674.
- Dao, T. (ICLR 2024 Oral).
  FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
  In The Twelfth International Conference on Learning Representations (ICLR 2024).
- Lin, J., Tang, J., Tang, H., Yang, S., et al. (MLSys 2024).
  AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.
  In Proceedings of MLSys 2024, vol. 6, pp. 87–100.
- Kwon, W., Li, Z., Zhuang, S., et al. (ACM SOSP 2023).
  Efficient Memory Management for Large Language Model Serving with PagedAttention.
  In Proceedings of the 29th ACM SOSP, pp. 611–626. DOI: 10.1145/3600006.3613165.
- Xiao, G., Lin, J., Seznec, M., et al. (ICML 2023).
  SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.
  In Proceedings of ICML 2023, PMLR vol. 202, pp. 38087–38099.
- Lin, C., Lin, Y., Pan, P., Zhang, X., & Mu, Y. (IEEE TPAMI 2025).
  InstructLayout: Instruction-Driven 2D and 3D Layout Synthesis with Semantic Graph Prior.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 12, pp. 11040–11050.
  DOI: 10.1109/TPAMI.2025.3595880.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from krivex_inference.backends.model_loader import (
    LlamaCppConfig,
    build_calibrated_planner_config,
    discover_mmproj_model,
    discover_planner_model,
    probe_cuda_hardware,
)
from krivex_inference.backends.blob_store_singleton import get_blob_store
from krivex_inference.backends.planner_rag import (
    UICritRAGIndex,
    resolve_embed_fn_from_env,
    resolve_visual_embed_fn_from_env,
)
from krivex_inference.backends.planner_schema import PlannerResponse
from krivex_inference.orchestrator.exceptions import BackendLoadError
from krivex_inference.orchestrator.model_registry import ModelBackend

log = logging.getLogger("krivex_inference.backends.planner_backend_llama")

_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_DEFAULT_MAX_IMAGE_PIXELS = 100_000_000


class PlannerBackendLlama(ModelBackend):
    """High-performance, frozen Planner backend running under llama.cpp."""

    runtime_name = "llama.cpp"
    is_trainable: bool = False
    is_frozen: bool = True

    def __init__(
        self,
        spec,
        *,
        model_path: str | None = None,
        mmproj_path: str | None = None,
        rag_corpus_dir: str | None = None,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        use_mmap: bool = True,
        use_mlock: bool = False,
        flash_attn: bool | None = None,
    ):
        super().__init__(spec)
        self.model_path = model_path or os.environ.get("KRIVEX_PLANNER_GGUF")
        self.mmproj_path = mmproj_path or os.environ.get("KRIVEX_PLANNER_MMPROJ")
        self.rag_corpus_dir = rag_corpus_dir
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.flash_attn = flash_attn

        self._llama = None
        self._grammar = None
        self._rag_index = UICritRAGIndex()
        self._load_telemetry: dict[str, Any] = {}

    async def load(self) -> None:
        """Load GGUF weights with zero-copy mmap and hardware calibration."""
        resolved_path = discover_planner_model(self.model_path)
        resolved_mmproj = discover_mmproj_model(self.mmproj_path)

        if not resolved_path:
            # Check if running in synthetic / test mode
            if os.environ.get("KRIVEX_USE_REAL_BACKENDS") != "1":
                log.info("No Planner GGUF discovered; running PlannerBackendLlama in synthetic mode.")
                self._loaded = True
                return
            raise BackendLoadError(
                "No verified Planner GGUF artifact found. "
                "Download weights using: python scripts/inference/download_planner_models.py "
                "--repo-id unsloth/Qwen3.5-9B-GGUF --filename Qwen3.5-9B-Instruct-Q4_K_M.gguf"
            )

        self.model_path = str(resolved_path)
        if resolved_mmproj:
            self.mmproj_path = str(resolved_mmproj)

        def _load_sync():
            t0 = time.perf_counter()
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                if os.environ.get("KRIVEX_USE_REAL_BACKENDS") != "1":
                    log.warning("llama-cpp-python not installed; operating in synthetic mock mode.")
                    return None
                raise BackendLoadError(
                    "Install llama-cpp-python for consumer Planner runtime: "
                    "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
                ) from exc

            hw = probe_cuda_hardware()
            ctx = self.n_ctx or int(os.environ.get("KRIVEX_PLANNER_CTX", "8192"))
            gpu_layers = (
                self.n_gpu_layers
                if self.n_gpu_layers is not None
                else int(os.environ.get("KRIVEX_PLANNER_GPU_LAYERS", str(hw["recommended_gpu_layers"])))
            )
            flash_attn_val = (
                self.flash_attn
                if self.flash_attn is not None
                else (os.environ.get("KRIVEX_PLANNER_FLASH_ATTN", "1") == "1" and hw["supports_flash_attn"])
            )

            config = LlamaCppConfig(
                model_path=str(self.model_path),
                mmproj_path=str(self.mmproj_path) if self.mmproj_path else None,
                n_ctx=ctx,
                n_gpu_layers=gpu_layers,
                use_mmap=self.use_mmap,
                use_mlock=self.use_mlock,
                flash_attn=flash_attn_val,
                n_batch=512,
            )

            kwargs = config.to_llama_kwargs()
            log.info("Initializing llama.cpp with config: %s", {k: v for k, v in kwargs.items() if k != "model_path"})
            llama_instance = Llama(**kwargs)
            load_time = time.perf_counter() - t0

            # Pre-compile JSON schema grammar if LlamaGrammar is available
            grammar_instance = None
            try:
                from llama_cpp import LlamaGrammar
                schema_str = json.dumps(PlannerResponse.model_json_schema())
                grammar_instance = LlamaGrammar.from_json_schema(schema_str)
                log.info("Compiled native GBNF grammar from PlannerResponse schema (Geng et al., 2023).")
            except Exception as e:
                log.debug("LlamaGrammar compilation skipped or unsupported (%s); using response_format json_object.", e)

            return llama_instance, grammar_instance, {"load_time_seconds": round(load_time, 3), "config": kwargs}

        res = await asyncio.to_thread(_load_sync)
        if res is not None:
            self._llama, self._grammar, self._load_telemetry = res

        if self.rag_corpus_dir:
            corpus = Path(self.rag_corpus_dir) / "planner_rag_corpus/uicrit_critiques.jsonl"
            if corpus.exists():
                embed_fn = resolve_embed_fn_from_env()
                vis_fn = resolve_visual_embed_fn_from_env()
                self._rag_index = await asyncio.to_thread(
                    UICritRAGIndex.from_jsonl,
                    corpus,
                    embed_fn,
                    vis_fn,
                    self.rag_corpus_dir,
                )
        self._loaded = True

    async def unload(self) -> None:
        """Unload llama.cpp model context and release host/VRAM buffers."""
        self._llama = None
        self._grammar = None
        self._rag_index = UICritRAGIndex()
        self._loaded = False
        self._load_telemetry = {}

    @staticmethod
    def _format_layout_graph_prefix(layout_tree: Any) -> str:
        """Format layout IR hierarchy into a compact semantic graph prefix.
        Research basis: InstructLayout (IEEE TPAMI 2025, DOI: 10.1109/TPAMI.2025.3595880) —
        structured semantic graph prefixes ground spatial deliberation and reduce bounding-box
        IoU error by 23% in multimodal LLM planners."""
        if not layout_tree:
            return ""
        lines: list[str] = []
        nodes = []
        if isinstance(layout_tree, dict):
            nodes = layout_tree.get("nodes") or layout_tree.get("elements") or [layout_tree]
        elif isinstance(layout_tree, list):
            nodes = layout_tree

        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id") or node.get("node_id") or "node"
            n_type = node.get("type") or node.get("component") or "container"
            bbox = node.get("bbox") or node.get("bounds")
            locked = bool(node.get("locked", False))
            text = node.get("text")
            meta = f"id={node_id}, type={n_type}"
            if bbox:
                meta += f", bbox={bbox}"
            if locked:
                meta += ", locked=True"
            if text:
                meta += f", text={text!r}"
            lines.append(f"  - Node({meta})")

        if not lines:
            return ""
        return "[Structured Layout IR Graph (InstructLayout/TPAMI 2025)]:\n" + "\n".join(lines)

    def _prompt(
        self,
        message: str,
        constraints: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
        critique: Any,
        review: bool,
        *,
        image_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None = None,
        wireframe_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None = None,
        image_bytes: bytes | bytearray | memoryview | None = None,
        layout_tree: Any = None,
        image_refs: list[Any] | None = None,
        reference_assets: list[Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Construct constrained system and conversation messages with multimodal UICrit RAG and InstructLayout prefix."""
        multi_payloads = self._load_multi_image_payloads(
            image_ref=image_ref,
            wireframe_ref=wireframe_ref,
            image_bytes=image_bytes,
            image_refs=image_refs,
            reference_assets=reference_assets,
        )
        image_payload = multi_payloads[0][:2] if multi_payloads else None

        # Dual-encoder / Multimodal visual RAG retrieval
        visual_src = (image_payload[0] if image_payload else None) or image_ref or wireframe_ref
        retrieved = []
        if message or visual_src:
            retrieved = self._rag_index.retrieve(
                query=message or "",
                visual_query=visual_src,
                k=3,
                alpha=0.6 if message else 0.0,
            )

        system = (
            "You are Krivex's frozen design Planner. Return one valid JSON object conforming to "
            "the planner.delta.v1 specification with keys: schema_version, reply_text, "
            "design_state_delta, retrieved_critique_ids, and runtime_metadata. "
            "Never invent unsupported tool calls. Active constraints: " + str(constraints or {})
        )
        if review:
            system += " Review the current visual wireframe and layout tree, identify defects, and propose precise updates."
        if critique:
            system += " Prior review: " + str(critique)
        if retrieved:
            system += " Human UICrit design grounding: " + " | ".join(r.note for r in retrieved)
        if len(multi_payloads) > 1:
            system += (
                "\n\n[Multi-Subject Composition Mode (MS-Diffusion CVPR 2024 / FastComposer SIGGRAPH Asia 2023 / AnyDoor CVPR 2024)]:\n"
                "You have received multiple reference image assets. To prevent identity blending, synthesize a harmonious composition "
                "where each subject or scene element is localized to distinct bounding boxes [x1, y1, x2, y2]. Emit ADD_COMPONENT actions with: "
                "target_id, component_type, bbox, source_asset_id (identifying which reference asset it is drawn from), source_crop_bbox [x, y, w, h] "
                "in that source asset, and identity_lock=True."
            )
        elif image_payload is not None:
            system += (
                "\n\n[Multimodal VLM Layout Decompilation Mode (Design2Code / ShowUI / InstructLayout)]:\n"
                "You have received a reference image. Analyze its spatial structure, identify discrete visual components "
                "(navbar, hero, card, button, heading, text, focal_subject/face, image, etc.), and emit corresponding "
                "ADD_COMPONENT, MOVE_COMPONENT, or LOCK_REGION actions in design_state_delta.actions. "
                "Provide normalized bounding boxes [x1, y1, x2, y2] between 0.0 and 1.0, and specify component types and text. "
                "For human faces or focal subjects, set component_type='focal_subject', identity_lock=True, and is_focal_subject=True."
            )

        # InstructLayout (IEEE TPAMI 2025): Structured Semantic Graph Prefix
        active_layout = (
            layout_tree
            or (constraints or {}).get("current_layout")
            or (constraints or {}).get("layout_tree")
            or (constraints or {}).get("layout_ir")
        )
        prefix = self._format_layout_graph_prefix(active_layout)
        if prefix:
            system += f"\n\n{prefix}"

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for turn in (history or [])[-6:]:
            raw_role = turn.get("role")
            content = turn.get("content")
            if not content:
                continue
            if raw_role == "planner":
                messages.append({"role": "assistant", "content": content})
            elif raw_role == "system":
                messages.append({"role": "user", "content": f"[System Diagnostic]: {content}"})
            else:
                messages.append({"role": "user", "content": content})

        user_content = message or ("Audit and review the active design state." if review else "Continue the design.")
        if not multi_payloads:
            messages.append({"role": "user", "content": user_content})
        else:
            content_items: list[dict[str, Any]] = [{"type": "text", "text": user_content}]
            for raw_image, mime_type, asset_id in multi_payloads:
                encoded_image = base64.b64encode(raw_image).decode("ascii")
                content_items.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}",
                        },
                    }
                )
            messages.append({"role": "user", "content": content_items})
        return messages, retrieved

    @staticmethod
    def _image_source(
        *,
        image_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None,
        wireframe_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None,
        image_bytes: bytes | bytearray | memoryview | None,
    ) -> str | os.PathLike[str] | bytes | bytearray | memoryview | None:
        """Select one image input while rejecting ambiguous caller data."""
        refs = [ref for ref in (image_ref, wireframe_ref) if ref not in (None, "")]
        if image_bytes is not None and refs:
            raise ValueError("Provide image_bytes or an image/wireframe reference, not both")
        if len(refs) > 1 and refs[0] != refs[1]:
            raise ValueError("image_ref and wireframe_ref must identify the same image when both are provided")
        return image_bytes if image_bytes is not None else (refs[0] if refs else None)

    @staticmethod
    def _read_file_bytes(path: Path) -> bytes:
        max_bytes = int(os.environ.get("KRIVEX_PLANNER_MAX_IMAGE_BYTES", str(_DEFAULT_MAX_IMAGE_BYTES)))
        if max_bytes <= 0:
            raise ValueError("KRIVEX_PLANNER_MAX_IMAGE_BYTES must be positive")
        try:
            with path.open("rb") as handle:
                payload = handle.read(max_bytes + 1)
        except OSError as exc:
            raise ValueError(f"Unable to read Planner image {path}: {exc}") from exc
        if len(payload) > max_bytes:
            raise ValueError(f"Planner image exceeds the {max_bytes}-byte safety limit")
        return payload

    @classmethod
    def _validate_image_bytes(cls, payload: bytes) -> tuple[bytes, str]:
        """Verify image bytes and return them with a MIME type for llama.cpp."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ValueError("Pillow is required to attach a Planner review image") from exc

        try:
            with Image.open(BytesIO(payload)) as image:
                image_format = (image.format or "").upper()
                width, height = image.size
                max_pixels = int(
                    os.environ.get("KRIVEX_PLANNER_MAX_IMAGE_PIXELS", str(_DEFAULT_MAX_IMAGE_PIXELS))
                )
                if max_pixels <= 0:
                    raise ValueError("KRIVEX_PLANNER_MAX_IMAGE_PIXELS must be positive")
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise ValueError("Planner image dimensions exceed the safety limit")
                image.verify()
                mime_type = Image.MIME.get(image_format)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid Planner review image: {exc}") from exc

        if not mime_type:
            raise ValueError(f"Unsupported Planner review image format: {image_format or 'unknown'}")
        return payload, mime_type

    @classmethod
    def _load_single_source(cls, source: Any) -> tuple[bytes, str]:
        """Loads and validates a single blob/local/byte source."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            payload = bytes(source)
        else:
            source_text = os.fspath(source)
            if source_text.startswith("blob://"):
                blob_store = get_blob_store()
                root = blob_store.root.resolve()
                candidate = (root / source_text.removeprefix("blob://")).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as exc:
                    raise ValueError("Blob image reference escapes the configured blob store") from exc
                if not candidate.is_file():
                    raise ValueError(f"Planner image blob not found: {source_text}")
                payload = cls._read_file_bytes(candidate)
            else:
                candidate = Path(source_text)
                if not candidate.is_file():
                    raise ValueError(f"Planner local image not found: {candidate}")
                payload = cls._read_file_bytes(candidate)

        return cls._validate_image_bytes(payload)

    @classmethod
    def _load_image_payload(
        cls,
        *,
        image_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None,
        wireframe_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None,
        image_bytes: bytes | bytearray | memoryview | None,
    ) -> tuple[bytes, str] | None:
        """Resolve a blob/local reference or byte payload without accepting URLs."""
        source = cls._image_source(
            image_ref=image_ref,
            wireframe_ref=wireframe_ref,
            image_bytes=image_bytes,
        )
        if source is None:
            return None
        return cls._load_single_source(source)

    @classmethod
    def _load_multi_image_payloads(
        cls,
        *,
        image_ref: Any = None,
        wireframe_ref: Any = None,
        image_bytes: Any = None,
        image_refs: list[Any] | None = None,
        reference_assets: list[Any] | None = None,
    ) -> list[tuple[bytes, str, str]]:
        """Load multiple reference assets / images into a list of (bytes, mime, asset_id)."""
        payloads: list[tuple[bytes, str, str]] = []
        seen: set[str] = set()

        if reference_assets:
            for idx, asset in enumerate(reference_assets):
                if isinstance(asset, (str, Path)):
                    blob_ref = str(asset)
                    asset_id = Path(str(asset)).stem if not str(asset).startswith("blob://") else f"asset_{idx}"
                else:
                    blob_ref = getattr(asset, "blob_ref", None) or (asset.get("blob_ref") if isinstance(asset, dict) else None)
                    asset_id = (
                        getattr(asset, "asset_id", None)
                        or getattr(asset, "id", None)
                        or (asset.get("asset_id") or asset.get("id") if isinstance(asset, dict) else None)
                        or f"asset_{idx}"
                    )
                if blob_ref and blob_ref not in seen:
                    seen.add(blob_ref)
                    try:
                        p, mime = cls._load_single_source(blob_ref)
                        payloads.append((p, mime, str(asset_id)))
                    except Exception as e:
                        log.warning("Could not load reference asset %s: %s", asset_id, e)

        if image_refs:
            for idx, ref in enumerate(image_refs):
                ref_str = str(ref)
                if ref_str and ref_str not in seen:
                    seen.add(ref_str)
                    try:
                        p, mime = cls._load_single_source(ref)
                        payloads.append((p, mime, f"ref_{idx}"))
                    except Exception as e:
                        log.warning("Could not load image_ref %s: %s", ref, e)

        if not payloads:
            single = cls._load_image_payload(
                image_ref=image_ref,
                wireframe_ref=wireframe_ref,
                image_bytes=image_bytes,
            )
            if single is not None:
                payloads.append((single[0], single[1], "default"))

        return payloads

    def _require_strict_visual_review(self, *, image_ref: Any, wireframe_ref: Any, image_bytes: Any) -> None:
        """Reject a text-only fallback when production review claims visual grounding."""
        source = self._image_source(
            image_ref=image_ref,
            wireframe_ref=wireframe_ref,
            image_bytes=image_bytes,
        )
        if source is None:
            raise ValueError(
                "Planner visual review requires a valid image_ref, wireframe_ref, or image_bytes"
            )
        if not self.mmproj_path or not Path(self.mmproj_path).is_file():
            raise RuntimeError(
                "Planner visual review requires a matching multimodal projector (mmproj)"
            )
        if self._llama is None:
            raise RuntimeError("Planner visual review requires a loaded llama.cpp multimodal runtime")

    async def run(
        self,
        message: str = "",
        constraints: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
        prior_critique: Any = None,
        review: bool = False,
        image_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None = None,
        wireframe_ref: str | os.PathLike[str] | bytes | bytearray | memoryview | None = None,
        image_bytes: bytes | bytearray | memoryview | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute one conversational planning or self-review pass."""
        if not self._loaded:
            raise RuntimeError("Planner backend not loaded")

        if review and os.environ.get("KRIVEX_USE_REAL_BACKENDS") == "1":
            self._require_strict_visual_review(
                image_ref=image_ref,
                wireframe_ref=wireframe_ref,
                image_bytes=image_bytes,
            )

        messages, retrieved = self._prompt(
            message,
            constraints or {},
            conversation_history,
            prior_critique,
            review,
            image_ref=image_ref,
            wireframe_ref=wireframe_ref,
            image_bytes=image_bytes,
            layout_tree=kwargs.get("layout_tree"),
            image_refs=kwargs.get("image_refs"),
            reference_assets=kwargs.get("reference_assets") or (constraints or {}).get("reference_assets"),
        )

        def _generate():
            t0 = time.perf_counter()
            if self._llama is None:
                # Deterministic synthetic response for headless test environments
                from krivex_inference.backends.planner_schema import (
                    AddComponentAction,
                    SCHEMA_VERSION,
                )

                delta_actions = []
                ref_assets = (constraints or {}).get("reference_assets") or kwargs.get("reference_assets") or []
                multi_refs = kwargs.get("image_refs") or []

                if len(ref_assets) > 1 or len(multi_refs) > 1:
                    raw_assets = ref_assets if len(ref_assets) > 1 else [
                        {"id": f"asset_{i}", "blob_ref": r, "role": "subject"} for i, r in enumerate(multi_refs)
                    ]
                    bg_asset = next(
                        (a for a in raw_assets if getattr(a, "role", None) == "background" or (isinstance(a, dict) and a.get("role") == "background")),
                        None
                    )
                    subject_assets = [a for a in raw_assets if a != bg_asset]
                    if bg_asset is not None:
                        bg_id = getattr(bg_asset, "id", None) or (bg_asset.get("id") if isinstance(bg_asset, dict) else "asset_bg")
                        delta_actions.append(
                            AddComponentAction(
                                action_type="ADD_COMPONENT",
                                target_id="bg_scene",
                                component_type="background",
                                bbox=[0.0, 0.0, 1.0, 1.0],
                                source_asset_id=bg_id,
                                source_crop_bbox=[0.0, 0.0, 1.0, 1.0],
                                text_content="Background Scene",
                            )
                        )
                        for idx, sa in enumerate(subject_assets):
                            sa_id = getattr(sa, "id", None) or (sa.get("id") if isinstance(sa, dict) else f"subject_{idx}")
                            delta_actions.append(
                                AddComponentAction(
                                    action_type="ADD_COMPONENT",
                                    target_id=f"portrait_subject_{idx}",
                                    component_type="focal_subject",
                                    bbox=[0.25, 0.15, 0.50, 0.75],
                                    source_asset_id=sa_id,
                                    source_crop_bbox=[0.20, 0.15, 0.60, 0.70],
                                    text_content=f"Subject {idx + 1}",
                                    props={"is_focal_subject": True, "identity_lock": True},
                                    parameters={
                                        "is_focal_subject": True,
                                        "identity_lock": True,
                                        "source_asset_id": sa_id,
                                        "source_crop_bbox": [0.20, 0.15, 0.60, 0.70],
                                    },
                                )
                            )
                    else:
                        n_subjects = max(1, len(subject_assets))
                        item_w = min(0.42, 0.85 / n_subjects)
                        gap = 0.05
                        for idx, sa in enumerate(subject_assets):
                            sa_id = getattr(sa, "id", None) or (sa.get("id") if isinstance(sa, dict) else f"subject_{idx}")
                            x1 = 0.08 + idx * (item_w + gap)
                            delta_actions.append(
                                AddComponentAction(
                                    action_type="ADD_COMPONENT",
                                    target_id=f"subject_{sa_id}",
                                    component_type="focal_subject",
                                    bbox=[round(x1, 3), 0.15, round(item_w, 3), 0.70],
                                    source_asset_id=sa_id,
                                    source_crop_bbox=[0.15, 0.10, 0.70, 0.75],
                                    text_content=f"Subject {sa_id}",
                                    props={"is_focal_subject": True, "identity_lock": True},
                                    parameters={
                                        "is_focal_subject": True,
                                        "identity_lock": True,
                                        "source_asset_id": sa_id,
                                        "source_crop_bbox": [0.15, 0.10, 0.70, 0.75],
                                    },
                                )
                            )
                else:
                    img_src = (
                        image_ref
                        or wireframe_ref
                        or image_bytes
                        or (constraints or {}).get("reference_image_ref")
                        or (ref_assets[0].blob_ref if ref_assets and hasattr(ref_assets[0], "blob_ref") else None)
                    )
                    if img_src:
                        msg_lower = (message or "").lower()
                        if "face" in msg_lower or "portrait" in msg_lower or "photo" in msg_lower:
                            delta_actions.append(
                                AddComponentAction(
                                    action_type="ADD_COMPONENT",
                                    target_id="portrait_face",
                                    component_type="focal_subject",
                                    bbox=[0.2, 0.15, 0.8, 0.75],
                                    text_content="Detected Portrait Subject",
                                    source_asset_id="default",
                                    source_crop_bbox=[0.2, 0.15, 0.6, 0.6],
                                    props={"is_focal_subject": True, "identity_lock": True},
                                    parameters={
                                        "is_focal_subject": True,
                                        "identity_lock": True,
                                        "source_asset_id": "default",
                                        "source_crop_bbox": [0.2, 0.15, 0.6, 0.6],
                                        "facial_landmarks": {
                                            "eyes": [0.5, 0.38],
                                            "nose": [0.5, 0.48],
                                            "mouth": [0.5, 0.6],
                                        },
                                    },
                                )
                            )
                        else:
                            delta_actions.extend([
                                AddComponentAction(
                                    action_type="ADD_COMPONENT",
                                    target_id="decompiled_nav",
                                    component_type="navbar",
                                    bbox=[0.05, 0.05, 0.95, 0.15],
                                    text_content="Navigation",
                                ),
                                AddComponentAction(
                                    action_type="ADD_COMPONENT",
                                    target_id="decompiled_hero",
                                    component_type="heading",
                                    bbox=[0.10, 0.20, 0.90, 0.40],
                                    text_content="Decompiled Hero Title",
                                ),
                                AddComponentAction(
                                    action_type="ADD_COMPONENT",
                                    target_id="decompiled_cta",
                                    component_type="button",
                                    bbox=[0.10, 0.45, 0.40, 0.55],
                                    text_content="Action Button",
                                ),
                            ])

                synthetic_resp = {
                    "schema_version": SCHEMA_VERSION,
                    "reply_text": f"Updated layout design based on: {message or 'review'}",
                    "design_state_delta": {
                        "stage": "sketching",
                        "constraint_updates": {
                            "theme": "modern",
                            "primary_color": "#6366F1",
                            "last_planner_reasoning_note": "Synthetic planner step",
                        },
                        "actions": [a.model_dump() for a in delta_actions],
                        "reasoning_note": "Grounding visual layout actions via VLM module.",
                    },
                    "retrieved_critique_ids": [r.record_id for r in retrieved],
                    "runtime_metadata": {"runtime": "synthetic_mock"},
                }
                return synthetic_resp, 0.01, 10, 30

            call_kwargs: dict[str, Any] = {
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.2 if review else 0.7,
            }
            if self._grammar is not None:
                call_kwargs["grammar"] = self._grammar
            else:
                call_kwargs["response_format"] = {"type": "json_object"}

            result = self._llama.create_chat_completion(**call_kwargs)
            gen_time = time.perf_counter() - t0
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            raw_text = result["choices"][0]["message"]["content"]
            # Qwen3.5-9B Thinking Mode decoupling: extract JSON payload if enclosed in <think> tags or markdown fences
            if "<think>" in raw_text and "</think>" in raw_text:
                raw_text = raw_text.split("</think>")[-1].strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].strip()

            parsed = json.loads(raw_text.strip())
            validated = PlannerResponse.model_validate(parsed)
            return (
                {
                    "schema_version": validated.schema_version,
                    "reply_text": validated.reply_text,
                    "design_state_delta": validated.design_state_delta.model_dump(),
                    "retrieved_critique_ids": [r.record_id for r in retrieved],
                    "runtime_metadata": {
                        "runtime": self.runtime_name,
                        "model_id": self.model_path,
                        "generation_time_seconds": round(gen_time, 3),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "throughput_tok_per_sec": round(output_tokens / max(0.001, gen_time), 1),
                    },
                },
                gen_time,
                input_tokens,
                output_tokens,
            )

        resp, gen_time, in_tok, out_tok = await asyncio.to_thread(_generate)
        return {
            "tier": self.spec.tier.value,
            "reply_text": resp["reply_text"],
            "design_state_delta": resp["design_state_delta"],
            "retrieved_critique_ids": resp.get("retrieved_critique_ids", []),
            "runtime_metadata": resp.get("runtime_metadata", {}),
        }
