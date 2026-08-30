"""Configuration loading, validation, and environment overlay for data-forge."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml


def _default_data_root() -> Path:
    """Return platform-appropriate default DATA_ROOT."""
    if platform.system() == "Windows":
        return Path(r"D:\data_krisna")
    return Path("/data_krisna")


@dataclass
class VLLMServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = "data-forge-internal"
    startup_timeout_seconds: int = 120
    health_check_interval_seconds: int = 5
    graceful_shutdown_timeout_seconds: int = 30


@dataclass
class ModelSpec:
    model_id: str
    revision: str = "main"
    quantization: str | None = None
    dtype: str = "auto"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = True
    role: str = ""
    load_on_demand: bool = False
    capabilities: list[str] = field(default_factory=list)
    vram_estimate_gb: float = 0.0


@dataclass
class EncoderSpec:
    model_id: str
    revision: str = "main"
    dtype: str = "float16"
    device: str = "cuda"
    role: str = ""
    vram_estimate_gb: float = 0.0
    expected_channels: int | None = None
    expected_downsample: int | None = None


@dataclass
class DatasetSpec:
    display_name: str
    source_type: str  # "huggingface", "github", "url"
    category: str
    license_status: str = "unverified"
    license_url: str | None = None
    license_notes: str | None = None
    expected_record_count: int = 0
    # HuggingFace-specific
    repo_id: str | None = None
    revision: str | None = "main"
    subset: str | None = None
    # GitHub-specific
    repo_url: str | None = None
    branch: str | None = "main"
    # Fetch settings
    fetch_config: dict[str, Any] = field(default_factory=dict)
    note: str | None = None
    # Set true for sources whose value is annotations to JOIN against
    # already-ingested images (e.g. UICrit's ratings against RICO screens),
    # not standalone images of their own. Prevents the generic image-glob
    # fetch path from creating duplicate/meaningless image records for a
    # repo that's really just annotation files — see uicrit_ingest.py.
    annotation_only: bool = False

    # True for sources that exist purely to benchmark against (TASTE,
    # PartiPrompts) — fetched via download_mode "eval_reference" straight
    # into heldout/external_eval/ (see fetcher.py::_fetch_eval_reference)
    # and never eligible for training_pool or preference_pairs/ under any
    # stage configuration. Kept as an explicit field (not inferred from
    # download_mode alone) so anything auditing datasets.yaml can answer
    # "is this training data?" without knowing fetch-path internals.
    eval_only: bool = False

    # Download modes whose fetch path never inserts a standalone image
    # record into the main manifest at all: `annotation_only` sources
    # (UICrit — joins onto existing RICO records), `caption_join`'s
    # realistic path (Screen2Words — joins captions onto existing RICO
    # records; see fetcher.py::_fetch_huggingface_caption_join),
    # `preference_pair` (Pick-a-Pic v2/HPDv2/DesignSense-10k/DesignPref —
    # written directly to preference_pairs/, see
    # fetcher.py::_fetch_huggingface_preference_pairs), and
    # `eval_reference` (TASTE/PartiPrompts — written directly to
    # heldout/external_eval/, never touches training data at all). These
    # consume ~0 of the per-record image-storage budget in pipeline.yaml's
    # `per_record_estimates`, which assumes a raw+scrubbed+latent+vq+
    # control+metadata image record — projecting full storage for them
    # would double-count storage that either doesn't exist (annotations
    # are tiny compared to per_record_estimates) or was already counted
    # under the dataset they join onto (rico_core/rico_semantic), or is
    # tracked separately under its own preference-pair/eval storage
    # estimate instead.
    _ZERO_IMAGE_STORAGE_MODES: ClassVar[frozenset[str]] = frozenset(
        {"caption_join", "preference_pair", "eval_reference"}
    )

    def storage_relevant_record_count(self) -> int:
        """Records this dataset actually contributes to the per-record
        image-storage projection in StorageManager.calculate_projected_size.

        BUG FIX: the pre-flight check previously summed every dataset's
        raw `expected_record_count` — PD12M's full 12.4M, CC12M's full
        12.4M, etc. — completely ignoring `fetch_config.sample_size`,
        which actually caps what gets downloaded (200K/150K respectively).
        That inflated the projected corpus from the PRD's real ~100K-500K
        target (see PRD §8.3, "3TB budget is generous at this scale") to
        ~26M records / ~33TB, which would false-fail `pre_flight_check`
        against any realistic single-workstation disk before Stage 1 ever
        ran. This method is the single place that reconciles the two:
        respect `sample_size` when set, and zero out sources whose fetch
        path doesn't produce a standalone image record at all (see
        `_ZERO_IMAGE_STORAGE_MODES` and `annotation_only` above).
        """
        if self.annotation_only:
            return 0
        download_mode = self.fetch_config.get("download_mode")
        if download_mode in self._ZERO_IMAGE_STORAGE_MODES:
            return 0
        sample_size = self.fetch_config.get("sample_size")
        if sample_size is not None:
            return min(self.expected_record_count, int(sample_size))
        return self.expected_record_count

    def preference_pair_relevant_count(self) -> int:
        """Records this dataset contributes to the preference-pair storage
        projection specifically — the counterpart to
        storage_relevant_record_count() for `download_mode:
        "preference_pair"` sources, which are excluded from that method
        (see _ZERO_IMAGE_STORAGE_MODES) because a pair costs a different,
        larger amount of storage than a single image record (two source
        images + two latents, tracked via
        storage.per_record_estimates.preference_pair_*). Mixing the two
        into one count would either under- or over-project depending on
        which per-record constant got applied to it.
        """
        if self.fetch_config.get("download_mode") != "preference_pair":
            return 0
        sample_size = self.fetch_config.get("sample_size")
        if sample_size is not None:
            return min(self.expected_record_count, int(sample_size))
        return self.expected_record_count


@dataclass
class PathsConfig:
    raw: str = "raw/"
    scrubbed: str = "scrubbed/"
    processed_root: str = "processed/"
    latents_zimage: str = "processed/latents_zimage/"
    vq_tokens_sketch: str = "processed/vq_tokens_sketch/"
    control_tokens: str = "processed/control_tokens/"
    training_pool: str = "training_pool/"
    heldout: str = "heldout/"
    manifests: str = "manifests/"
    checkpoints: str = ".checkpoints/"
    logs: str = "logs/"
    registry_reports: str = "registry_reports/"
    audit_reports: str = "audit_reports/"
    compliance_briefs: str = "compliance_briefs/"
    # REMOVED (data-pipeline upgrade): latents_qwenimage/, edit_pairs/, and
    # planner_data/conversations/ all existed to feed training jobs that no
    # longer exist. Qwen-Image-Edit-2511 and the Qwen3.5-9B planner now ship
    # frozen (zero-shot ICL / SDEdit inference for the former, RAG over
    # UICrit's real critique text for the latter) — see PRD "no-RLHF-loop"
    # revision. Deleting these fields is deliberate, not an oversight: their
    # only two producers (s07_5_edit_pairs.py, s01_6_planner_synthesis.py)
    # and only consumer (s12_model_data_export.py's old per-model exporters)
    # were removed in the same pass. See docs/DATA_COMPLETENESS.md.
    ui_critique: str = "ui_critique/"
    preference_pairs: str = "preference_pairs/"         # raw pairs, per-source subfolders
    dpo_latents: str = "processed/dpo_latents/"          # encoded pairs, ready for Diffusion-DPO
    model_data_root: str = "model_data/"                 # Final per-model segmented export

    def resolve(self, data_root: Path) -> dict[str, Path]:
        """Resolve all paths relative to DATA_ROOT, creating dirs as needed."""
        resolved: dict[str, Path] = {}
        for name, rel_path in vars(self).items():
            full_path = data_root / rel_path
            full_path.mkdir(parents=True, exist_ok=True)
            resolved[name] = full_path
        return resolved


@dataclass
class StageConfig:
    """Per-stage threshold and parameter container."""

    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass
class StorageConfig:
    safety_margin: float = 0.10
    per_record_estimates: dict[str, int] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level configuration container for the entire pipeline."""

    version: str = "0.13.0"
    dataset_version_prefix: str = "krisna_v"
    data_root: Path = field(default_factory=_default_data_root)
    paths: PathsConfig = field(default_factory=PathsConfig)
    resolved_paths: dict[str, Path] = field(default_factory=dict)

    # Orchestrator
    chunk_size: int = 10000
    max_retries_per_record: int = 2
    fail_fast: bool = False
    checkpoint_enabled: bool = True

    # Models
    models: dict[str, ModelSpec] = field(default_factory=dict)
    encoders: dict[str, EncoderSpec] = field(default_factory=dict)
    vllm_server: VLLMServerConfig = field(default_factory=VLLMServerConfig)

    # Datasets
    datasets: dict[str, DatasetSpec] = field(default_factory=dict)

    # Stages
    stages: dict[str, StageConfig] = field(default_factory=dict)

    # Storage
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Prompts directory
    prompts_dir: Path = field(default_factory=lambda: Path("configs/prompts"))
    schemas_dir: Path = field(default_factory=lambda: Path("configs/schemas"))

    def get_prompt(self, name: str) -> str:
        """Load a prompt template by name (without extension)."""
        prompt_file = self.prompts_dir / f"{name}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8")

    def get_stage(self, stage_name: str) -> StageConfig:
        """Get stage config, returning a disabled default if not configured."""
        return self.stages.get(stage_name, StageConfig(enabled=False))


def _parse_model_spec(data: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        model_id=data["model_id"],
        revision=data.get("revision", "main"),
        quantization=data.get("quantization"),
        dtype=data.get("dtype", "auto"),
        max_model_len=data.get("max_model_len", 32768),
        gpu_memory_utilization=data.get("gpu_memory_utilization", 0.85),
        trust_remote_code=data.get("trust_remote_code", True),
        role=data.get("role", ""),
        load_on_demand=data.get("load_on_demand", False),
        capabilities=data.get("capabilities", []),
        vram_estimate_gb=data.get("vram_estimate_gb", 0.0),
    )


def _parse_encoder_spec(data: dict[str, Any]) -> EncoderSpec:
    return EncoderSpec(
        model_id=data["model_id"],
        revision=data.get("revision", "main"),
        dtype=data.get("dtype", "float16"),
        device=data.get("device", "cuda"),
        role=data.get("role", ""),
        vram_estimate_gb=data.get("vram_estimate_gb", 0.0),
        expected_channels=data.get("expected_channels"),
        expected_downsample=data.get("expected_downsample"),
    )


def _parse_dataset_spec(key: str, data: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        display_name=data.get("display_name", key),
        source_type=data.get("source_type", "huggingface"),
        category=data.get("category", "unknown"),
        license_status=data.get("license_status", "unverified"),
        license_url=data.get("license_url"),
        license_notes=data.get("license_notes"),
        expected_record_count=data.get("expected_record_count", 0),
        repo_id=data.get("repo_id"),
        revision=data.get("revision", "main"),
        subset=data.get("subset"),
        repo_url=data.get("repo_url"),
        branch=data.get("branch", "main"),
        fetch_config=data.get("fetch_config", {}),
        note=data.get("note"),
        annotation_only=data.get("annotation_only", False),
        eval_only=data.get("eval_only", False),
    )


def _parse_paths(data: dict[str, Any]) -> PathsConfig:
    pc = PathsConfig()
    if "raw" in data:
        pc.raw = data["raw"]
    if "scrubbed" in data:
        pc.scrubbed = data["scrubbed"]
    processed = data.get("processed", {})
    if isinstance(processed, dict):
        pc.processed_root = processed.get("root", pc.processed_root)
        pc.latents_zimage = processed.get("latents_zimage", pc.latents_zimage)
        pc.vq_tokens_sketch = processed.get("vq_tokens_sketch", pc.vq_tokens_sketch)
        pc.control_tokens = processed.get("control_tokens", pc.control_tokens)
        pc.dpo_latents = processed.get("dpo_latents", pc.dpo_latents)
    for key in (
        "training_pool", "heldout", "manifests", "checkpoints",
        "logs", "registry_reports", "audit_reports", "compliance_briefs",
        "ui_critique", "preference_pairs",
    ):
        if key in data:
            setattr(pc, key, data[key])
    return pc


def load_config(
    pipeline_yaml: str | Path = "configs/pipeline.yaml",
    models_yaml: str | Path = "configs/models.yaml",
    datasets_yaml: str | Path = "configs/datasets.yaml",
) -> PipelineConfig:
    """Load and merge all YAML configs into a single PipelineConfig.

    Environment variable overrides:
      - DATA_ROOT: base path for all data
      - HF_TOKEN: HuggingFace auth (validated at CLI startup, not here)
      - CUDA_VISIBLE_DEVICES: GPU selection
    """
    config = PipelineConfig()

    # --- DATA_ROOT from env ---
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        config.data_root = Path(env_root)

    # --- pipeline.yaml ---
    pipeline_path = Path(pipeline_yaml)
    if pipeline_path.exists():
        with open(pipeline_path, encoding="utf-8") as f:
            pdata = yaml.safe_load(f) or {}

        top = pdata.get("pipeline", {})
        config.version = top.get("version", config.version)
        config.dataset_version_prefix = top.get(
            "dataset_version_prefix", config.dataset_version_prefix
        )

        if "paths" in pdata:
            config.paths = _parse_paths(pdata["paths"])

        orch = pdata.get("orchestrator", {})
        config.chunk_size = orch.get("chunk_size", config.chunk_size)
        config.max_retries_per_record = orch.get(
            "max_retries_per_record", config.max_retries_per_record
        )
        config.fail_fast = orch.get("fail_fast", config.fail_fast)
        config.checkpoint_enabled = orch.get("checkpoint_enabled", config.checkpoint_enabled)

        for stage_key, stage_data in pdata.get("stages", {}).items():
            if isinstance(stage_data, dict):
                enabled = stage_data.pop("enabled", True)
                config.stages[stage_key] = StageConfig(enabled=enabled, params=stage_data)

        storage_data = pdata.get("storage", {})
        config.storage = StorageConfig(
            safety_margin=storage_data.get("safety_margin", 0.10),
            per_record_estimates=storage_data.get("per_record_estimates", {}),
        )

    # --- models.yaml ---
    models_path = Path(models_yaml)
    if models_path.exists():
        with open(models_path, encoding="utf-8") as f:
            mdata = yaml.safe_load(f) or {}

        for key, spec_data in mdata.get("models", {}).items():
            if isinstance(spec_data, dict) and "model_id" in spec_data:
                config.models[key] = _parse_model_spec(spec_data)

        for key, spec_data in mdata.get("encoders", {}).items():
            if isinstance(spec_data, dict) and "model_id" in spec_data:
                config.encoders[key] = _parse_encoder_spec(spec_data)

        vllm_data = mdata.get("vllm_server", {})
        if vllm_data:
            config.vllm_server = VLLMServerConfig(
                host=vllm_data.get("host", "127.0.0.1"),
                port=vllm_data.get("port", 8000),
                api_key=vllm_data.get("api_key", "data-forge-internal"),
                startup_timeout_seconds=vllm_data.get("startup_timeout_seconds", 120),
                health_check_interval_seconds=vllm_data.get("health_check_interval_seconds", 5),
                graceful_shutdown_timeout_seconds=vllm_data.get(
                    "graceful_shutdown_timeout_seconds", 30
                ),
            )

    # --- datasets.yaml ---
    datasets_path = Path(datasets_yaml)
    if datasets_path.exists():
        with open(datasets_path, encoding="utf-8") as f:
            ddata = yaml.safe_load(f) or {}

        for key, ds_data in ddata.get("datasets", {}).items():
            if isinstance(ds_data, dict):
                config.datasets[key] = _parse_dataset_spec(key, ds_data)

    # --- Resolve paths ---
    config.resolved_paths = config.paths.resolve(config.data_root)

    # --- Resolve prompts/schemas dirs relative to pipeline.yaml location ---
    if pipeline_path.exists():
        base_dir = pipeline_path.parent.parent  # configs/ → project root
        config.prompts_dir = base_dir / "configs" / "prompts"
        config.schemas_dir = base_dir / "configs" / "schemas"

    return config
