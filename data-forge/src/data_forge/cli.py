"""CLI entry point for data-forge.

Commands:
  data-forge run [--dry-run] [--resume] [--stages 0,1,2]
  data-forge doctor
  data-forge registry check
  data-forge manifest stats
  data-forge manifest query --status <status>
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
import sys
from contextlib import suppress
from pathlib import Path

if sys.platform == 'win32':
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "reconfigure"):
        with suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        with suppress(Exception):
            sys.stderr.reconfigure(encoding="utf-8")

import click
from rich.console import Console
from rich.table import Table

from data_forge.config import PipelineConfig

console = Console(legacy_windows=False if sys.platform == "win32" else None)


def _load_env_file() -> None:
    """Load key-value pairs from .env into os.environ if not already set."""
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
                break
            except Exception:
                pass


_load_env_file()


def _validate_environment() -> None:
    """Pre-flight environment validation."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        console.print(
            "[bold yellow]WARNING:[/] HF_TOKEN environment variable not set.\n"
            "Public open-source datasets (PD12M, CC12M, Rico, HPDv2, GameLabel-10K, TASTE, etc.)\n"
            "will download without authentication. Gated datasets will be skipped.",
            highlight=False,
        )
        return

    # Validate HF token has read access via whoami-v2
    try:
        import httpx
        resp = httpx.get("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {hf_token}"}, timeout=5)
        resp.raise_for_status()
        console.print("[green]✓[/] HuggingFace token validated")
    except Exception as e:
        console.print(f"[bold yellow]WARNING:[/] Could not validate HF token: {e}")


def _find_configs_dir() -> Path:
    """Locate data-forge/configs directory robustly across invocation locations."""
    candidates = [
        Path.cwd() / "configs",
        Path.cwd() / "data-forge" / "configs",
        Path(__file__).resolve().parent.parent / "configs",
        Path(__file__).resolve().parent.parent.parent / "configs",
        Path(__file__).resolve().parent.parent.parent / "data-forge" / "configs",
        Path(__file__).resolve().parent.parent.parent.parent / "data-forge" / "configs",
    ]
    for c in candidates:
        if (c / "pipeline.yaml").is_file():
            return c
    return Path.cwd() / "configs"


def _load_pipeline_config(
    profile: str = "default",
    *,
    pipeline_yaml: str | Path | None = None,
    models_yaml: str | Path | None = None,
    datasets_yaml: str | Path | None = None,
) -> PipelineConfig:
    from data_forge.config import load_config

    configs_dir = _find_configs_dir()
    if profile in ("default", "local-vlm"):
        default_pipeline = configs_dir / "pipeline.yaml"
        default_models = configs_dir / "models.yaml"
        default_datasets = configs_dir / "datasets.yaml"
    else:
        raise click.ClickException(f"Unknown config profile: {profile}")

    pipeline_path = pipeline_yaml or os.environ.get("KRIVEX_PIPELINE_YAML") or default_pipeline
    models_path = models_yaml or os.environ.get("KRIVEX_MODELS_YAML") or default_models
    datasets_yaml_path = datasets_yaml or os.environ.get("KRIVEX_DATASETS_YAML") or default_datasets
    config = load_config(
        pipeline_yaml=pipeline_path,
        models_yaml=models_path,
        datasets_yaml=datasets_yaml_path,
    )
    return config


def _require_local_vlm_hardware(config: PipelineConfig) -> None:
    """Fail before model startup when the strict local contract is impossible."""
    import importlib.util

    if importlib.util.find_spec("vllm") is None:
        raise click.ClickException(
            "Strict local-VLM preprocessing requires vLLM; install the GPU runtime before starting."
        )
    if not config.strict_vlm_no_fallback:
        raise click.ClickException(
            "The selected profile is not strict_vlm_no_fallback=True; refusing to run a fallback-capable preprocessing path."
        )
    model = config.models.get(config.local_vlm.model_key)
    if model is None or "vision" not in {str(cap).lower() for cap in model.capabilities}:
        raise click.ClickException(
            f"Configured local VLM '{config.local_vlm.model_key}' is missing or has no vision capability."
        )
    if not isinstance(model.revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", model.revision) is None:
        raise click.ClickException(
            f"Strict local-VLM model revision must be an immutable 40-character SHA, got {model.revision!r}."
        )
    try:
        import torch
    except ImportError as exc:
        raise click.ClickException("Strict local-VLM preprocessing requires a CUDA-enabled PyTorch install.") from exc
    if config.local_vlm.require_cuda and not torch.cuda.is_available():
        raise click.ClickException(
            "Strict local-VLM preprocessing requires CUDA; CPU execution and model fallback are disabled."
        )
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_gb < config.local_vlm.gpu_memory_budget_gb:
            raise click.ClickException(
                f"GPU has {total_gb:.1f} GiB, below the configured strict local-VLM budget "
                f"of {config.local_vlm.gpu_memory_budget_gb} GiB."
            )
    try:
        import psutil

        host_ram_gb = psutil.virtual_memory().total / (1024**3)
        if host_ram_gb < config.local_vlm.host_ram_budget_gb:
            raise click.ClickException(
                f"Host has {host_ram_gb:.1f} GiB RAM, below the configured strict local-VLM "
                f"budget of {config.local_vlm.host_ram_budget_gb} GiB."
            )
    except ImportError as exc:
        raise click.ClickException("Strict local-VLM preprocessing requires psutil for host-memory validation.") from exc


@click.group()
@click.version_option(version="0.14.0", prog_name="data-forge")
def main() -> None:
    """Data-Forge: Zero-touch data pipeline for the Krivex project (v14)."""
    import multiprocessing
    import sys
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn', force=True)


@main.command()
@click.option("--dry-run", is_flag=True, help="Validate config and walk stages without inference")
@click.option("--resume/--no-resume", default=True, help="Resume from checkpoints")
@click.option("--profile", type=click.Choice(["default", "local-vlm"]), default="default", show_default=True,
              help="Configuration profile; default is the server Qwen3.8-27B GPU pipeline profile")
@click.option("--stages", default=None, help="Comma-separated stage numbers to run (e.g., '0,1,2')")
@click.option("--chunk-size", default=None, type=int, help="Override chunk size from config")
@click.option("--limit", default=None, type=int, help="Cap total records processed this run (smoke tests)")
@click.option("--datasets-yaml", default=None, type=click.Path(exists=True, path_type=Path), help="Path to datasets.yaml")
@click.option("--pipeline-yaml", default=None, type=click.Path(exists=True, path_type=Path), help="Path to pipeline.yaml")
@click.option("--models-yaml", default=None, type=click.Path(exists=True, path_type=Path), help="Path to models.yaml")
@click.option("--log-level", default="INFO", help="Logging level")
def run(
    dry_run: bool,
    resume: bool,
    profile: str,
    stages: str | None,
    chunk_size: int | None,
    limit: int | None,
    datasets_yaml: Path | None,
    pipeline_yaml: Path | None,
    models_yaml: Path | None,
    log_level: str,
) -> None:
    """Execute the data pipeline."""
    from data_forge.logging_setup import setup_logging

    config = _load_pipeline_config(
        profile,
        pipeline_yaml=pipeline_yaml,
        datasets_yaml=datasets_yaml,
        models_yaml=models_yaml,
    )
    config.dry_run = dry_run

    if chunk_size:
        config.chunk_size = chunk_size

    setup_logging(config.resolved_paths["logs"], log_level)

    if not dry_run:
        _validate_environment()

    # Parse stage filter
    stages_filter: list[str] | None = None
    if stages:
        from data_forge.stages import STAGE_MODULES

        stage_map = {
            "0": "s00_manifest_planning",
            "1": "s01_fetch",
            "1.5": "s01_5_uicrit_join",
            "1.6": "s01_6_preference_pairs",
            "2": "s02_dedup",
            "3": "s03_quality",
            "3.5": "s03_5_pii_scrub",
            "4": "s04_safety",
            "4.5": "s04_5_escalation",
            "5": "s05_recaption",
            "5-ocr": "s05_ocr_enrichment",
            "5_ocr": "s05_ocr_enrichment",
            "5.1": "s05_ocr_enrichment",
            "ocr": "s05_ocr_enrichment",
            "5.5": "s05_5_pii_text_redact",
            "6": "s06_structure",
            "7": "s07_routing",
            "8": "s08_encoding",
            "8.5": "s08_5_dpo_encoding",
            "9": "s09_heldout",
            "10": "s10_audit",
            "11": "s11_registry_watcher",
            "12": "s12_model_data_export",
        }
        stages_filter = []
        for s in stages.split(","):
            s = s.strip()
            if not s:
                continue
            if s in stage_map:
                stages_filter.append(stage_map[s])
            elif s in STAGE_MODULES:
                stages_filter.append(s)
            elif f"s{s}" in STAGE_MODULES:
                stages_filter.append(f"s{s}")
            else:
                console.print(f"[red]Unknown stage: {s}[/]")
                console.print(f"Valid options include numbers ({', '.join(sorted(stage_map.keys()))}) or canonical names ({', '.join(STAGE_MODULES)})")
                sys.exit(1)

    console.print(f"\n[bold cyan]Data-Forge Pipeline v{config.version}[/]")
    console.print(f"  DATA_ROOT: {config.data_root}")
    console.print(f"  Chunk size: {config.chunk_size}")
    console.print(f"  Dry run: {dry_run}")
    console.print(f"  Resume: {resume}")
    if limit is not None:
        console.print(f"  Record limit: {limit}")
    if stages_filter:
        console.print(f"  Stages: {', '.join(stages_filter)}")
    console.print()

    # Import stages to trigger registration
    _register_all_stages()

    from data_forge.manifest import Manifest
    from data_forge.orchestrator import Orchestrator

    manifest_path = config.resolved_paths["manifests"] / "manifest.db"
    manifest = Manifest(manifest_path)

    try:
        orch = Orchestrator(config, manifest)
        asyncio.run(
            orch.execute_pipeline(
                stages_filter=stages_filter,
                dry_run=dry_run,
                resume=resume,
                limit=limit,
            )
        )

        # Print summary
        stats = manifest.stats()
        console.print("\n[bold green]Pipeline Complete[/]")
        _print_stats(stats)
    finally:
        manifest.close()


@main.command()
@click.option("--dataset", "dataset_key", required=True, help="Registry key to preprocess")
@click.option("--source-path", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True,
              help="Local directory containing the downloaded source artifacts")
@click.option("--limit", type=int, default=None, help="Optional deterministic record cap")
@click.option("--adapter", type=click.Choice(["figma2code", "mobileviews"]), default=None,
              help="Override the registry adapter mode")
def preprocess(dataset_key: str, source_path: Path, limit: int | None, adapter: str | None) -> None:
    """Normalize local UI artifacts without models, network, or synthetic labels.

    This command intentionally stops at manifest status ``fetched``. It does
    not claim license approval, safety, PII clearance, structure-model
    completion, or training eligibility. Run the normal pipeline stages after
    reviewing the generated provenance and source-structure receipt.
    """
    import json

    from data_forge.data.fetcher import DatasetFetcher
    from data_forge.manifest import Manifest

    config = _load_pipeline_config()
    spec = config.datasets.get(dataset_key)
    if spec is None:
        raise click.ClickException(f"Unknown dataset registry key: {dataset_key}")
    if not source_path.is_dir():
        raise click.ClickException(f"Source path is not a directory: {source_path}")

    local_spec = copy.deepcopy(spec)
    local_spec.source_type = "local"
    local_spec.fetch_config = dict(local_spec.fetch_config)
    local_spec.fetch_config["download_mode"] = adapter or (
        "figma2code_local" if dataset_key == "figma2code" else "mobileviews_local"
    )
    local_spec.fetch_config["local_path"] = str(source_path.resolve())
    if limit is not None:
        if limit < 0:
            raise click.ClickException("--limit must be non-negative")
        local_spec.fetch_config["sample_size"] = limit

    fetcher = DatasetFetcher(config)
    records = asyncio.run(fetcher.fetch_dataset(dataset_key, local_spec))
    manifest = Manifest(config.resolved_paths["manifests"] / "manifest.db")
    try:
        inserted = manifest.bulk_create_records(records, source_dataset=dataset_key, license_verified=None)
    finally:
        manifest.close()

    receipt = {
        "command": "data-forge preprocess",
        "dataset": dataset_key,
        "adapter": local_spec.fetch_config["download_mode"],
        "source_path": str(source_path.resolve()),
        "records_seen": len(records),
        "records_inserted": inserted,
        "status_after_preprocess": "fetched",
        "models_loaded": False,
        "network_used": False,
        "synthetic_labels_created": False,
        "note": "Run the governed pipeline stages separately after reviewing this receipt.",
    }
    receipt_path = config.resolved_paths["audit_reports"] / f"preprocess_{dataset_key}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    console.print(json.dumps({**receipt, "receipt_path": str(receipt_path)}, indent=2))


@main.command("preprocess-vlm")
@click.option("--profile", type=click.Choice(["local-vlm"]), default="local-vlm", show_default=True)
@click.option("--dataset", "dataset_key", default=None, help="Optional source dataset filter")
@click.option("--status", "record_status", default="fetched", show_default=True,
              help="Manifest status to enrich with the strict VLM bundle")
@click.option("--limit", type=int, default=None, help="Optional deterministic record cap")
@click.option("--concurrency", type=int, default=None,
              help="Override local VLM concurrency; default comes from pipeline.local_vlm")
@click.option("--receipt", type=click.Path(path_type=Path), default=None,
              help="Optional output path for the atomic VLM preprocessing receipt")
def preprocess_vlm(
    profile: str,
    dataset_key: str | None,
    record_status: str,
    limit: int | None,
    concurrency: int | None,
    receipt: Path | None,
) -> None:
    """Run the mandatory local VLM bundle over manifest records.

    This command is intentionally separate from model-free normalization. It
    starts the configured local Qwen VLM, requires CUDA/vLLM, validates every
    caption/structure/OCR/quality/safety response, and excludes failures. It
    never substitutes source metadata, a text-only model, CPU inference, or
    empty/freeform output.
    """
    import json

    from data_forge.inference.preprocessor import preprocess_fetched_records
    from data_forge.manifest import Manifest

    config = _load_pipeline_config(profile)
    config.strict_vlm_no_fallback = True
    config.validate_vlm_policy()
    _require_local_vlm_hardware(config)
    if limit is not None and limit < 0:
        raise click.ClickException("--limit must be non-negative")
    if concurrency is not None and concurrency < 1:
        raise click.ClickException("--concurrency must be positive")

    manifest = Manifest(config.resolved_paths["manifests"] / "manifest.db")
    try:
        records = manifest.query_by_status(record_status)
        if dataset_key is not None:
            records = [record for record in records if record.source_dataset == dataset_key]
        if limit is not None:
            records = records[:limit]
        result = asyncio.run(
            preprocess_fetched_records(
                config,
                manifest,
                records,
                concurrency=concurrency or config.local_vlm.concurrency,
                receipt_path=receipt,
            )
        )
    finally:
        manifest.close()

    console.print(json.dumps({
        "command": "data-forge preprocess-vlm",
        "profile": profile,
        "model": config.models[config.local_vlm.model_key].model_id,
        "records_considered": result.processed + result.excluded,
        "processed": result.processed,
        "excluded": result.excluded,
        "receipt_path": str(result.receipt_path),
        "fallbacks_allowed": False,
    }, indent=2))


@main.command()
@click.option("--github-token", default=None, help="GitHub token (avoids API rate limits)")
def doctor(github_token: str | None) -> None:
    """Pre-flight checks that are cheap to run but easy to forget.

    Currently checks whether the Unsloth toolchain has published support
    for the pinned Tier-1/Tier-2 model architectures — previously a real,
    working check (check_unsloth_support) that existed in the codebase but
    was never called from anywhere.
    """
    from data_forge.agents.toolchain_checker import (
        ToolchainCoverageError,
        check_unsloth_support,
    )

    config = _load_pipeline_config()
    github_token = github_token or os.environ.get("GITHUB_TOKEN")

    console.print("\n[bold cyan]data-forge doctor[/]\n")

    any_failed = False

    console.print("Checking stage ordering consistency (declared requires vs. actual execution order)...")
    _register_all_stages()
    from data_forge.orchestrator import validate_stage_ordering
    ordering_violations = validate_stage_ordering()
    if ordering_violations:
        any_failed = True
        for v in ordering_violations:
            console.print(f"  [bold red]✗[/] {v}")
    else:
        console.print("  [green]✓[/] Every stage's declared requires is consistent with actual execution order")

    for key in ("tier1", "tier2"):
        model_spec = config.models.get(key)
        if not model_spec:
            continue
        console.print(f"Checking Unsloth support for [bold]{key}[/] ({model_spec.model_id})...")
        try:
            check_result = asyncio.run(
                check_unsloth_support(
                    model_architecture=model_spec.model_id.split("/")[-1],
                    model_id=model_spec.model_id,
                    github_token=github_token,
                )
            )
            console.print(f"  [green]✓[/] {check_result['details']}")
        except ToolchainCoverageError as e:
            console.print(f"  [bold red]✗[/] {e}")
            any_failed = True
        except Exception as e:
            console.print(f"  [yellow]?[/] Could not determine support: {e}")

    if any_failed:
        console.print("\n[bold red]doctor found issues.[/] Review before committing a training run.")
        sys.exit(1)
    console.print("\n[green]All checks passed.[/]")


@main.group()
def registry() -> None:
    """Model & source registry management."""
    pass


@registry.command("check")
@click.option("--log-level", default="INFO")
def registry_check(log_level: str) -> None:
    """Run the registry watcher (designed for external cron trigger)."""
    from data_forge.logging_setup import setup_logging

    config = _load_pipeline_config()
    setup_logging(config.resolved_paths["logs"], log_level)
    _validate_environment()

    from data_forge.registry.watcher import RegistryWatcher

    watcher = RegistryWatcher(config)
    report = asyncio.run(watcher.check_all())

    output_path = config.resolved_paths["registry_reports"] / "latest.json"
    import json

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    console.print(f"\n[green]✓[/] Report written to {output_path}")


@main.group()
def manifest() -> None:
    """Manifest inspection and querying."""
    pass


@manifest.command("stats")
def manifest_stats() -> None:
    """Show manifest statistics."""
    config = _load_pipeline_config()
    from data_forge.manifest import Manifest

    manifest_path = config.resolved_paths["manifests"] / "manifest.db"
    if not manifest_path.exists():
        console.print("[yellow]No manifest found. Run the pipeline first.[/]")
        return

    m = Manifest(manifest_path)
    try:
        stats = m.stats()
        _print_stats(stats)
    finally:
        m.close()


@manifest.command("query")
@click.option("--status", required=True, help="Filter by record status")
@click.option("--limit", default=20, type=int, help="Max records to show")
def manifest_query(status: str, limit: int) -> None:
    """Query manifest records by status."""
    config = _load_pipeline_config()
    from data_forge.manifest import Manifest

    manifest_path = config.resolved_paths["manifests"] / "manifest.db"
    if not manifest_path.exists():
        console.print("[yellow]No manifest found.[/]")
        return

    m = Manifest(manifest_path)
    try:
        records = m.query_by_status(status, limit=limit)
        if not records:
            console.print(f"[yellow]No records with status '{status}'[/]")
            return

        table = Table(title=f"Records: {status} ({len(records)} shown)")
        table.add_column("ID", style="dim", max_width=12)
        table.add_column("Source")
        table.add_column("File")
        table.add_column("Reason")
        table.add_column("Updated")

        for r in records:
            table.add_row(
                r.id[:12],
                r.source_dataset,
                r.source_file or "—",
                r.exclusion_reason or "—",
                r.updated_at[:19] if r.updated_at else "—",
            )
        console.print(table)
    finally:
        m.close()


def _print_stats(stats: dict) -> None:  # type: ignore[type-arg]
    table = Table(title="Manifest Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Records", str(stats["total_records"]))
    table.add_row("Training Pool", str(stats["training_pool_count"]))
    table.add_row("Held-out Eval", str(stats["heldout_count"]))
    table.add_row("Excluded", str(stats["excluded_count"]))
    table.add_row("Raw Data (GB)", str(stats["total_raw_gb"]))
    table.add_row("Latest Version", stats["latest_version"] or "—")
    console.print(table)

    if stats["status_counts"]:
        detail = Table(title="Status Breakdown")
        detail.add_column("Status")
        detail.add_column("Count", justify="right")
        for status, count in sorted(stats["status_counts"].items()):
            detail.add_row(status, str(count))
        console.print(detail)


def _register_all_stages() -> None:
    """Import all stage modules to trigger @register_stage decorators."""
    from data_forge.stages import register_all_stages

    register_all_stages()


if __name__ == "__main__":
    main()
