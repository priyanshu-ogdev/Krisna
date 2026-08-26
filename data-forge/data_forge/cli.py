"""CLI entry point for data-forge.

Commands:
  data-forge run [--dry-run] [--resume] [--stages 0,1,2]
  data-forge registry check
  data-forge manifest stats
  data-forge manifest query --status <status>
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _validate_environment() -> None:
    """Pre-flight environment validation."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        console.print(
            "[bold red]ERROR:[/] HF_TOKEN environment variable not set.\n"
            "Set it with: export HF_TOKEN=hf_...",
            highlight=False,
        )
        sys.exit(1)

    # Validate HF token has read access
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        api.whoami()
        console.print("[green]✓[/] HuggingFace token validated")
    except Exception as e:
        console.print(f"[bold yellow]WARNING:[/] Could not validate HF token: {e}")


def _load_pipeline_config() -> "PipelineConfig":
    from data_forge.config import PipelineConfig, load_config

    # Find configs relative to the package or CWD
    base = Path.cwd()
    config = load_config(
        pipeline_yaml=base / "configs" / "pipeline.yaml",
        models_yaml=base / "configs" / "models.yaml",
        datasets_yaml=base / "configs" / "datasets.yaml",
    )
    return config


@click.group()
@click.version_option(version="0.13.0", prog_name="data-forge")
def main() -> None:
    """Data-Forge: Zero-touch data pipeline for the Krisna project (v13)."""
    pass


@main.command()
@click.option("--dry-run", is_flag=True, help="Validate config and walk stages without inference")
@click.option("--resume/--no-resume", default=True, help="Resume from checkpoints")
@click.option("--stages", default=None, help="Comma-separated stage numbers to run (e.g., '0,1,2')")
@click.option("--chunk-size", default=None, type=int, help="Override chunk size from config")
@click.option("--log-level", default="INFO", help="Logging level")
def run(
    dry_run: bool,
    resume: bool,
    stages: str | None,
    chunk_size: int | None,
    log_level: str,
) -> None:
    """Execute the data pipeline."""
    from data_forge.logging_setup import setup_logging

    config = _load_pipeline_config()

    if chunk_size:
        config.chunk_size = chunk_size

    setup_logging(config.resolved_paths["logs"], log_level)

    if not dry_run:
        _validate_environment()

    # Parse stage filter
    stages_filter: list[str] | None = None
    if stages:
        stage_map = {
            "0": "s00_manifest_planning",
            "1": "s01_fetch",
            "2": "s02_dedup",
            "3": "s03_quality",
            "3.5": "s03_5_pii_scrub",
            "4": "s04_safety",
            "4.5": "s04_5_escalation",
            "5": "s05_recaption",
            "6": "s06_structure",
            "7": "s07_routing",
            "8": "s08_encoding",
            "9": "s09_heldout",
            "10": "s10_audit",
            "11": "s11_registry_watcher",
        }
        stages_filter = []
        for s in stages.split(","):
            s = s.strip()
            if s in stage_map:
                stages_filter.append(stage_map[s])
            else:
                console.print(f"[red]Unknown stage: {s}[/]")
                sys.exit(1)

    console.print(f"\n[bold cyan]Data-Forge Pipeline v{config.version}[/]")
    console.print(f"  DATA_ROOT: {config.data_root}")
    console.print(f"  Chunk size: {config.chunk_size}")
    console.print(f"  Dry run: {dry_run}")
    console.print(f"  Resume: {resume}")
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
            )
        )

        # Print summary
        stats = manifest.stats()
        console.print("\n[bold green]Pipeline Complete[/]")
        _print_stats(stats)
    finally:
        manifest.close()


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
    import importlib

    stage_modules = [
        "data_forge.stages.s00_manifest_planning",
        "data_forge.stages.s01_fetch",
        "data_forge.stages.s02_dedup",
        "data_forge.stages.s03_quality",
        "data_forge.stages.s03_5_pii_scrub",
        "data_forge.stages.s04_safety",
        "data_forge.stages.s04_5_escalation",
        "data_forge.stages.s05_recaption",
        "data_forge.stages.s06_structure",
        "data_forge.stages.s07_routing",
        "data_forge.stages.s08_encoding",
        "data_forge.stages.s09_heldout",
        "data_forge.stages.s10_audit",
        "data_forge.stages.s11_registry_watcher",
    ]
    for mod in stage_modules:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            console.print(f"[yellow]Warning: Could not load stage {mod}: {e}[/]")


if __name__ == "__main__":
    main()
