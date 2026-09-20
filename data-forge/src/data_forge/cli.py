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
import os
import sys
from pathlib import Path

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

import click
from rich.console import Console
from rich.table import Table

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
            "[bold red]ERROR:[/] HF_TOKEN environment variable not set.\n"
            "Set it in .env or via: export HF_TOKEN=hf_...",
            highlight=False,
        )
        sys.exit(1)

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


def _load_pipeline_config() -> "PipelineConfig":
    from data_forge.config import PipelineConfig, load_config

    configs_dir = _find_configs_dir()
    config = load_config(
        pipeline_yaml=configs_dir / "pipeline.yaml",
        models_yaml=configs_dir / "models.yaml",
        datasets_yaml=configs_dir / "datasets.yaml",
    )
    return config


@click.group()
@click.version_option(version="0.14.0", prog_name="data-forge")
def main() -> None:
    """Data-Forge: Zero-touch data pipeline for the Krisna project (v14)."""
    import multiprocessing
    import sys
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn', force=True)


@main.command()
@click.option("--dry-run", is_flag=True, help="Validate config and walk stages without inference")
@click.option("--resume/--no-resume", default=True, help="Resume from checkpoints")
@click.option("--stages", default=None, help="Comma-separated stage numbers to run (e.g., '0,1,2')")
@click.option("--chunk-size", default=None, type=int, help="Override chunk size from config")
@click.option("--limit", default=None, type=int, help="Cap total records processed this run (smoke tests)")
@click.option("--log-level", default="INFO", help="Logging level")
def run(
    dry_run: bool,
    resume: bool,
    stages: str | None,
    chunk_size: int | None,
    limit: int | None,
    log_level: str,
) -> None:
    """Execute the data pipeline."""
    from data_forge.logging_setup import setup_logging

    config = _load_pipeline_config()
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
