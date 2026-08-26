"""Structured JSON-lines logging and audit trail for data-forge."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(log_dir: Path, log_level: str = "INFO") -> None:
    """Configure structlog for JSON-lines output to both file and stderr.

    Every log event includes:
      - timestamp (ISO 8601)
      - level
      - event (message)
      - logger (module name)
      - Any bound key-value context (stage, record_id, model, etc.)

    Two outputs:
      1. {log_dir}/pipeline.jsonl — append-only structured log
      2. stderr — human-readable for interactive use
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.jsonl"

    # --- stdlib root logger → file handler (JSON) ---
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stderr_handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # --- structlog configuration ---
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            # For file: JSON serialization
            # For console: human-readable
            structlog.processors.JSONRenderer() if not sys.stderr.isatty()
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **initial_context: object) -> structlog.stdlib.BoundLogger:
    """Get a named logger with optional initial bound context.

    Usage:
        log = get_logger("stages.s01_fetch", stage="fetch", chunk_id="c001")
        log.info("downloading dataset", dataset="rico_core", records=66000)
    """
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger  # type: ignore[return-value]
