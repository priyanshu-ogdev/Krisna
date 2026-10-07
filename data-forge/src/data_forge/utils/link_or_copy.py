"""Portable "link without duplicating storage" helper.

The model-data export stage needs to place the same underlying files
(images, latents, VQ tokens) under multiple per-model directory trees
without doubling disk usage across a corpus that's already sized in the
hundreds of GB. Symlinks are the ideal mechanism but are unreliable on
Windows without Developer Mode or admin rights (a real constraint this
project's own setup docs already account for elsewhere). This tries, in
order: symlink -> hardlink (same-volume only, near-zero overhead, no
special privileges needed on either platform) -> copy (last resort, only
if both fail) — and reports which strategy actually got used so a run log
doesn't quietly claim "linked" when it silently copied gigabytes instead.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from data_forge.logging_setup import get_logger

log = get_logger("utils.link_or_copy")


def link_or_copy(source: Path, dest: Path) -> str:
    """Place `source`'s content at `dest` as cheaply as the platform allows.

    Returns which strategy was used: "symlink", "hardlink", "copy", or
    "exists" (dest already there from a prior run — treated as success,
    not re-done, so repeated export runs are cheap).
    """
    if not source.is_file():
        raise FileNotFoundError(f"Export source does not exist or is not a file: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    had_destination = dest.exists() or dest.is_symlink()
    if had_destination:
        try:
            if os.path.samefile(source, dest):
                return "exists"
        except OSError:
            try:
                source_stat = source.stat()
                dest_stat = dest.stat()
                if source_stat.st_size == dest_stat.st_size and source_stat.st_mtime_ns == dest_stat.st_mtime_ns:
                    return "exists"
            except OSError:
                pass

    temp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")

    try:
        try:
            temp.symlink_to(source.resolve())
            strategy = "symlink"
        except (OSError, NotImplementedError):
            try:
                os.link(source, temp)
                strategy = "hardlink"
            except OSError:
                shutil.copy2(source, temp)
                strategy = "copy"

        os.replace(temp, dest)
        return f"replaced_{strategy}" if had_destination else strategy
    finally:
        if temp.exists() or temp.is_symlink():
            temp.unlink()
