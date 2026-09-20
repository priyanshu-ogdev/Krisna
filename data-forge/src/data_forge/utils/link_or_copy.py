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
from pathlib import Path

from data_forge.logging_setup import get_logger

log = get_logger("utils.link_or_copy")


def link_or_copy(source: Path, dest: Path) -> str:
    """Place `source`'s content at `dest` as cheaply as the platform allows.

    Returns which strategy was used: "symlink", "hardlink", "copy", or
    "exists" (dest already there from a prior run — treated as success,
    not re-done, so repeated export runs are cheap).
    """
    if dest.exists() or dest.is_symlink():
        return "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        dest.symlink_to(source.resolve())
        return "symlink"
    except (OSError, NotImplementedError):
        pass

    try:
        os.link(source, dest)
        return "hardlink"
    except OSError:
        pass

    shutil.copy2(source, dest)
    return "copy"
