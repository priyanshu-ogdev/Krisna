"""Toolchain coverage checker — validates Unsloth/toolchain support for pinned models (v13 §4.4).

CI-style check that queries the toolchain's repo for named-model support
before each training run.
"""

from __future__ import annotations

from typing import Any

import httpx

from data_forge.logging_setup import get_logger

log = get_logger("agents.toolchain_checker")

_UNSLOTH_REPO = "unslothai/unsloth"
_GITHUB_API = "https://api.github.com"


class ToolchainCoverageError(Exception):
    """Raised when the toolchain doesn't support the required model."""


async def check_unsloth_support(
    model_architecture: str,
    model_id: str,
    github_token: str | None = None,
) -> dict[str, Any]:
    """Check if Unsloth supports the given model architecture.

    Queries:
    1. Unsloth's GitHub releases for supported model mentions
    2. The README/docs for architecture compatibility

    Returns:
        {
            "supported": bool,
            "model_id": str,
            "latest_release": str,
            "check_method": str,
            "details": str,
        }

    Raises:
        ToolchainCoverageError if definitively not supported.
    """
    headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        # Check releases
        releases_url = f"{_GITHUB_API}/repos/{_UNSLOTH_REPO}/releases"
        try:
            resp = await client.get(releases_url, params={"per_page": 5})
            resp.raise_for_status()
            releases = resp.json()
        except Exception as e:
            log.warning("github_api_error", error=str(e))
            return {
                "supported": None,  # Unknown
                "model_id": model_id,
                "latest_release": "unknown",
                "check_method": "github_api_failed",
                "details": str(e),
            }

        latest_release = releases[0]["tag_name"] if releases else "unknown"

        # Search release notes and README for model mentions
        model_short = model_architecture.lower()
        found_in_releases = False

        for release in releases:
            body = (release.get("body") or "").lower()
            name = (release.get("name") or "").lower()
            if model_short in body or model_short in name:
                found_in_releases = True
                break

        # Check README
        readme_url = f"{_GITHUB_API}/repos/{_UNSLOTH_REPO}/readme"
        try:
            resp = await client.get(readme_url)
            resp.raise_for_status()
            readme_data = resp.json()
            import base64

            readme_content = base64.b64decode(
                readme_data.get("content", "")
            ).decode("utf-8", errors="replace").lower()
            found_in_readme = model_short in readme_content
        except Exception:
            found_in_readme = False

        supported = found_in_releases or found_in_readme

        result = {
            "supported": supported,
            "model_id": model_id,
            "architecture": model_architecture,
            "latest_release": latest_release,
            "check_method": "github_releases_and_readme",
            "found_in_releases": found_in_releases,
            "found_in_readme": found_in_readme,
            "details": (
                f"Model architecture '{model_architecture}' "
                f"{'found' if supported else 'NOT found'} in Unsloth "
                f"releases/docs (latest: {latest_release})"
            ),
        }

        log.info("toolchain_check_completed", **result)

        if not supported:
            raise ToolchainCoverageError(
                f"Unsloth does not appear to support '{model_architecture}' "
                f"(model: {model_id}). Latest release: {latest_release}. "
                f"Check https://github.com/{_UNSLOTH_REPO} for updates."
            )

        return result
