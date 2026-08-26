"""HuggingFace model release checker."""

from __future__ import annotations

import os
from typing import Any

import httpx

from data_forge.logging_setup import get_logger

log = get_logger("registry.hf_checker")


async def check_model_updates(
    model_id: str,
    current_revision: str = "main",
) -> dict[str, Any]:
    """Check HuggingFace for new model versions.

    Returns:
        {
            "has_update": bool,
            "latest_revision": str,
            "latest_tag": str | None,
            "reason": str,
        }
    """
    hf_token = os.environ.get("HF_TOKEN")
    headers: dict[str, str] = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    api_url = f"https://huggingface.co/api/models/{model_id}"

    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            resp = await client.get(api_url)
            resp.raise_for_status()
            data = resp.json()

        # Check last modified
        last_modified = data.get("lastModified", "")

        # Check tags/siblings for version info
        tags = data.get("tags", [])
        sha = data.get("sha", "")

        # Check if there are newer refs
        refs_url = f"https://huggingface.co/api/models/{model_id}/refs"
        refs_resp = await httpx.AsyncClient(timeout=15, headers=headers).get(refs_url)

        latest_tag = None
        if refs_resp.status_code == 200:
            refs_data = refs_resp.json()
            branches = refs_data.get("branches", [])
            tags_list = refs_data.get("tags", [])

            if tags_list:
                latest_tag = tags_list[-1].get("name")

            # Check if main branch has moved
            for branch in branches:
                if branch.get("name") == "main":
                    if branch.get("ref") != current_revision and current_revision != "main":
                        return {
                            "has_update": True,
                            "latest_revision": branch.get("ref", sha),
                            "latest_tag": latest_tag,
                            "reason": f"Main branch updated (was {current_revision})",
                        }

        # If we're tracking main and there's a new tag, suggest investigation
        if latest_tag and current_revision == "main":
            return {
                "has_update": False,
                "latest_revision": sha,
                "latest_tag": latest_tag,
                "reason": "On main, consider pinning to latest tag",
            }

        return {
            "has_update": False,
            "latest_revision": sha,
            "latest_tag": latest_tag,
            "reason": "Up to date",
        }

    except Exception as e:
        log.warning("hf_check_failed", model_id=model_id, error=str(e))
        return {
            "has_update": False,
            "latest_revision": current_revision,
            "latest_tag": None,
            "reason": f"Check failed: {e}",
        }
