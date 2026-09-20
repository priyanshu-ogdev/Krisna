"""HTTP client for license page fetching and web content retrieval."""

from __future__ import annotations

import re

import httpx

from data_forge.logging_setup import get_logger

log = get_logger("utils.web_fetch")

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_MAX_CONTENT_LENGTH = 500_000  # 500KB max for license pages


async def fetch_page_text(url: str) -> str | None:
    """Fetch a web page and extract its text content.

    Returns the text content stripped of HTML tags, or None on failure.
    """
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=_TIMEOUT
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content = response.text
            if len(content) > _MAX_CONTENT_LENGTH:
                content = content[:_MAX_CONTENT_LENGTH]

            # Strip HTML tags
            text = _strip_html(content)
            log.info("page_fetched", url=url, text_length=len(text))
            return text

    except httpx.HTTPStatusError as e:
        log.warning("page_fetch_http_error", url=url, status=e.response.status_code)
        return None
    except Exception as e:
        log.warning("page_fetch_error", url=url, error=str(e))
        return None


def _strip_html(html: str) -> str:
    """Basic HTML tag removal — good enough for license pages."""
    # Remove script and style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    html = html.replace("&amp;", "&")
    html = html.replace("&lt;", "<")
    html = html.replace("&gt;", ">")
    html = html.replace("&quot;", '"')
    html = html.replace("&#39;", "'")
    html = html.replace("&nbsp;", " ")
    # Collapse whitespace
    html = re.sub(r"\s+", " ", html).strip()
    return html
