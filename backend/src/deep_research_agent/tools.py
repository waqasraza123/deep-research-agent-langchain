from __future__ import annotations

import httpx


def fetch_url(url: str, *, timeout_s: float = 25.0, max_chars: int = 250_000) -> str:
    """
    Free web fetch tool (no paid search API).
    Safety:
      - only http(s)
      - follows redirects
      - truncates large responses
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Blocked: only http(s) URLs are allowed."

    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "deep-research-agent/0.1"})
            r.raise_for_status()
            text = r.text or ""

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[TRUNCATED]\n"
        return text
    except Exception as e:
        return f"Fetch failed: {type(e).__name__}: {e}"