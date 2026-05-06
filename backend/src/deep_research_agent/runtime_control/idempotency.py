from __future__ import annotations

import hashlib
import json
from typing import Any

SECRET_MARKERS = ("key", "token", "secret", "password", "credential", "authorization")


def redact_settings_snapshot(settings: Any) -> dict[str, Any]:
    if settings is None:
        return {}
    if hasattr(settings, "__dataclass_fields__"):
        raw = {key: getattr(settings, key) for key in settings.__dataclass_fields__}
    elif isinstance(settings, dict):
        raw = dict(settings)
    else:
        raw = dict(getattr(settings, "__dict__", {}))
    return _redact(raw)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_s = str(key)
            if any(marker in key_s.lower() for marker in SECRET_MARKERS):
                out[key_s] = "[REDACTED]" if item else ""
            else:
                out[key_s] = _redact(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if hasattr(value, "as_posix"):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_question(question: str) -> str:
    return " ".join((question or "").strip().lower().split())


def normalize_urls(urls: list[str]) -> list[str]:
    normalized = []
    for url in urls:
        clean = (url or "").strip()
        if clean:
            normalized.append(clean.rstrip("/"))
    return sorted(set(normalized))


def compute_idempotency_key(
    *,
    question: str,
    urls: list[str],
    settings_snapshot: dict[str, Any] | None = None,
    explicit_key: str | None = None,
) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    relevant_settings = settings_snapshot or {}
    payload = {
        "question": normalize_question(question),
        "urls": normalize_urls(urls),
        "settings": relevant_settings,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

