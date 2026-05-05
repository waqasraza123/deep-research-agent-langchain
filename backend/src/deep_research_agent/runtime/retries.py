from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from .contracts import RetryPolicy

T = TypeVar("T")


NONRETRYABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    PermissionError,
    FileNotFoundError,
)

RETRYABLE_NAMES = (
    "Timeout",
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectionError",
    "RateLimitError",
    "APIConnectionError",
    "ServiceUnavailable",
    "InternalServerError",
)


def classify_error(exc: BaseException) -> str:
    if isinstance(exc, NONRETRYABLE_EXCEPTIONS):
        return "nonretryable"
    name = type(exc).__name__
    if any(token in name for token in RETRYABLE_NAMES):
        return "retryable"
    msg = str(exc).lower()
    if any(token in msg for token in ("timeout", "temporarily", "rate limit", "connection")):
        return "retryable"
    return "unknown"


def is_retryable(exc: BaseException) -> bool:
    return classify_error(exc) == "retryable"


def retry_sync(fn: Callable[[], T], policy: RetryPolicy) -> tuple[T, dict]:
    attempts = 0
    failures: list[dict[str, str | int]] = []
    while True:
        attempts += 1
        try:
            result = fn()
            return result, {"attempts": attempts, "failures": failures}
        except Exception as exc:
            classification = classify_error(exc)
            failures.append(
                {
                    "attempt": attempts,
                    "error_type": type(exc).__name__,
                    "classification": classification,
                    "message": str(exc),
                }
            )
            if classification == "nonretryable" or attempts >= policy.max_attempts:
                raise
            delay = min(
                policy.max_backoff_s,
                policy.initial_backoff_s * (policy.backoff_multiplier ** (attempts - 1)),
            )
            if policy.jitter_s:
                delay += random.uniform(0, policy.jitter_s)
            if delay > 0:
                time.sleep(delay)
