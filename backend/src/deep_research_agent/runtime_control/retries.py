from __future__ import annotations

from datetime import timedelta

from .contracts import RetryPolicy, utc_now


def classify_error(exc: BaseException, policy: RetryPolicy | None = None) -> str:
    policy = policy or RetryPolicy()
    name = type(exc).__name__
    message = str(exc).lower()
    if name in policy.non_retryable_error_types:
        return "nonretryable"
    if any(marker in message for marker in ("path traversal", "ssrf", "validation", "api key")):
        return "nonretryable"
    if name in policy.retryable_error_types:
        return "retryable"
    if any(marker in message for marker in ("timeout", "temporar", "rate limit", "429", "503")):
        return "retryable"
    return "nonretryable"


def retry_delay_seconds(attempts: int, policy: RetryPolicy) -> float:
    exponent = max(0, attempts - 1)
    return min(
        policy.backoff_max_seconds,
        policy.backoff_initial_seconds * (policy.backoff_multiplier**exponent),
    )


def retry_after(attempts: int, policy: RetryPolicy):
    return utc_now() + timedelta(seconds=retry_delay_seconds(attempts, policy))

