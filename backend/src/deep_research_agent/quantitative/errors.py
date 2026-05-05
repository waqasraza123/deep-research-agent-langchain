from __future__ import annotations


class QuantitativeError(Exception):
    """Base error for deterministic quantitative processing."""


class QuantitativeParseError(QuantitativeError):
    """Raised when a structured quantitative artifact cannot be parsed."""
