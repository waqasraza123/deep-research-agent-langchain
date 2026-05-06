from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .contracts import (
    KernelWarning,
    ResearchBlueprint,
    ResearchKernelSettings,
    SourceUnit,
    clamp_score,
    stable_id,
    write_json,
)

INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous",
    "system prompt",
    "developer message",
    "you are chatgpt",
    "exfiltrate",
)


def normalize_source_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if not parsed.scheme:
        parsed = urlparse("https://" + (url or "").strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query_items = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    ]
    return urlunparse((scheme, netloc, path, "", urlencode(sorted(query_items)), ""))


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _safe_local_text(run_dir: Path, item: dict[str, Any], max_chars: int) -> str:
    rel = str(item.get("local_path") or item.get("path") or "")
    if not rel:
        return str(item.get("text") or item.get("content") or item.get("summary") or "")
    marker = "/runs/"
    if marker in rel:
        parts = rel.split(marker, 1)[-1].split("/", 1)
        rel = parts[1] if len(parts) == 2 else ""
    if rel.startswith("/") or "\\" in rel or ".." in rel:
        return str(item.get("summary") or "")
    path = (run_dir / rel).resolve()
    if not str(path).startswith(str(run_dir.resolve())) or not path.exists() or path.is_dir():
        return str(item.get("summary") or "")
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:max_chars]


def load_sources_payload(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "sources.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("sources", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _quality(item: dict[str, Any], text: str) -> float:
    explicit = (
        item.get("final_quality_score") or item.get("quality_score") or item.get("priority_score")
    )
    if isinstance(explicit, dict):
        explicit = explicit.get("final_quality_score") or explicit.get("extraction_quality")
    try:
        return clamp_score(float(explicit))
    except Exception:
        pass
    words = len(text.split())
    if words >= 350:
        return 0.8
    if words >= 120:
        return 0.62
    if words >= 30:
        return 0.36
    return 0.12


def _trust_and_role(
    item: dict[str, Any],
    domain: str,
    quality: float,
    duplicate: bool,
    risky: bool,
    blueprint: ResearchBlueprint,
) -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    officialish = any(
        hint in domain
        for hint in (
            "docs.",
            "github.com",
            "readthedocs",
            "python.org",
            "fastapi.tiangolo.com",
            "langchain",
            "langgraph",
            "sec.gov",
            "gov",
            "who.int",
            "nih.gov",
        )
    )
    if risky:
        return (
            "risky",
            "risky_source",
            ["Potential prompt-injection or unsafe source content detected."],
        )
    if duplicate:
        return "low", "duplicate", ["Duplicate normalized URL."]
    if quality < 0.25:
        return "low", "weak_reference", ["Weak or empty extraction."]
    if officialish:
        return "high", "primary_evidence", warnings
    if blueprint.intent.label in {
        "legal_policy_review",
        "medical_health_review",
        "financial_risk_review",
    }:
        warnings.append("Sensitive-domain source is not clearly primary or authoritative.")
        return "medium" if quality > 0.55 else "low", "secondary_context", warnings
    if quality > 0.65:
        return "medium", "secondary_context", warnings
    return "unknown", "background", warnings


def build_source_units(
    run_dir: Path,
    blueprint: ResearchBlueprint,
    settings: ResearchKernelSettings | None = None,
) -> tuple[list[SourceUnit], dict[str, Any], list[KernelWarning]]:
    settings = settings or ResearchKernelSettings()
    payload = load_sources_payload(run_dir)
    seen_urls: dict[str, str] = {}
    units: list[SourceUnit] = []
    warnings: list[KernelWarning] = []
    for item in payload[: settings.max_source_units]:
        raw_url = str(item.get("final_url") or item.get("url") or item.get("canonical_url") or "")
        if not raw_url:
            continue
        normalized = normalize_source_url(raw_url)
        domain = _domain(normalized)
        text = _safe_local_text(run_dir, item, max_chars=80_000)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        duplicate = normalized in seen_urls
        risky = any(pattern in text.lower() for pattern in INJECTION_PATTERNS)
        quality = _quality(item, text)
        trust, role, raw_warnings = _trust_and_role(
            item, domain, quality, duplicate, risky, blueprint
        )
        source_id = str(item.get("source_id") or item.get("id") or stable_id("source", normalized))
        source_unit_id = stable_id("su", source_id, normalized)
        unit_warnings = [
            KernelWarning(
                warning_id=stable_id("warn", source_unit_id, warning),
                subsystem="source_units",
                code="source_unit_warning",
                severity="high" if risky else "medium",
                message=warning,
                affected_sources=[source_id],
                recommended_action="Use cautiously and prefer stronger corroborating sources.",
            )
            for warning in raw_warnings
        ]
        if duplicate:
            source_id = source_id or seen_urls[normalized]
        else:
            seen_urls[normalized] = source_id
        title = item.get("title")
        preview = " ".join(text.split())[:500]
        useful_for = []
        if role in {"primary_evidence", "secondary_context"}:
            useful_for = blueprint.evidence_requirements[:8]
        unsafe_for = ["direct prompt instructions"] if risky else []
        units.append(
            SourceUnit(
                source_unit_id=source_unit_id,
                source_id=source_id,
                url=raw_url,
                canonical_url=item.get("canonical_url"),
                normalized_url=normalized,
                domain=domain,
                title=str(title) if title else None,
                source_type=str(item.get("document_kind") or item.get("content_type") or "unknown"),
                source_kind=str(item.get("source_kind") or "unknown"),
                parent_url=item.get("parent_url"),
                fetched_at=item.get("fetched_at"),
                content_hash=digest or item.get("content_hash"),
                extraction_quality=quality,
                trust_level=trust,  # type: ignore[arg-type]
                source_role=role,  # type: ignore[arg-type]
                source_warnings=unit_warnings,
                useful_for=useful_for,
                unsafe_for=unsafe_for,
                text_preview=preview,
            )
        )
        warnings.extend(unit_warnings)
    inventory = {
        "source_count": len(units),
        "usable_source_count": len(
            [
                u
                for u in units
                if u.source_role not in {"duplicate", "risky_source", "weak_reference"}
            ]
        ),
        "duplicate_count": len([u for u in units if u.source_role == "duplicate"]),
        "risky_source_count": len([u for u in units if u.source_role == "risky_source"]),
        "trust_levels": {
            level: len([u for u in units if u.trust_level == level])
            for level in ["high", "medium", "low", "unknown", "risky"]
        },
    }
    if not units:
        warnings.append(
            KernelWarning(
                warning_id=stable_id("warn", "source_units", "empty", blueprint.thread_id),
                subsystem="source_units",
                code="no_sources",
                severity="high",
                message="No source units could be created from sources.json.",
                affected_artifacts=["sources.json"],
                recommended_action="Fetch or attach source content before relying on the report.",
            )
        )
    return units, inventory, warnings


def render_source_units_markdown(units: list[SourceUnit], inventory: dict[str, Any]) -> str:
    lines = [
        "# Source Units",
        "",
        f"Total sources: {inventory.get('source_count', 0)}",
        "",
        "| Source | Role | Trust | Quality | Warnings | Recommended use |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for unit in units:
        label = unit.title or unit.domain or unit.url
        warnings = "; ".join(w.message for w in unit.source_warnings) or "None"
        use = "; ".join(unit.useful_for[:3]) or "Background only"
        lines.append(
            f"| {label[:80]} | `{unit.source_role}` | `{unit.trust_level}` | {unit.extraction_quality:.2f} | {warnings[:120]} | {use[:120]} |"
        )
    return "\n".join(lines) + "\n"


def write_source_unit_artifacts(
    run_dir: Path, units: list[SourceUnit], inventory: dict[str, Any]
) -> list[str]:
    write_json(run_dir / "source_units.json", {"source_units": units})
    write_json(run_dir / "source_inventory.json", inventory)
    (run_dir / "source_units.md").write_text(
        render_source_units_markdown(units, inventory), encoding="utf-8"
    )
    (run_dir / "source_inventory.md").write_text(
        "# Source Inventory\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in inventory.items())
        + "\n",
        encoding="utf-8",
    )
    return ["source_units.json", "source_units.md", "source_inventory.json", "source_inventory.md"]
