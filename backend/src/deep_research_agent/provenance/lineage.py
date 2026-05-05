from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from deep_research_agent.artifacts import INTERNAL_FILES, safe_thread_id

from .contracts import (
    ArtifactDependency,
    ModelInvocationFingerprint,
    RunInputFingerprint,
    SourceFingerprint,
    SubsystemInvocation,
)
from .errors import ProvenancePathError

PROVENANCE_ARTIFACTS = {
    "artifact_manifest.json",
    "artifact_manifest.md",
    "artifact_dependency_dag.json",
    "artifact_dependency_dag.md",
    "reproducibility_report.json",
    "reproducibility_report.md",
    "replay_plan.json",
    "replay_plan.md",
}

SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
    "auth",
)

SUBSYSTEM_VERSION = "1.0"


def model_to_plain(model: Any) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    if isinstance(model, dict):
        return model
    return {}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_run_dir(runs_dir: Path, thread_id: str) -> Path:
    safe_thread_id(thread_id)
    root = runs_dir.resolve()
    td = (root / thread_id).resolve()
    if root != td and root not in td.parents:
        raise ProvenancePathError("Invalid thread_id")
    if not td.exists() or not td.is_dir():
        raise FileNotFoundError(thread_id)
    return td


def safe_relative_path(thread_dir: Path, path: Path) -> str:
    resolved_thread_dir = thread_dir.resolve()
    resolved_path = path.resolve()
    if resolved_thread_dir != resolved_path and resolved_thread_dir not in resolved_path.parents:
        raise ProvenancePathError("Artifact path escaped run directory")
    rel = str(resolved_path.relative_to(resolved_thread_dir)).replace(os.sep, "/")
    if rel.startswith("/") or "\\" in rel or ".." in rel.split("/"):
        raise ProvenancePathError("Invalid artifact path")
    return rel


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_s = str(key)
            if any(part in key_s.lower() for part in SECRET_KEY_PARTS):
                redacted[key_s] = "[REDACTED]"
            else:
                redacted[key_s] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and _looks_like_secret(value):
        return "[REDACTED]"
    return value


def _looks_like_secret(value: str) -> bool:
    stripped = value.strip()
    if stripped.startswith(("sk-", "Bearer ", "ghp_", "github_pat_")):
        return True
    return len(stripped) >= 32 and any(c.isdigit() for c in stripped) and any(
        c.isalpha() for c in stripped
    )


def settings_snapshot_from_run(run: Any | None) -> dict[str, Any]:
    if run is None:
        return {}
    snapshot = getattr(run, "input_snapshot", None)
    if snapshot is not None:
        settings = getattr(snapshot, "settings", None)
        if isinstance(settings, dict):
            return redact_secrets(settings)
    data = model_to_plain(run)
    return redact_secrets(data.get("input_snapshot", {}).get("settings", {}))


def build_run_input_fingerprint(run: Any | None) -> RunInputFingerprint | None:
    if run is None:
        return None
    question = getattr(run, "question", "") or ""
    urls = list(getattr(run, "urls", []) or [])
    if not question:
        data = model_to_plain(run)
        question = str(data.get("question") or "")
        urls = [str(u) for u in data.get("urls") or []]
    normalized_urls = sorted(_normalize_url(url) for url in urls)
    return RunInputFingerprint(
        question_hash=stable_hash(question.strip()),
        urls_hash=stable_hash(normalized_urls),
        combined_hash=stable_hash(
            {"question": question.strip(), "normalized_urls": normalized_urls}
        ),
        question_length=len(question.strip()),
        urls=urls,
        normalized_urls=normalized_urls,
    )


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    return url.rstrip("/")


def iter_artifact_files(thread_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for path in thread_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = safe_relative_path(thread_dir, path)
        if rel in INTERNAL_FILES or rel.endswith(".tmp"):
            continue
        out.append((rel, path))
    out.sort(key=lambda item: item[0])
    return out


def artifact_type_for_path(path: str) -> str:
    name = Path(path).name
    suffix = Path(path).suffix.lower().lstrip(".")
    explicit = {
        "plan.md": "plan",
        "notes.md": "notes",
        "sources.json": "source_manifest",
        "report.md": "report",
        "artifact_manifest.json": "provenance_manifest",
        "artifact_dependency_dag.json": "provenance_dag",
        "reproducibility_report.json": "reproducibility_report",
        "replay_plan.json": "replay_plan",
        "quality_score.json": "quality_score",
        "verification_report.json": "verification_report",
        "evidence_ledger.json": "evidence_ledger",
        "context_packs.json": "context_packs",
        "document_chunks.json": "document_chunks",
    }
    if name in explicit:
        return explicit[name]
    if suffix in {"json", "jsonl", "md", "txt", "html"}:
        return suffix
    return "artifact"


def producer_for_path(path: str) -> str:
    name = Path(path).name
    if path in PROVENANCE_ARTIFACTS:
        return "provenance"
    checks = (
        ("protocol", ("protocol",)),
        ("memory", ("memory_",)),
        (
            "source_discovery",
            (
                "source_acquisition",
                "source_candidates",
                "source_selection",
                "search_queries",
                "source_discovery",
            ),
        ),
        ("source_fetching", ("sources.json", "source_graph", "source_rankings", "source_warnings")),
        ("source_audit", ("source_audit", "citation_readiness")),
        ("document_intelligence", ("document_", "chunks", "tables")),
        ("retrieval", ("retrieval_", "context_pack", "context_packs")),
        ("orchestration", ("task_graph", "stage_outputs", "orchestration")),
        ("evidence", ("evidence_", "claims", "citation_map", "contradiction")),
        ("verification", ("verification", "confidence_calibration", "claim_rewrite")),
        (
            "synthesis",
            ("argument_map", "comparison_matrix", "decision_memo", "uncertainty", "synthesis"),
        ),
        (
            "evaluation",
            ("evaluation", "quality_score", "hallucination", "coverage", "freshness", "balance"),
        ),
        ("agent", ("plan.md", "notes.md", "report.md", "metadata.json")),
        ("runtime", ("events.", "budget.json", "run.json")),
    )
    for subsystem, markers in checks:
        if any(marker in name for marker in markers):
            return subsystem
    if path.startswith("sources/"):
        return "source_fetching"
    return "unknown"


def infer_artifact_dependencies(path: str) -> list[ArtifactDependency]:
    name = Path(path).name
    deps: list[ArtifactDependency] = []

    def add(identifier: str, dep_path: str | None = None, relationship: str = "consumed") -> None:
        deps.append(
            ArtifactDependency(
                dependency_type="artifact",
                identifier=identifier,
                path=dep_path or identifier,
                relationship=relationship,
            )
        )

    if name in {"report.md", "synthesis.json", "synthesis.md"}:
        for dep in ("context_packs.json", "notes.md", "sources.json"):
            add(dep)
    if "retrieval" in name or "context_pack" in name:
        for dep in ("document_chunks.json", "sources.json"):
            add(dep)
    if name.startswith("document_") or name in {"document_chunks.json", "documents.json"}:
        add("sources.json")
    if name.startswith("source_audit") or name == "citation_readiness.json":
        add("sources.json")
    if "verification" in name or "confidence_calibration" in name:
        for dep in ("report.md", "evidence_ledger.json"):
            add(dep)
    if "evaluation" in name or name == "quality_score.json":
        for dep in ("verification_report.json", "report.md", "sources.json"):
            add(dep)
    if name.startswith("evidence_") or "citation" in name:
        for dep in ("report.md", "sources.json"):
            add(dep)
    if name.startswith("source_") and name != "sources.json":
        deps.append(
            ArtifactDependency(
                dependency_type="input",
                identifier="input.urls",
                relationship="seeded",
            )
        )
    if path in PROVENANCE_ARTIFACTS:
        add("all_artifacts", relationship="scanned")
    return deps


def build_source_fingerprints(thread_dir: Path) -> list[SourceFingerprint]:
    path = thread_dir / "sources.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    fingerprints: list[SourceFingerprint] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        local_path = item.get("local_path")
        local_rel = _local_path_to_rel(thread_dir, str(local_path or ""))
        local_hash = None
        if local_rel:
            source_file = (thread_dir / local_rel).resolve()
            if source_file.exists() and source_file.is_file():
                local_hash = file_sha256(source_file)
        url = str(item.get("url") or item.get("final_url") or "")
        source_id = str(item.get("source_id") or item.get("id") or f"S{idx}")
        fingerprints.append(
            SourceFingerprint(
                source_id=source_id,
                url=url,
                normalized_url=str(item.get("normalized_url") or url),
                title=item.get("title"),
                fetched_at=item.get("fetched_at"),
                local_path=local_rel,
                content_hash=local_hash or item.get("content_hash"),
                metadata_hash=stable_hash(redact_secrets(item)),
                live_dependency=url.startswith(("http://", "https://")),
                may_have_changed=url.startswith(("http://", "https://")),
            )
        )
    return fingerprints


def _local_path_to_rel(thread_dir: Path, local_path: str) -> str | None:
    if not local_path:
        return None
    marker = f"runs/{thread_dir.name}/"
    if marker in local_path:
        local_path = local_path.split(marker, 1)[-1]
    if local_path.startswith("/") or "\\" in local_path or ".." in local_path.split("/"):
        return None
    candidate = (thread_dir / local_path).resolve()
    try:
        return safe_relative_path(thread_dir, candidate)
    except ProvenancePathError:
        return None


def build_model_fingerprints(
    settings_snapshot: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[ModelInvocationFingerprint]:
    invocations: dict[tuple[str, str, str], ModelInvocationFingerprint] = {}
    provider = str(settings_snapshot.get("model_provider") or "")
    model_name = _model_name_for_provider(provider, settings_snapshot)
    if provider or model_name:
        _add_model_invocation(
            invocations,
            provider,
            model_name,
            "configured_model",
            settings_snapshot,
        )
    for event in events:
        if event.get("event_type") != "model_call_started":
            continue
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        provider = str(metadata.get("provider") or settings_snapshot.get("model_provider") or "")
        model_name = str(
            metadata.get("model_name") or _model_name_for_provider(provider, settings_snapshot)
        )
        purpose = str(event.get("message") or "model_call")
        _add_model_invocation(invocations, provider, model_name, purpose, settings_snapshot)
    return sorted(
        invocations.values(),
        key=lambda item: (item.provider, item.model_name, item.purpose),
    )


def _model_name_for_provider(provider: str, settings_snapshot: dict[str, Any]) -> str:
    if provider == "ollama":
        return str(settings_snapshot.get("ollama_model") or "")
    if provider == "mock":
        return str(settings_snapshot.get("mock_model_name") or "deterministic-mock-research-model")
    return str(settings_snapshot.get("openai_model") or "")


def _add_model_invocation(
    invocations: dict[tuple[str, str, str], ModelInvocationFingerprint],
    provider: str,
    model_name: str,
    purpose: str,
    settings_snapshot: dict[str, Any],
) -> None:
    redacted = redact_secrets(
        {
            "provider": provider,
            "model_name": model_name,
            "temperature": settings_snapshot.get("temperature"),
            "openai_base_url": settings_snapshot.get("openai_base_url"),
            "openai_max_tokens": settings_snapshot.get("openai_max_tokens"),
            "openai_timeout_s": settings_snapshot.get("openai_timeout_s"),
            "openai_max_retries": settings_snapshot.get("openai_max_retries"),
            "ollama_num_predict": settings_snapshot.get("ollama_num_predict"),
        }
    )
    credentials = []
    if provider == "openai":
        credentials.append("OPENAI_API_KEY")
    key = (provider, model_name, purpose)
    invocations[key] = ModelInvocationFingerprint(
        provider=provider,
        model_name=model_name,
        purpose=purpose,
        config_hash=stable_hash(redacted),
        redacted_config=redacted,
        nondeterministic=provider != "mock",
        credentials_required=credentials,
    )


def read_events(thread_dir: Path) -> list[dict[str, Any]]:
    path = thread_dir / "events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def build_subsystem_invocations(artifact_paths: list[str]) -> list[SubsystemInvocation]:
    by_subsystem: dict[str, SubsystemInvocation] = {}
    for path in artifact_paths:
        subsystem = producer_for_path(path)
        current = by_subsystem.setdefault(
            subsystem,
            SubsystemInvocation(subsystem_name=subsystem, subsystem_version=SUBSYSTEM_VERSION),
        )
        current.output_artifacts.append(path)
        current.input_artifacts = sorted(
            {
                dep.path or dep.identifier
                for output in current.output_artifacts
                for dep in infer_artifact_dependencies(output)
                if dep.dependency_type == "artifact"
            }
        )
    return sorted(by_subsystem.values(), key=lambda item: item.subsystem_name)


def utc_timestamp_for_file(path: Path, attr: str) -> str:
    ts = path.stat().st_ctime if attr == "created" else path.stat().st_mtime
    return now_iso_utc_from_epoch(ts)


def now_iso_utc_from_epoch(epoch: float) -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))
