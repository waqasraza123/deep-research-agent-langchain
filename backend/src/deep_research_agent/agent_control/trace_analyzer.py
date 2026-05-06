from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_writer import (
    append_trace_jsonl,
    render_violations,
    write_json_artifact,
    write_text_artifact,
)
from .context_quarantine import ContextQuarantineManager
from .contracts import (
    AgentTraceEvent,
    AgentTraceEventType,
    PolicyViolation,
    ResearchAgentRole,
    model_to_plain,
)
from .policy import PolicyEngine


class TraceAnalyzer:
    def __init__(self, *, policy_engine: PolicyEngine | None = None):
        self.policy_engine = policy_engine or PolicyEngine()

    def append_trace_event(self, runs_dir: Path, event: AgentTraceEvent) -> str:
        return append_trace_jsonl(runs_dir, event.thread_id, event)

    def read_trace_events(self, run_dir: Path, thread_id: str) -> list[AgentTraceEvent]:
        path = run_dir / "agent_trace.jsonl"
        if not path.exists():
            events = self._synthetic_events_from_artifacts(run_dir, thread_id)
            return events
        out: list[AgentTraceEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(AgentTraceEvent(**json.loads(line)))
            except Exception:
                continue
        return out

    def analyze_trace(
        self, *, thread_id: str, run_dir: Path, events: list[AgentTraceEvent] | None = None
    ) -> dict[str, Any]:
        events = events if events is not None else self.read_trace_events(run_dir, thread_id)
        violations = []
        violations.extend(self.detect_policy_violations(events))
        violations.extend(self.detect_unexpected_artifact_writes(events))
        role_confusion = self.detect_role_confusion(events, run_dir)
        leakage = self.detect_source_content_leakage_into_instructions(run_dir)
        warnings = [*role_confusion, *leakage]
        return {
            "thread_id": thread_id,
            "event_count": len(events),
            "events": [model_to_plain(event) for event in events],
            "policy_violations": [model_to_plain(item) for item in violations],
            "role_confusion_warnings": role_confusion,
            "source_leakage_warnings": leakage,
            "warnings": warnings,
        }

    def detect_policy_violations(self, events: list[AgentTraceEvent]) -> list[PolicyViolation]:
        violations: list[PolicyViolation] = []
        for event in events:
            if event.event_type == AgentTraceEventType.TOOL_DENIED:
                violations.append(
                    PolicyViolation(
                        thread_id=event.thread_id,
                        role=event.role,
                        policy_type="tool",
                        message=event.message or "Denied tool use",
                        attempted_tool=event.tool_name,
                    )
                )
        return violations

    def detect_role_confusion(self, events: list[AgentTraceEvent], run_dir: Path) -> list[str]:
        warnings: list[str] = []
        for event in events:
            if (
                event.role == ResearchAgentRole.SOURCE_READER
                and event.artifact_name == "report.md"
                and event.event_type == AgentTraceEventType.ARTIFACT_WRITTEN
            ):
                warnings.append("source_reader wrote final report artifact")
            if event.role == ResearchAgentRole.SKEPTICAL_REVIEWER and event.artifact_name in {
                "report.md",
                "report.raw.md",
            }:
                warnings.append("skeptical_reviewer acted as synthesis writer")
        report = run_dir / "report.md"
        if report.exists():
            text = report.read_text(encoding="utf-8", errors="ignore").lower()
            if "ignore previous instructions" in text and "suspicious" not in text:
                warnings.append(
                    "report includes prompt-injection phrase without suspicious framing"
                )
        return sorted(set(warnings))

    def detect_unexpected_artifact_writes(
        self, events: list[AgentTraceEvent]
    ) -> list[PolicyViolation]:
        violations: list[PolicyViolation] = []
        for event in events:
            if event.event_type != AgentTraceEventType.ARTIFACT_WRITTEN or not event.artifact_name:
                continue
            if not self.policy_engine.is_artifact_write_allowed(event.role, event.artifact_name):
                violations.append(
                    PolicyViolation(
                        thread_id=event.thread_id,
                        role=event.role,
                        policy_type="filesystem",
                        message=f"Unexpected artifact write: {event.artifact_name}",
                        attempted_artifact=event.artifact_name,
                    )
                )
        return violations

    def detect_missing_expected_outputs(self, run_dir: Path, expected: list[str]) -> list[str]:
        return [name for name in expected if not (run_dir / name).exists()]

    def detect_tool_use_anomalies(self, events: list[AgentTraceEvent]) -> list[str]:
        counts: dict[tuple[ResearchAgentRole, str], int] = {}
        for event in events:
            if event.tool_name:
                key = (event.role, event.tool_name)
                counts[key] = counts.get(key, 0) + 1
        return [
            f"{role.value}:{tool} used {count} times"
            for (role, tool), count in counts.items()
            if count > 50
        ]

    def detect_source_content_leakage_into_instructions(self, run_dir: Path) -> list[str]:
        warnings: list[str] = []
        compiled = run_dir / "compiled_instructions.json"
        if not compiled.exists():
            return warnings
        text = compiled.read_text(encoding="utf-8", errors="ignore")
        findings = ContextQuarantineManager().detect_source_instruction_like_text(text)
        if findings and "<UNTRUSTED_SOURCE" in text:
            warnings.append("untrusted source wrapper leaked into compiled trusted instructions")
        return warnings

    def write_trace_artifacts(
        self,
        runs_dir: Path,
        thread_id: str,
        run_dir: Path,
        events: list[AgentTraceEvent] | None = None,
    ) -> tuple[dict[str, Any], list[PolicyViolation]]:
        analysis = self.analyze_trace(thread_id=thread_id, run_dir=run_dir, events=events)
        violations = [PolicyViolation(**item) for item in analysis["policy_violations"]]
        write_text_artifact(
            runs_dir,
            thread_id,
            "agent_trace.md",
            "# Agent Trace\n\n"
            + "\n".join(
                f"- {event['timestamp']} `{event['role']}` {event['event_type']} "
                f"{event.get('artifact_name') or event.get('tool_name') or ''}"
                for event in analysis.get("events", [])
            )
            + "\n",
        )
        write_json_artifact(runs_dir, thread_id, "trace_analysis.json", analysis)
        write_text_artifact(
            runs_dir, thread_id, "trace_analysis.md", self.render_trace_analysis(analysis)
        )
        write_json_artifact(runs_dir, thread_id, "policy_violations.json", violations)
        write_text_artifact(
            runs_dir, thread_id, "policy_violations.md", render_violations(violations)
        )
        return analysis, violations

    @staticmethod
    def render_trace_analysis(analysis: dict[str, Any]) -> str:
        lines = ["# Trace Analysis", "", f"- Events: {analysis.get('event_count', 0)}"]
        for warning in analysis.get("warnings", []):
            lines.append(f"- Warning: {warning}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _synthetic_events_from_artifacts(run_dir: Path, thread_id: str) -> list[AgentTraceEvent]:
        events: list[AgentTraceEvent] = []
        role_map = {
            "plan.md": ResearchAgentRole.PLANNER,
            "notes.md": ResearchAgentRole.SOURCE_READER,
            "sources.json": ResearchAgentRole.SOURCE_TRIAGER,
            "report.md": ResearchAgentRole.FINAL_EDITOR,
        }
        for name, role in role_map.items():
            if (run_dir / name).exists():
                events.append(
                    AgentTraceEvent(
                        thread_id=thread_id,
                        role=role,
                        event_type=AgentTraceEventType.ARTIFACT_WRITTEN,
                        artifact_name=name,
                        message="synthetic artifact event from filesystem",
                    )
                )
        return events
