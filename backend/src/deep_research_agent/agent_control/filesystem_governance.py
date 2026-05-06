from __future__ import annotations

import json
from pathlib import Path

from .artifact_writer import render_contracts, write_json_artifact, write_text_artifact
from .contracts import AgentOutputValidation, ArtifactContract, ResearchAgentRole

FORBIDDEN_REPORT_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "developer message",
    "reveal secrets",
    "delete files",
]


def built_in_artifact_contracts() -> list[ArtifactContract]:
    return [
        ArtifactContract(
            artifact_name="plan.md",
            producer_role=ResearchAgentRole.PLANNER,
            required=True,
            format="markdown",
            min_size_bytes=8,
            must_contain_sections=["plan"],
            description="Required planning artifact.",
        ),
        ArtifactContract(
            artifact_name="notes.md",
            producer_role=ResearchAgentRole.SOURCE_READER,
            required=True,
            format="markdown",
            min_size_bytes=8,
            description="Required source notes artifact.",
        ),
        ArtifactContract(
            artifact_name="sources.json",
            producer_role=ResearchAgentRole.SOURCE_TRIAGER,
            required=True,
            format="json",
            min_size_bytes=2,
            description="Required valid JSON source manifest.",
        ),
        ArtifactContract(
            artifact_name="report.md",
            producer_role=ResearchAgentRole.FINAL_EDITOR,
            required=True,
            format="markdown",
            min_size_bytes=8,
            forbidden_patterns=FORBIDDEN_REPORT_PATTERNS,
            description="Required final report.",
        ),
        *[
            ArtifactContract(
                artifact_name=name,
                producer_role=ResearchAgentRole.SUPERVISOR,
                required=name not in {"agent_output_validation.json", "agent_control_summary.json"},
                format="json" if name.endswith(".json") else "markdown",
                min_size_bytes=2,
                description="Control-plane audit artifact.",
            )
            for name in [
                "agent_control_plan.json",
                "skill_selection.json",
                "agent_policies.json",
                "context_bundles.json",
                "compiled_instructions.json",
                "subagent_specs.json",
                "agent_handoffs.json",
                "policy_violations.json",
                "agent_output_validation.json",
                "agent_control_summary.json",
            ]
        ],
    ]


class FilesystemGovernance:
    def __init__(self, contracts: list[ArtifactContract] | None = None):
        self.contracts = contracts or built_in_artifact_contracts()

    def path_is_safe(self, rel_path: str) -> bool:
        return (
            bool(rel_path)
            and not rel_path.startswith("/")
            and "\\" not in rel_path
            and ".." not in rel_path
        )

    def validate_contract(
        self,
        *,
        thread_id: str,
        run_dir: Path,
        contract: ArtifactContract,
        producer_role: ResearchAgentRole | None = None,
    ) -> AgentOutputValidation:
        errors: list[str] = []
        warnings: list[str] = []
        if not self.path_is_safe(contract.artifact_name):
            errors.append("unsafe artifact path")
            return AgentOutputValidation(
                thread_id=thread_id,
                role=producer_role or contract.producer_role,
                artifact_name=contract.artifact_name,
                expected=contract.required,
                exists=False,
                valid=False,
                errors=errors,
            )
        path = run_dir / contract.artifact_name
        exists = path.exists() and path.is_file()
        valid = exists
        if contract.required and not exists:
            errors.append("required artifact missing")
        if exists:
            size = path.stat().st_size
            if size < contract.min_size_bytes:
                errors.append(f"artifact too small: {size} bytes")
            if size > contract.max_size_bytes:
                warnings.append(f"artifact oversized: {size} bytes")
            text = path.read_text(encoding="utf-8", errors="ignore")
            if contract.format == "json":
                try:
                    json.loads(text)
                except Exception as e:
                    errors.append(f"invalid JSON: {e}")
            elif not text.strip():
                errors.append("markdown artifact is empty")
            lower = text.lower()
            for pattern in contract.forbidden_patterns:
                if pattern.lower() in lower:
                    warnings.append(f"forbidden pattern present: {pattern}")
            for section in contract.must_contain_sections:
                if section.lower() not in lower:
                    warnings.append(f"expected section not found: {section}")
            if producer_role is not None and producer_role != contract.producer_role:
                warnings.append(
                    "producer mismatch: expected "
                    f"{contract.producer_role.value}, got {producer_role.value}"
                )
        valid = exists and not errors
        return AgentOutputValidation(
            thread_id=thread_id,
            role=producer_role or contract.producer_role,
            artifact_name=contract.artifact_name,
            expected=contract.required,
            exists=exists,
            valid=valid,
            errors=errors,
            warnings=warnings,
        )

    def validate_run(self, *, thread_id: str, run_dir: Path) -> list[AgentOutputValidation]:
        return [
            self.validate_contract(thread_id=thread_id, run_dir=run_dir, contract=contract)
            for contract in self.contracts
        ]

    def write_contract_artifacts(self, runs_dir: Path, thread_id: str) -> list[str]:
        return [
            write_json_artifact(runs_dir, thread_id, "artifact_contracts.json", self.contracts),
            write_text_artifact(
                runs_dir, thread_id, "artifact_contracts.md", render_contracts(self.contracts)
            ),
        ]
