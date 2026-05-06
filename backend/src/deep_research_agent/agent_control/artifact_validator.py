from __future__ import annotations

from pathlib import Path

from .artifact_writer import render_validations, write_json_artifact, write_text_artifact
from .contracts import AgentOutputValidation, ResearchAgentRole
from .filesystem_governance import FilesystemGovernance, built_in_artifact_contracts


class ArtifactValidator:
    def __init__(self, governance: FilesystemGovernance | None = None):
        self.governance = governance or FilesystemGovernance(built_in_artifact_contracts())

    def validate_outputs(
        self,
        *,
        thread_id: str,
        run_dir: Path,
        producer_roles: dict[str, ResearchAgentRole] | None = None,
    ) -> list[AgentOutputValidation]:
        producer_roles = producer_roles or {}
        validations = []
        for contract in self.governance.contracts:
            validations.append(
                self.governance.validate_contract(
                    thread_id=thread_id,
                    run_dir=run_dir,
                    contract=contract,
                    producer_role=producer_roles.get(contract.artifact_name),
                )
            )
        return validations

    def write_validation_artifacts(
        self, runs_dir: Path, thread_id: str, validations: list[AgentOutputValidation]
    ) -> list[str]:
        return [
            write_json_artifact(runs_dir, thread_id, "agent_output_validation.json", validations),
            write_text_artifact(
                runs_dir,
                thread_id,
                "agent_output_validation.md",
                render_validations(validations),
            ),
        ]

    @staticmethod
    def missing_required(validations: list[AgentOutputValidation]) -> list[str]:
        return [
            item.artifact_name
            for item in validations
            if item.expected and (not item.exists or "required artifact missing" in item.errors)
        ]

    @staticmethod
    def invalid_outputs(validations: list[AgentOutputValidation]) -> list[str]:
        return [item.artifact_name for item in validations if item.exists and not item.valid]
