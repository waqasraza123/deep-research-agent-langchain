from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .contracts import WorkflowInput, WorkflowMode, WorkflowStageType, WorkflowTemplate
from .templates import built_in_templates


class WorkflowTemplateRegistry:
    def __init__(self, templates: Iterable[WorkflowTemplate] | None = None):
        self._templates: dict[str, WorkflowTemplate] = {}
        for template in templates or built_in_templates():
            self.register_template(template)

    def list_templates(self) -> list[WorkflowTemplate]:
        return sorted(self._templates.values(), key=lambda item: item.template_id)

    def get_template(self, template_id: str) -> WorkflowTemplate:
        try:
            return self._templates[template_id]
        except KeyError as e:
            raise KeyError(f"Unknown workflow template: {template_id}") from e

    def get_template_for_mode(self, mode: WorkflowMode | str) -> WorkflowTemplate:
        resolved = WorkflowMode(mode)
        for template in self._templates.values():
            if template.mode == resolved and template.enabled:
                return template
        raise KeyError(f"No enabled workflow template for mode: {resolved.value}")

    def register_template(self, template: WorkflowTemplate) -> None:
        validate_template(template)
        self._templates[template.template_id] = template

    def select_template(self, workflow_input: WorkflowInput) -> WorkflowTemplate:
        if workflow_input.template_id:
            return self.get_template(workflow_input.template_id)
        if workflow_input.mode:
            return self.get_template_for_mode(workflow_input.mode)
        return self.get_template_for_mode(infer_mode_from_question(workflow_input))


def validate_template(template: WorkflowTemplate) -> None:
    if not template.template_id:
        raise ValueError("template_id is required")
    stage_ids = [stage.stage_id for stage in template.stages]
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError(f"Template {template.template_id} has duplicate stage ids")
    stage_id_set = set(stage_ids)
    for stage in template.stages:
        for dep in [*stage.depends_on, *stage.optional_depends_on]:
            if dep not in stage_id_set:
                raise ValueError(
                    f"Template {template.template_id} stage {stage.stage_id} "
                    f"depends on missing {dep}"
                )
    for policy in template.policies:
        denied = set(policy.denied_stage_types)
        for stage in template.stages:
            if stage.stage_type in denied:
                raise ValueError(
                    f"Template {template.template_id} contains denied stage {stage.stage_type}"
                )


def infer_mode_from_question(workflow_input: WorkflowInput) -> WorkflowMode:
    question = (workflow_input.question or "").lower()
    has_existing = bool(workflow_input.existing_run_id or workflow_input.existing_thread_id)
    if any(
        term in question for term in ("prompt injection", "malicious source", "source poisoning")
    ):
        return WorkflowMode.adversarial_source_review
    if has_existing and any(
        term in question for term in ("evaluate existing run", "evaluate artifacts")
    ):
        return WorkflowMode.evaluation_only
    if has_existing and any(term in question for term in ("verify", "fact check", "fact-check")):
        return WorkflowMode.verification_only
    if any(term in question for term in ("audit source", "check source", "source quality")):
        return WorkflowMode.source_audit_only
    if any(term in question for term in ("legal", "policy", "regulation", "compliance")):
        return WorkflowMode.legal_policy_review
    if any(term in question for term in ("implement", "build", "architecture", "plan")):
        return WorkflowMode.implementation_planning
    if any(term in question for term in ("compare", " vs ", " versus ", "better", "alternative")):
        if any(term in question for term in ("vendor", "pricing", "contract", "sla")):
            return WorkflowMode.vendor_evaluation
        return WorkflowMode.framework_comparison
    if len(question.split()) > 24 or len(workflow_input.urls) > 3:
        return WorkflowMode.deep_research
    return WorkflowMode.quick_brief


def load_custom_templates(
    templates_dir: Path | None,
    *,
    enabled: bool,
    allow_custom_stage_type: bool = False,
) -> list[WorkflowTemplate]:
    if not enabled:
        return []
    if templates_dir is None:
        return []
    root = templates_dir.resolve()
    if not root.exists():
        return []
    templates: list[WorkflowTemplate] = []
    for path in sorted(root.glob("*.json")):
        resolved = path.resolve()
        if not str(resolved).startswith(str(root)) or ".." in path.parts:
            raise ValueError("Unsafe custom template path")
        data = json.loads(path.read_text(encoding="utf-8"))
        template = WorkflowTemplate(**data)
        if not allow_custom_stage_type:
            for stage in template.stages:
                if stage.stage_type == WorkflowStageType.custom:
                    raise ValueError("Custom stage types are disabled")
        validate_template(template)
        templates.append(template)
    return templates


_DEFAULT_REGISTRY = WorkflowTemplateRegistry()


def list_templates() -> list[WorkflowTemplate]:
    return _DEFAULT_REGISTRY.list_templates()


def get_template(template_id: str) -> WorkflowTemplate:
    return _DEFAULT_REGISTRY.get_template(template_id)


def get_template_for_mode(mode: WorkflowMode | str) -> WorkflowTemplate:
    return _DEFAULT_REGISTRY.get_template_for_mode(mode)


def register_template(template: WorkflowTemplate) -> None:
    _DEFAULT_REGISTRY.register_template(template)


def select_template(workflow_input: WorkflowInput) -> WorkflowTemplate:
    return _DEFAULT_REGISTRY.select_template(workflow_input)
