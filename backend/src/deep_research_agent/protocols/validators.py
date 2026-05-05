from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import PolicyPack, ResearchProtocol
from .errors import PolicyPackLoadError


def validate_protocol(protocol: ResearchProtocol) -> ResearchProtocol:
    if protocol.safety_warnings.require_human_review:
        warnings = protocol.operator_review_required_when or protocol.safety_warnings.operator_review_required_when
        if not warnings:
            raise ValueError(
                f"{protocol.protocol_id} requires human review but has no review conditions"
            )
    if not protocol.required_artifacts:
        raise ValueError(f"{protocol.protocol_id} must declare required artifacts")
    return protocol


def validate_policy_pack(pack: PolicyPack) -> PolicyPack:
    if not pack.applies_to_protocols:
        raise ValueError(f"{pack.pack_id} must apply to at least one protocol")
    return pack


def load_policy_packs_from_json(path: Path) -> list[PolicyPack]:
    if not path.exists():
        return []
    if not path.is_file() or path.suffix.lower() != ".json":
        raise PolicyPackLoadError(f"Policy pack path must be a JSON file: {path}")
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PolicyPackLoadError(f"Could not read policy pack file {path}: {exc}") from exc

    items = raw.get("packs") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise PolicyPackLoadError(f"Policy pack file {path} must contain a list or packs list")

    packs: list[PolicyPack] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise PolicyPackLoadError(f"Policy pack entry {idx} in {path} is not an object")
        try:
            packs.append(validate_policy_pack(PolicyPack(**item)))
        except (ValidationError, ValueError) as exc:
            raise PolicyPackLoadError(f"Invalid policy pack entry {idx} in {path}: {exc}") from exc
    return packs

