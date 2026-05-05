from __future__ import annotations

from pathlib import Path

from .contracts import PolicyPack, ProtocolWarning
from .validators import load_policy_packs_from_json

PACKS_DIR = Path(__file__).resolve().parent / "packs"
DEFAULT_PACK_PATH = PACKS_DIR / "default_policy_packs.json"


def load_policy_packs(path: Path | None = None) -> list[PolicyPack]:
    return load_policy_packs_from_json(path or DEFAULT_PACK_PATH)


def policy_packs_for_protocol(protocol_id: str, packs: list[PolicyPack] | None = None) -> list[PolicyPack]:
    loaded = packs if packs is not None else load_policy_packs()
    return [pack for pack in loaded if protocol_id in pack.applies_to_protocols or "*" in pack.applies_to_protocols]


def warnings_for_policy_packs(packs: list[PolicyPack]) -> list[ProtocolWarning]:
    warnings: list[ProtocolWarning] = []
    for pack in packs:
        warnings.extend(pack.warnings)
    return warnings

