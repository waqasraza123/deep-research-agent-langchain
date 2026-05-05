from __future__ import annotations


class ProtocolError(Exception):
    """Base exception for protocol subsystem failures."""


class UnknownProtocolError(ProtocolError):
    def __init__(self, protocol_id: str):
        super().__init__(f"Unknown research protocol: {protocol_id}")
        self.protocol_id = protocol_id


class UnknownProfileError(ProtocolError):
    def __init__(self, profile_id: str):
        super().__init__(f"Unknown intelligence profile: {profile_id}")
        self.profile_id = profile_id


class PolicyPackLoadError(ProtocolError):
    pass

