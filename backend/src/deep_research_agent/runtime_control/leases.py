from __future__ import annotations

from .repository import RuntimeRepository


class LeaseManager:
    def __init__(self, repository: RuntimeRepository):
        self.repository = repository

    def expire_stale(self):
        return self.repository.expire_stale_leases()

