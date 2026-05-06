from __future__ import annotations

from .errors import RuntimeLeaseError
from .repository import RuntimeRepository


class RuntimeLockManager:
    def __init__(self, repository: RuntimeRepository):
        self.repository = repository

    def assert_not_locked(self, job_id: str) -> None:
        lease = self.repository.get_active_lease(job_id)
        if lease is not None:
            raise RuntimeLeaseError(f"Job has active lease: {job_id}")

