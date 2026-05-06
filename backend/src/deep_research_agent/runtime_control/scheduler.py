from __future__ import annotations

from .worker import RuntimeWorker


class RuntimeScheduler:
    def __init__(self, worker: RuntimeWorker):
        self.worker = worker

    def process_next(self):
        return self.worker.process_next_job()

