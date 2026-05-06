from __future__ import annotations

from pathlib import Path

from .artifact_writer import RuntimeArtifactWriter
from .contracts import (
    ORDERED_STAGES,
    RecoveryPlan,
    ResearchJobStatus,
    ResearchStage,
    RuntimeEventType,
)
from .events import RuntimeEventWriter
from .repository import RuntimeRepository

REQUIRED_ARTIFACTS = ("plan.md", "notes.md", "sources.json", "report.md")


class ResumeInspector:
    def __init__(self, *, repository: RuntimeRepository, runs_dir: Path):
        self.repository = repository
        self.runs_dir = runs_dir

    def inspect_job(self, job_id: str) -> RecoveryPlan:
        job = self.repository.get_job(job_id)
        return self.inspect_run_directory(job.thread_id, job_id=job.job_id)

    def inspect_run_directory(
        self, thread_id: str, *, job_id: str | None = None
    ) -> RecoveryPlan:
        job = (
            self.repository.get_job_by_thread_id(thread_id)
            if job_id is None
            else self.repository.get_job(job_id)
        )
        completed = [
            record.stage
            for record in self.repository.get_stage_records(job.job_id)
            if record.status in {"completed", "skipped"}
        ]
        next_stage = self.determine_next_stage(completed)
        missing = self.detect_missing_required_artifacts(thread_id)
        unsafe: list[str] = []
        if next_stage in {None, ResearchStage.FINALIZATION} and missing:
            unsafe.append("required final artifacts are missing")
        plan = RecoveryPlan(
            job_id=job.job_id,
            thread_id=thread_id,
            resumable=not unsafe,
            next_stage=next_stage,
            completed_stages=completed,
            missing_artifacts=missing,
            unsafe_to_resume_reasons=unsafe,
            recommended_action="resume" if not unsafe else "rebuild_missing_artifacts",
        )
        self.write_recovery_plan(plan)
        return plan

    def determine_next_stage(self, completed_stages: list[ResearchStage]) -> ResearchStage | None:
        completed = set(completed_stages)
        for stage in ORDERED_STAGES:
            if stage not in completed:
                return stage
        return None

    def detect_missing_required_artifacts(self, thread_id: str) -> list[str]:
        td = self.runs_dir / thread_id
        return [rel for rel in REQUIRED_ARTIFACTS if not (td / rel).exists()]

    def detect_completed_stage_outputs(self, job_id: str) -> dict[str, list[str]]:
        return {
            record.stage.value: record.output_artifacts
            for record in self.repository.get_stage_records(job_id)
            if record.status in {"completed", "skipped"}
        }

    def build_recovery_plan(self, job_id: str) -> RecoveryPlan:
        return self.inspect_job(job_id)

    def write_recovery_plan(self, plan: RecoveryPlan) -> None:
        writer = RuntimeArtifactWriter(runs_dir=self.runs_dir, thread_id=plan.thread_id)
        writer.write_json("runtime_recovery_plan.json", plan)
        writer.write_text(
            "runtime_recovery_plan.md",
            "# Runtime Recovery Plan\n\n"
            f"- Resumable: {plan.resumable}\n"
            f"- Next stage: `{plan.next_stage}`\n"
            f"- Missing artifacts: {', '.join(plan.missing_artifacts) or 'none'}\n"
            f"- Recommended action: {plan.recommended_action}\n",
        )


class RuntimeRecoveryService:
    def __init__(self, *, repository: RuntimeRepository, runs_dir: Path):
        self.repository = repository
        self.runs_dir = runs_dir
        self.events = RuntimeEventWriter(repository=repository, runs_dir=runs_dir)
        self.inspector = ResumeInspector(repository=repository, runs_dir=runs_dir)

    def recover_stale_leases(self) -> dict:
        expired = self.repository.expire_stale_leases()
        recovered = []
        failed = []
        for lease in expired:
            job = self.repository.get_job(lease.job_id)
            self.events.emit(
                job,
                RuntimeEventType.RECOVERY_DETECTED,
                severity="warning",
                message="stale lease detected",
                data={"lease_id": lease.lease_id},
            )
            plan = self.inspector.inspect_job(job.job_id)
            if plan.resumable:
                self.repository.mark_job_status(
                    job.job_id, ResearchJobStatus.QUEUED, validate=False
                )
                self.repository.enqueue_job(job.job_id)
                recovered.append(job.job_id)
                self.events.emit(job, RuntimeEventType.RECOVERY_APPLIED, message="job requeued")
            else:
                self.repository.mark_failed(job.job_id)
                failed.append(job.job_id)
        return {"expired_leases": len(expired), "requeued": recovered, "failed": failed}
