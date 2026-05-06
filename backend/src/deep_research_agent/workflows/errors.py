class WorkflowError(Exception):
    """Base workflow runtime error."""


class WorkflowCompilationError(WorkflowError):
    """Raised when a workflow cannot be compiled safely."""


class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution cannot continue."""


class UnsafeWorkflowPathError(WorkflowError):
    """Raised for unsafe workflow artifact or template paths."""
