from __future__ import annotations


class RuntimeControlError(RuntimeError):
    pass


class RuntimeNotFoundError(RuntimeControlError):
    pass


class InvalidRuntimeTransitionError(RuntimeControlError):
    def __init__(self, current: str, target: str):
        super().__init__(f"Invalid runtime transition: {current} -> {target}")
        self.current = current
        self.target = target


class RuntimeLeaseError(RuntimeControlError):
    pass


class RuntimeBudgetExceeded(RuntimeControlError):
    def __init__(self, reasons: list[str]):
        super().__init__("Runtime budget exceeded: " + "; ".join(reasons))
        self.reasons = reasons

