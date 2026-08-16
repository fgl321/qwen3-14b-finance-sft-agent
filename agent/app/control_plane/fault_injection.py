from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, TypeVar


class FaultKind(StrEnum):
    NONE = "none"
    PROTOCOL_FAILURE = "protocol_failure"
    SERVICE_FAILURE = "service_failure"
    TIMEOUT = "timeout"
    EMPTY_RESULT = "empty_result"


class InjectedFault(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FaultSpec:
    component: str
    kind: FaultKind


T = TypeVar("T")


class FaultInjector:
    """Deterministic, test-only fault boundary; disabled unless explicitly populated."""

    def __init__(self, faults: tuple[FaultSpec, ...] = ()) -> None:
        self._faults = {item.component: item.kind for item in faults}

    def kind_for(self, component: str) -> FaultKind:
        return self._faults.get(component, FaultKind.NONE)

    def invoke(self, component: str, operation: Callable[[], T]) -> T | None:
        kind = self.kind_for(component)
        if kind == FaultKind.EMPTY_RESULT:
            return None
        if kind != FaultKind.NONE:
            raise InjectedFault(f"{component}:{kind.value}")
        return operation()
