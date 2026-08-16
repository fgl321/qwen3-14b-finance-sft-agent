from __future__ import annotations

from app.control_plane.enums import (
    CapabilityOutcomeStatus,
    DeliveryStatus,
    GuardState,
    RequestRiskClass,
    SystemHealth,
    TaskStatus,
)
from app.control_plane.reason_codes import ReasonCode
from app.control_plane.satisfaction import TaskRequirementOutcome
from app.control_plane.schemas import FinalRunStatus, FrozenModel


class ComponentHealthEvent(FrozenModel):
    component: str
    health: SystemHealth
    critical: bool = False
    reason_codes: tuple[ReasonCode, ...] = ()


class DeliveryGuardOutcome(FrozenModel):
    guard_state: GuardState
    risk_class: RequestRiskClass
    verified_content_available: bool
    verified_subset_available: bool = False
    reason_codes: tuple[ReasonCode, ...] = ()


_PRIMARY_REASON_PRIORITY = {
    ReasonCode.CONTRACT_BLOCKED_BY_POLICY: 100,
    ReasonCode.CONTRACT_PERMISSION_CONFLICT: 95,
    ReasonCode.SCOPE_RESOLUTION_FAILED: 90,
    ReasonCode.SCOPE_EXECUTION_PRECONDITION_FAILED: 88,
    ReasonCode.CAPABILITY_UNAVAILABLE: 80,
    ReasonCode.TOOL_RESULT_UNKNOWN: 78,
    ReasonCode.TOOL_EXECUTION_FAILED: 75,
    ReasonCode.RETRIEVAL_NO_EVIDENCE: 70,
    ReasonCode.CONTRACT_REQUIRED_CAPABILITY_MISSING: 65,
    ReasonCode.DEADLINE_EXCEEDED: 60,
    ReasonCode.BUDGET_EXHAUSTED: 55,
    ReasonCode.RUN_CANCELLED: 50,
}


def derive_task_status(
    *,
    legally_started: bool,
    task_outcomes: tuple[TaskRequirementOutcome, ...],
) -> TaskStatus:
    required = [item for item in task_outcomes if item.required]
    if not required:
        return TaskStatus.COMPLETED
    satisfied = sum(item.outcome == CapabilityOutcomeStatus.SATISFIED for item in required)
    progress = sum(
        item.outcome in {
            CapabilityOutcomeStatus.SATISFIED,
            CapabilityOutcomeStatus.PARTIALLY_SATISFIED,
        }
        for item in required
    )
    if satisfied == len(required):
        return TaskStatus.COMPLETED
    if not legally_started and progress == 0:
        return TaskStatus.BLOCKED
    if progress == 0:
        return TaskStatus.FAILED
    return TaskStatus.PARTIAL


def derive_system_health(
    events: tuple[ComponentHealthEvent, ...],
) -> tuple[SystemHealth, tuple[str, ...]]:
    failed_critical = any(
        event.critical and event.health == SystemHealth.FAILED for event in events
    )
    degraded_components = tuple(
        dict.fromkeys(
            event.component for event in events if event.health != SystemHealth.HEALTHY
        )
    )
    if failed_critical:
        return SystemHealth.FAILED, degraded_components
    if degraded_components:
        return SystemHealth.DEGRADED, degraded_components
    return SystemHealth.HEALTHY, ()


def derive_delivery_status(outcome: DeliveryGuardOutcome) -> DeliveryStatus:
    guard = outcome.guard_state
    risk = outcome.risk_class
    if guard == GuardState.REJECTED:
        return DeliveryStatus.REJECTED
    if guard == GuardState.PASSED:
        return (
            DeliveryStatus.VALIDATED
            if outcome.verified_content_available
            else DeliveryStatus.NOT_GENERATED
        )
    if guard == GuardState.PASSED_WITH_LIMITATIONS:
        return (
            DeliveryStatus.VALIDATED_WITH_LIMITATIONS
            if outcome.verified_content_available or outcome.verified_subset_available
            else DeliveryStatus.NOT_GENERATED
        )
    if guard in {GuardState.PROTOCOL_DEGRADED, GuardState.NOT_RUN}:
        if risk == RequestRiskClass.HIGH:
            return DeliveryStatus.REJECTED
        if risk == RequestRiskClass.MEDIUM:
            return (
                DeliveryStatus.GUARD_DEGRADED
                if outcome.verified_subset_available
                else DeliveryStatus.NOT_GENERATED
            )
        return (
            DeliveryStatus.GUARD_DEGRADED
            if outcome.verified_content_available or outcome.verified_subset_available
            else DeliveryStatus.NOT_GENERATED
        )
    return DeliveryStatus.NOT_GENERATED


def _legacy_status(task: TaskStatus, delivery: DeliveryStatus) -> str:
    if task == TaskStatus.COMPLETED:
        if delivery == DeliveryStatus.VALIDATED:
            return "completed"
        if delivery in {
            DeliveryStatus.VALIDATED_WITH_LIMITATIONS,
            DeliveryStatus.GUARD_DEGRADED,
        }:
            return "completed_with_limitations"
        return "delivery_rejected"
    return task.value


def assemble_run_status(
    *,
    legally_started: bool,
    task_outcomes: tuple[TaskRequirementOutcome, ...],
    component_events: tuple[ComponentHealthEvent, ...],
    guard_outcome: DeliveryGuardOutcome,
    additional_reason_codes: tuple[ReasonCode, ...] = (),
) -> FinalRunStatus:
    task_status = derive_task_status(
        legally_started=legally_started,
        task_outcomes=task_outcomes,
    )
    system_health, degraded_components = derive_system_health(component_events)
    delivery_status = derive_delivery_status(guard_outcome)
    reasons = tuple(
        dict.fromkeys(
            [
                *(code for item in task_outcomes for code in item.reason_codes),
                *(code for item in component_events for code in item.reason_codes),
                *guard_outcome.reason_codes,
                *additional_reason_codes,
            ]
        )
    )
    primary = max(
        reasons,
        key=lambda code: _PRIMARY_REASON_PRIORITY.get(code, 0),
        default=None,
    )
    return FinalRunStatus(
        task_status=task_status,
        system_health=system_health,
        delivery_status=delivery_status,
        degraded_components=degraded_components,
        primary_reason_code=primary,
        reason_codes=reasons,
        legacy_overall_status=_legacy_status(task_status, delivery_status),
    )
