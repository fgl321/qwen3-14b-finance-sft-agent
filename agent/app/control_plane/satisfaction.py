from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from pydantic import Field

from app.control_plane.enums import CapabilityOutcomeStatus, RuntimeCapabilityStatus
from app.control_plane.reason_codes import ReasonCode
from app.control_plane.schemas import (
    CapabilityOutcome,
    CapabilitySatisfactionPolicy,
    FrozenModel,
    TaskRequirement,
)


class CapabilityEvidence(FrozenModel):
    capability: str
    output_refs: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    quality: dict[str, Decimal | int | str] = Field(default_factory=dict)
    runtime_status: RuntimeCapabilityStatus
    degradation_used: str | None = None
    reason_codes: tuple[ReasonCode, ...] = ()


class TaskRequirementOutcome(FrozenModel):
    task_id: str
    required: bool
    outcome: CapabilityOutcomeStatus
    capability_outcomes: tuple[str, ...] = ()
    reason_codes: tuple[ReasonCode, ...] = ()


def _quality_satisfied(
    actual: dict[str, Decimal | int | str],
    minimum: dict[str, Decimal | int | str],
) -> bool:
    for key, expected in minimum.items():
        if key not in actual:
            return False
        value = actual[key]
        try:
            if Decimal(str(value)) < Decimal(str(expected)):
                return False
        except Exception:
            if str(value) != str(expected):
                return False
    return True


def evaluate_capability(
    *,
    policy: CapabilitySatisfactionPolicy,
    evidence: CapabilityEvidence | None,
    required: bool,
) -> CapabilityOutcome:
    if not required:
        return CapabilityOutcome(
            capability=policy.capability,
            required=False,
            outcome=CapabilityOutcomeStatus.NOT_REQUIRED,
            runtime_status=(evidence.runtime_status if evidence else RuntimeCapabilityStatus.UNKNOWN),
        )
    if evidence is None:
        return CapabilityOutcome(
            capability=policy.capability,
            required=True,
            outcome=CapabilityOutcomeStatus.UNSATISFIED,
            runtime_status=RuntimeCapabilityStatus.UNKNOWN,
            reason_codes=(ReasonCode.CONTRACT_REQUIRED_CAPABILITY_MISSING,),
        )
    available_outputs = set(evidence.output_types)
    required_outputs = set(policy.required_outputs)
    complete_outputs = required_outputs.issubset(available_outputs)
    some_outputs = bool(required_outputs & available_outputs) or (
        not required_outputs and bool(evidence.output_refs)
    )
    runtime_acceptable = evidence.runtime_status in policy.acceptable_runtime_statuses
    degradation_acceptable = (
        evidence.degradation_used is None
        or evidence.degradation_used in policy.allowed_degradations
    )
    quality_ok = _quality_satisfied(evidence.quality, policy.minimum_quality)
    if complete_outputs and runtime_acceptable and degradation_acceptable and quality_ok:
        outcome = CapabilityOutcomeStatus.SATISFIED
    elif some_outputs:
        outcome = CapabilityOutcomeStatus.PARTIALLY_SATISFIED
    else:
        outcome = CapabilityOutcomeStatus.UNSATISFIED
    reasons = list(evidence.reason_codes)
    if evidence.runtime_status == RuntimeCapabilityStatus.DEGRADED:
        reasons.append(ReasonCode.CAPABILITY_DEGRADED)
    elif evidence.runtime_status in {
        RuntimeCapabilityStatus.UNAVAILABLE,
        RuntimeCapabilityStatus.UNKNOWN,
    }:
        reasons.append(ReasonCode.CAPABILITY_UNAVAILABLE)
    if outcome != CapabilityOutcomeStatus.SATISFIED and not reasons:
        reasons.append(ReasonCode.CONTRACT_REQUIRED_CAPABILITY_MISSING)
    return CapabilityOutcome(
        capability=policy.capability,
        required=True,
        outcome=outcome,
        actual_output_refs=evidence.output_refs,
        runtime_status=evidence.runtime_status,
        allowed_degradation_used=evidence.degradation_used,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def aggregate_task_requirements(
    *,
    requirements: tuple[TaskRequirement, ...],
    capability_outcomes: tuple[CapabilityOutcome, ...],
) -> tuple[TaskRequirementOutcome, ...]:
    outcome_by_capability = {item.capability: item for item in capability_outcomes}
    results: dict[str, TaskRequirementOutcome] = {}
    remaining = {task.task_id: task for task in requirements}
    while remaining:
        progressed = False
        for task_id, task in tuple(remaining.items()):
            if any(dependency not in results for dependency in task.depends_on):
                continue
            dependencies = [results[dependency] for dependency in task.depends_on]
            capability_items = [outcome_by_capability.get(name) for name in task.capabilities]
            reasons: list[ReasonCode] = [
                code
                for item in capability_items
                if item is not None
                for code in item.reason_codes
            ]
            dependency_unsatisfied = any(
                item.outcome != CapabilityOutcomeStatus.SATISFIED
                for item in dependencies
            )
            if dependency_unsatisfied or any(item is None for item in capability_items):
                outcome = CapabilityOutcomeStatus.UNSATISFIED
                reasons.append(ReasonCode.CONTRACT_REQUIRED_CAPABILITY_MISSING)
            elif all(item.outcome == CapabilityOutcomeStatus.SATISFIED for item in capability_items):
                outcome = CapabilityOutcomeStatus.SATISFIED
            elif any(
                item.outcome in {
                    CapabilityOutcomeStatus.SATISFIED,
                    CapabilityOutcomeStatus.PARTIALLY_SATISFIED,
                }
                for item in capability_items
            ):
                outcome = CapabilityOutcomeStatus.PARTIALLY_SATISFIED
            else:
                outcome = CapabilityOutcomeStatus.UNSATISFIED
            results[task_id] = TaskRequirementOutcome(
                task_id=task_id,
                required=task.required,
                outcome=outcome,
                capability_outcomes=tuple(task.capabilities),
                reason_codes=tuple(dict.fromkeys(reasons)),
            )
            del remaining[task_id]
            progressed = True
        if not progressed:
            raise ValueError("task requirement graph has a cycle or unknown dependency")
    return tuple(results[task.task_id] for task in requirements)
