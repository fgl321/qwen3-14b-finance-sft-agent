from __future__ import annotations

from decimal import Decimal

import pytest

from app.control_plane.enums import (
    CapabilityOutcomeStatus,
    DeliveryStatus,
    GuardState,
    RequestRiskClass,
    RuntimeCapabilityStatus,
    SystemHealth,
    TaskStatus,
)
from app.control_plane.reason_codes import ReasonCode
from app.control_plane.satisfaction import (
    CapabilityEvidence,
    TaskRequirementOutcome,
    aggregate_task_requirements,
    evaluate_capability,
)
from app.control_plane.schemas import CapabilityOutcome, CapabilitySatisfactionPolicy, TaskRequirement
from app.control_plane.status_assembler import (
    ComponentHealthEvent,
    DeliveryGuardOutcome,
    assemble_run_status,
    derive_delivery_status,
    derive_task_status,
)


def _policy(
    capability: str,
    *,
    outputs: tuple[str, ...],
    degradations: tuple[str, ...] = (),
) -> CapabilitySatisfactionPolicy:
    return CapabilitySatisfactionPolicy(
        capability=capability,
        policy_version="v1",
        acceptable_runtime_statuses=(
            RuntimeCapabilityStatus.AVAILABLE,
            RuntimeCapabilityStatus.DEGRADED,
        ),
        required_outputs=outputs,
        minimum_quality={"coverage": Decimal("0.8")},
        allowed_degradations=degradations,
    )


def test_capability_satisfaction_uses_outputs_quality_and_allowed_degradation() -> None:
    policy = _policy(
        "knowledge_retrieval",
        outputs=("verified_evidence", "verified_citations"),
        degradations=("reranker_failed_hybrid_fallback",),
    )
    evidence = CapabilityEvidence(
        capability="knowledge_retrieval",
        output_refs=("citation:1",),
        output_types=("verified_evidence", "verified_citations"),
        quality={"coverage": Decimal("0.9")},
        runtime_status=RuntimeCapabilityStatus.DEGRADED,
        degradation_used="reranker_failed_hybrid_fallback",
    )
    outcome = evaluate_capability(policy=policy, evidence=evidence, required=True)
    assert outcome.outcome == CapabilityOutcomeStatus.SATISFIED
    assert ReasonCode.CAPABILITY_DEGRADED in outcome.reason_codes

    provisional = evidence.model_copy(
        update={
            "output_types": ("provisional_evidence",),
            "degradation_used": "assessor_protocol_failed",
        }
    )
    incomplete = evaluate_capability(policy=policy, evidence=provisional, required=True)
    assert incomplete.outcome == CapabilityOutcomeStatus.UNSATISFIED


def test_task_aggregation_respects_dependency_dag() -> None:
    retrieval = TaskRequirement(
        task_id="retrieve",
        description="retrieve evidence",
        capabilities=("knowledge_retrieval",),
    )
    synthesis = TaskRequirement(
        task_id="synthesize",
        description="synthesize answer",
        capabilities=("complex_reasoning",),
        depends_on=("retrieve",),
    )
    outcomes = (
        CapabilityOutcome(
            capability="knowledge_retrieval",
            required=True,
            outcome=CapabilityOutcomeStatus.UNSATISFIED,
            runtime_status=RuntimeCapabilityStatus.AVAILABLE,
            reason_codes=(ReasonCode.RETRIEVAL_NO_EVIDENCE,),
        ),
        CapabilityOutcome(
            capability="complex_reasoning",
            required=True,
            outcome=CapabilityOutcomeStatus.SATISFIED,
            actual_output_refs=("answer:draft",),
            runtime_status=RuntimeCapabilityStatus.AVAILABLE,
        ),
    )
    task_outcomes = aggregate_task_requirements(
        requirements=(retrieval, synthesis),
        capability_outcomes=outcomes,
    )
    assert task_outcomes[0].outcome == CapabilityOutcomeStatus.UNSATISFIED
    assert task_outcomes[1].outcome == CapabilityOutcomeStatus.UNSATISFIED


def test_task_status_distinguishes_blocked_failed_partial_completed() -> None:
    satisfied = TaskRequirementOutcome(
        task_id="tool",
        required=True,
        outcome=CapabilityOutcomeStatus.SATISFIED,
    )
    partial = TaskRequirementOutcome(
        task_id="rag",
        required=True,
        outcome=CapabilityOutcomeStatus.PARTIALLY_SATISFIED,
    )
    unmet = partial.model_copy(update={"outcome": CapabilityOutcomeStatus.UNSATISFIED})
    assert derive_task_status(legally_started=False, task_outcomes=(unmet,)) == TaskStatus.BLOCKED
    assert derive_task_status(legally_started=True, task_outcomes=(unmet,)) == TaskStatus.FAILED
    assert derive_task_status(legally_started=True, task_outcomes=(satisfied, unmet)) == TaskStatus.PARTIAL
    assert derive_task_status(legally_started=True, task_outcomes=(partial,)) == TaskStatus.PARTIAL
    assert derive_task_status(legally_started=True, task_outcomes=(satisfied,)) == TaskStatus.COMPLETED


@pytest.mark.parametrize(
    ("risk", "guard", "verified", "subset", "expected"),
    [
        (RequestRiskClass.LOW, GuardState.PASSED, True, False, DeliveryStatus.VALIDATED),
        (RequestRiskClass.MEDIUM, GuardState.PROTOCOL_DEGRADED, True, True, DeliveryStatus.GUARD_DEGRADED),
        (RequestRiskClass.MEDIUM, GuardState.PROTOCOL_DEGRADED, False, False, DeliveryStatus.NOT_GENERATED),
        (RequestRiskClass.HIGH, GuardState.PROTOCOL_DEGRADED, True, True, DeliveryStatus.REJECTED),
        (RequestRiskClass.LOW, GuardState.REJECTED, True, True, DeliveryStatus.REJECTED),
    ],
)
def test_delivery_matrix(
    risk: RequestRiskClass,
    guard: GuardState,
    verified: bool,
    subset: bool,
    expected: DeliveryStatus,
) -> None:
    result = derive_delivery_status(
        DeliveryGuardOutcome(
            guard_state=guard,
            risk_class=risk,
            verified_content_available=verified,
            verified_subset_available=subset,
        )
    )
    assert result == expected


def test_status_assembler_does_not_turn_router_degradation_into_false_partial() -> None:
    task = TaskRequirementOutcome(
        task_id="all_required",
        required=True,
        outcome=CapabilityOutcomeStatus.SATISFIED,
    )
    status = assemble_run_status(
        legally_started=True,
        task_outcomes=(task,),
        component_events=(
            ComponentHealthEvent(
                component="semantic_router",
                health=SystemHealth.DEGRADED,
                reason_codes=(ReasonCode.ROUTER_PROTOCOL_DEGRADED,),
            ),
        ),
        guard_outcome=DeliveryGuardOutcome(
            guard_state=GuardState.PASSED,
            risk_class=RequestRiskClass.MEDIUM,
            verified_content_available=True,
        ),
    )
    assert status.task_status == TaskStatus.COMPLETED
    assert status.system_health == SystemHealth.DEGRADED
    assert status.delivery_status == DeliveryStatus.VALIDATED
    assert status.legacy_overall_status == "completed"


def test_normal_no_evidence_is_partial_but_system_healthy() -> None:
    tool = TaskRequirementOutcome(
        task_id="tool",
        required=True,
        outcome=CapabilityOutcomeStatus.SATISFIED,
    )
    rag = TaskRequirementOutcome(
        task_id="rag",
        required=True,
        outcome=CapabilityOutcomeStatus.UNSATISFIED,
        reason_codes=(ReasonCode.RETRIEVAL_NO_EVIDENCE,),
    )
    status = assemble_run_status(
        legally_started=True,
        task_outcomes=(tool, rag),
        component_events=(),
        guard_outcome=DeliveryGuardOutcome(
            guard_state=GuardState.PASSED_WITH_LIMITATIONS,
            risk_class=RequestRiskClass.MEDIUM,
            verified_content_available=True,
            verified_subset_available=True,
        ),
    )
    assert status.task_status == TaskStatus.PARTIAL
    assert status.system_health == SystemHealth.HEALTHY
    assert status.delivery_status == DeliveryStatus.VALIDATED_WITH_LIMITATIONS
    assert status.primary_reason_code == ReasonCode.RETRIEVAL_NO_EVIDENCE
