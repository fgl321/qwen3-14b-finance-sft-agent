from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.control_plane.acceptance import AcceptanceCaseResult, build_acceptance_report
from app.control_plane.enums import (
    Authority,
    EnforcementStrength,
    InvocationStatus,
    PermissionLevel,
    RequirementLevel,
    RuntimeCapabilityStatus,
)
from app.control_plane.fault_injection import FaultInjector, FaultKind, FaultSpec, InjectedFault
from app.control_plane.metrics import ControlPlaneMetrics, RED_LINE_METRICS
from app.control_plane.schemas import (
    CONTROL_PLANE_SCHEMA_VERSION,
    CapabilityAvailability,
    CapabilityConstraint,
    ConstraintSource,
    PreliminaryStrategy,
    RuntimeCapabilitySnapshot,
    SemanticRequirementContract,
    TaskRequirement,
)
from app.control_plane.semantic_extractor import IndependentSemanticRequirementExtractor
from app.control_plane.shadow import ShadowCapabilityRegistry, ShadowControlPlane
from app.core.config import Settings


NOW = datetime(2026, 8, 14, tzinfo=UTC)


def _constraint(capability: str, requirement: RequirementLevel, permission: PermissionLevel = PermissionLevel.ALLOWED) -> CapabilityConstraint:
    return CapabilityConstraint(
        capability=capability,
        requirement=requirement,
        permission=permission,
        source=ConstraintSource(
            constraint_id=f"extractor:{capability}",
            authority=Authority.SEMANTIC_EXTRACTOR,
            enforcement_strength=EnforcementStrength.INFERRED,
            rule_id="semantic-v1",
        ),
    )


def _extractor(*constraints: CapabilityConstraint, status: InvocationStatus = InvocationStatus.SUCCESS) -> SemanticRequirementContract:
    value = SemanticRequirementContract(
        request_id="req",
        constraints=constraints,
        task_requirements=tuple(
            TaskRequirement(
                task_id=f"task_{item.capability}",
                description=f"perform {item.capability}",
                capabilities=(item.capability,),
                requires_citations=item.capability == "citation_validation",
            )
            for item in constraints if item.requirement == RequirementLevel.REQUIRED
        ),
        invocation_status=status,
    )
    return value.model_copy(update={"canonical_hash": value.calculate_hash()})


def _preliminary(*capabilities: str) -> PreliminaryStrategy:
    value = PreliminaryStrategy(
        request_id="req",
        orchestration_mode="hybrid",
        proposed_capabilities=capabilities,
        proposed_tasks=tuple(
            TaskRequirement(task_id=f"task_{cap}", description=cap, capabilities=(cap,))
            for cap in capabilities
        ),
        invocation_status=InvocationStatus.SUCCESS,
    )
    return value.model_copy(update={"canonical_hash": value.calculate_hash()})


def _snapshot(*capabilities: str) -> RuntimeCapabilitySnapshot:
    value = RuntimeCapabilitySnapshot(
        run_id="shadow-run",
        observed_at_utc=NOW,
        capabilities=tuple(
            CapabilityAvailability(
                capability=cap,
                provider_or_tool=f"fixture:{cap}",
                status=RuntimeCapabilityStatus.AVAILABLE,
                checked_at_utc=NOW,
            )
            for cap in capabilities
        ),
    )
    return value.model_copy(update={"canonical_hash": value.calculate_hash()})


def _metrics() -> ControlPlaneMetrics:
    return ControlPlaneMetrics(runtime_revision="test", schema_versions={"control_plane": CONTROL_PLANE_SCHEMA_VERSION})


def test_shadow_preserves_required_removes_forbidden_and_has_no_effect_interfaces() -> None:
    required = _constraint("knowledge_retrieval", RequirementLevel.REQUIRED)
    forbidden = _constraint("web_access", RequirementLevel.NOT_NEEDED, PermissionLevel.FORBIDDEN)
    metrics = _metrics()
    result, diff = ShadowControlPlane(registry=ShadowCapabilityRegistry()).evaluate(
        request_id="req",
        production_run_id="prod",
        production_revision="v1",
        user_message="ordinary request",
        extractor_contract=_extractor(required, forbidden),
        preliminary_strategy=_preliminary("web_access"),
        runtime_snapshot=_snapshot("knowledge_retrieval", "web_access"),
        metrics=metrics,
    )
    assert diff.required_dropped == ()
    assert diff.forbidden_planned == ()
    assert diff.side_effect_count == diff.memory_write_count == 0
    assert result.side_effects_permitted is False
    assert not hasattr(ShadowCapabilityRegistry(), "execute")
    assert metrics.acceptance_passed()


def test_shadow_protocol_failure_does_not_mutate_production_or_execute() -> None:
    degraded = _extractor(status=InvocationStatus.PROTOCOL_FAILED)
    metrics = _metrics()
    production = {"status": "completed", "memory": []}
    before = repr(production)
    _, diff = ShadowControlPlane().evaluate(
        request_id="req", production_run_id="prod", production_revision="v1",
        user_message="ordinary request", extractor_contract=degraded,
        preliminary_strategy=None, runtime_snapshot=_snapshot(), metrics=metrics,
    )
    assert repr(production) == before
    assert diff.side_effect_count == 0
    assert metrics.count("extractor_protocol_degraded_rate") == 1


@pytest.mark.asyncio
async def test_independent_extractor_fails_closed_on_invalid_schema() -> None:
    async def bad_gateway(_request_id: str, _message: str) -> dict[str, object]:
        return {"constraints": "not-a-list"}
    result = await IndependentSemanticRequirementExtractor(bad_gateway).extract(
        request_id="req", user_message="message"
    )
    assert result.invocation_status == InvocationStatus.PROTOCOL_FAILED
    assert result.constraints == ()


def test_fault_injector_is_explicit_and_deterministic() -> None:
    injector = FaultInjector((FaultSpec("router", FaultKind.TIMEOUT),))
    with pytest.raises(InjectedFault, match="router:timeout"):
        injector.invoke("router", lambda: "never")
    assert injector.invoke("rag", lambda: "ok") == "ok"


def test_red_line_slo_gate_and_acceptance_report() -> None:
    metrics = _metrics()
    metrics.observe_request()
    assert set(RED_LINE_METRICS).issubset(metrics.snapshot()["metrics"])
    passed = build_acceptance_report(
        runtime_revision="v2", schema_versions={"control_plane": "v1"},
        cases=(AcceptanceCaseResult("CP-001", "hash", True, {}, {}),), metrics=metrics,
    )
    assert passed["gate"] == "PASS_SHADOW"
    metrics.increment("guard_false_validated_count")
    failed = build_acceptance_report(
        runtime_revision="v2", schema_versions={"control_plane": "v1"},
        cases=(AcceptanceCaseResult("GD-007", "hash", False, {}, {}),), metrics=metrics,
    )
    assert failed["gate"] == "FAIL_CONTRACT_INVARIANT"


def test_shadow_feature_flags_are_safe_by_default() -> None:
    settings = Settings(_env_file=None)
    assert settings.control_plane_shadow_enabled is False
    assert settings.control_plane_shadow_sample_rate == 0.0
    assert settings.control_plane_fault_injection_enabled is False
