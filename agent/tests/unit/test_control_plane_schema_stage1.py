from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.control_plane.canonical import canonical_json, content_hash, identity_fingerprint
from app.control_plane.clock import MonotonicDeadline
from app.control_plane.enums import (
    Authority,
    EnforcementStrength,
    InvocationStatus,
    PermissionLevel,
    RequirementLevel,
    ScopeResolutionStatus,
    StepExecutionState,
)
from app.control_plane.reason_codes import ReasonCode, REASON_CODE_REGISTRY, validate_reason_registry
from app.control_plane.schemas import (
    CONTROL_PLANE_SCHEMA_VERSION,
    CapabilityConstraint,
    ConstraintSource,
    ExplicitRequirementFloor,
    RequestedResourceScope,
    ResolvedResourceScope,
    RuntimeBudget,
    RuntimePolicyEnvelope,
    SealedEffectiveContract,
    StepIdempotencyRecord,
    TaskRequirement,
    build_step_idempotency_key,
)


def _source(identifier: str = "c1") -> ConstraintSource:
    return ConstraintSource(
        constraint_id=identifier,
        authority=Authority.USER_EXPLICIT,
        enforcement_strength=EnforcementStrength.EXPLICIT_CONSTRAINT,
        rule_id="explicit_document_retrieval_v1",
        source_start=0,
        source_end=4,
        source_hash="sha256:test",
    )


def test_reason_registry_is_complete_and_immutable() -> None:
    validate_reason_registry()
    assert set(REASON_CODE_REGISTRY) == set(ReasonCode)
    with pytest.raises(TypeError):
        REASON_CODE_REGISTRY[ReasonCode.RUN_CANCELLED] = None  # type: ignore[index]


def test_canonical_json_is_stable_and_preserves_ordered_lists() -> None:
    left = {"b": 2, "a": "e\u0301", "steps": ["one", "two"]}
    right = {"steps": ["one", "two"], "a": "é", "b": 2}
    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)
    assert content_hash({"steps": ["one", "two"]}) != content_hash(
        {"steps": ["two", "one"]}
    )
    assert content_hash(
        {"capabilities": ["rag", "tool"]},
        semantic_set_paths=("capabilities",),
    ) == content_hash(
        {"capabilities": ["tool", "rag"]},
        semantic_set_paths=("capabilities",),
    )


def test_canonical_numeric_and_identity_rules() -> None:
    assert canonical_json({"amount": Decimal("1.000")}) == '{"amount":"1"}'
    with pytest.raises(TypeError):
        canonical_json({"amount": 1.0})
    first = identity_fingerprint(
        "13800000000",
        key=b"secret",
        key_id="k1",
        tenant_id="tenant-a",
        field_type="phone",
        purpose="audit",
    )
    second = identity_fingerprint(
        "13800000000",
        key=b"secret",
        key_id="k1",
        tenant_id="tenant-b",
        field_type="phone",
        purpose="audit",
    )
    assert first["fingerprint"] != second["fingerprint"]
    assert first["key_id"] == "k1"


def test_monotonic_deadline_restores_without_extending_original_budget() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    deadline = MonotonicDeadline.start(1000, now_utc=now)
    restored = MonotonicDeadline.restore(
        deadline_at_utc=deadline.deadline_at_utc,
        original_budget_ms=1000,
        now_utc=now + timedelta(milliseconds=400),
    )
    assert 0 < restored.remaining_ms() <= 600
    expired = MonotonicDeadline.restore(
        deadline_at_utc=deadline.deadline_at_utc,
        original_budget_ms=1000,
        now_utc=now + timedelta(seconds=2),
    )
    assert expired.expired()


def test_contract_models_are_frozen_and_sealed_hash_is_verified() -> None:
    constraint = CapabilityConstraint(
        capability="knowledge_retrieval",
        requirement=RequirementLevel.REQUIRED,
        permission=PermissionLevel.ALLOWED,
        source=_source(),
    )
    floor = ExplicitRequirementFloor(
        request_id="req-1",
        constraints=(constraint,),
        parser_version="floor-v1",
    )
    assert floor.calculate_hash().startswith("sha256:")
    with pytest.raises(ValidationError):
        floor.request_id = "changed"  # type: ignore[misc]

    task = TaskRequirement(
        task_id="retrieve_docs",
        description="Retrieve selected documents",
        capabilities=("knowledge_retrieval",),
        requires_citations=True,
    )
    sealed_at = datetime(2026, 8, 14, tzinfo=UTC)
    payload = {
        "schema_version": CONTROL_PLANE_SCHEMA_VERSION,
        "contract_version": 1,
        "request_id": "req-1",
        "constraints": (constraint,),
        "task_requirements": (task,),
        "requested_scope_refs": (),
        "conflicts": (),
        "sealed_at_utc": sealed_at,
        "parent_draft_hash": "sha256:draft",
    }
    sealed = SealedEffectiveContract(
        **payload,
        canonical_hash=content_hash(payload),
    )
    assert sealed.canonical_hash == sealed.calculate_hash()
    with pytest.raises(ValidationError):
        SealedEffectiveContract(**payload, canonical_hash="sha256:wrong")


def test_scope_and_runtime_policy_validate_safety_fields() -> None:
    requested = RequestedResourceScope(
        scope_id="scope-1",
        requested_description="刚上传的文档",
        web_access=PermissionLevel.FORBIDDEN,
    )
    assert requested.web_access == PermissionLevel.FORBIDDEN
    with pytest.raises(ValidationError):
        ResolvedResourceScope(
            scope_id="scope-1",
            requested_scope_hash=requested.calculate_hash(),
            authorization_snapshot_id="auth-1",
            resolved_at_utc=datetime.now(UTC),
            resolution_status=ScopeResolutionStatus.RESOLVED,
        )

    budget = RuntimeBudget(
        max_llm_calls=8,
        max_tool_calls=6,
        max_retrieval_queries=8,
        max_protocol_repairs=2,
        max_execution_rounds=3,
        token_budget=10000,
        monetary_budget=Decimal("200"),
    )
    with pytest.raises(ValidationError):
        RuntimePolicyEnvelope(
            request_id="req-1",
            hard_limits=budget,
            soft_targets=budget,
            reserved_delivery_budget_ms=1201,
            created_at_utc=datetime.now(UTC),
            deadline_at_utc=datetime.now(UTC) + timedelta(milliseconds=1200),
            original_budget_ms=1200,
            source_authority=Authority.USER_EXPLICIT,
        )


def test_step_idempotency_key_covers_contract_scope_tool_and_effects() -> None:
    base = dict(
        idempotency_key="placeholder",
        request_id="req-1",
        run_id="run-1",
        sealed_contract_hash="sha256:contract",
        resolved_scope_hash="sha256:scope",
        step_id="step-1",
        tool_name="loan_amortization_compare",
        tool_version="1",
        normalized_arguments_hash="sha256:args",
        effect_profile_hash="sha256:effects",
        state=StepExecutionState.SUCCEEDED,
        result_ref="result:1",
        original_completed_at_utc=datetime.now(UTC),
        freshness_validated=True,
    )
    record = StepIdempotencyRecord(**base)
    key = build_step_idempotency_key(record)
    changed = StepIdempotencyRecord(**{**base, "tool_version": "2"})
    assert key != build_step_idempotency_key(changed)
    with pytest.raises(ValidationError):
        StepIdempotencyRecord(**{**base, "state": StepExecutionState.SUCCEEDED, "result_ref": None})
