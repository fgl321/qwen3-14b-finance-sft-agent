from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.control_plane.clock import MonotonicDeadline
from app.control_plane.contracts import (
    effective_constraint_map,
    merge_contract_draft,
    repair_draft_from_floor,
    seal_contract,
)
from app.control_plane.effects import EffectPolicy, effect_closure, effects_allowed, resolve_unknown_effects
from app.control_plane.enums import (
    Authority,
    EnforcementStrength,
    InvocationStatus,
    MutationEffect,
    NetworkEffect,
    PermissionLevel,
    RequirementLevel,
    RuntimeCapabilityStatus,
    ScopeResolutionStatus,
    StepExecutionState,
    StrategyStatus,
)
from app.control_plane.floor import ExplicitConstraintParser
from app.control_plane.reason_codes import ReasonCode
from app.control_plane.reconciler import reconcile_strategy
from app.control_plane.runtime_decisions import ReplayAction, decide_replay, runtime_gate
from app.control_plane.scope import executor_scope_preflight, resolve_resource_scope
from app.control_plane.schemas import (
    CapabilityAvailability,
    CapabilityConstraint,
    CancellationState,
    ConstraintSource,
    ExplicitRequirementFloor,
    PreliminaryStrategy,
    RequestedResourceScope,
    ResolvedResourceRef,
    RuntimeCapabilitySnapshot,
    SemanticRequirementContract,
    StepIdempotencyRecord,
    TaskRequirement,
    ToolEffects,
    ToolManifest,
)


NOW = datetime(2026, 8, 14, tzinfo=UTC)


def _constraint(
    capability: str,
    *,
    identifier: str,
    requirement: RequirementLevel,
    permission: PermissionLevel = PermissionLevel.ALLOWED,
    authority: Authority = Authority.USER_EXPLICIT,
    strength: EnforcementStrength = EnforcementStrength.EXPLICIT_CONSTRAINT,
    scope_ref: str | None = None,
) -> CapabilityConstraint:
    return CapabilityConstraint(
        capability=capability,
        requirement=requirement,
        permission=permission,
        source=ConstraintSource(
            constraint_id=identifier,
            authority=authority,
            enforcement_strength=strength,
            rule_id=identifier,
        ),
        scope_ref=scope_ref,
    )


def _floor(*constraints: CapabilityConstraint) -> ExplicitRequirementFloor:
    floor = ExplicitRequirementFloor(
        request_id="req-1",
        constraints=constraints,
        parser_version="floor-v1",
    )
    return floor.model_copy(update={"canonical_hash": floor.calculate_hash()})


def _extractor(*constraints: CapabilityConstraint) -> SemanticRequirementContract:
    contract = SemanticRequirementContract(
        request_id="req-1",
        constraints=constraints,
        task_requirements=tuple(
            TaskRequirement(
                task_id=f"task_{index}",
                description=f"complete {item.capability}",
                capabilities=(item.capability,),
            )
            for index, item in enumerate(constraints, start=1)
            if item.requirement == RequirementLevel.REQUIRED
        ),
        invocation_status=InvocationStatus.SUCCESS,
    )
    return contract.model_copy(update={"canonical_hash": contract.calculate_hash()})


def _snapshot(*items: CapabilityAvailability) -> RuntimeCapabilitySnapshot:
    snapshot = RuntimeCapabilitySnapshot(
        run_id="run-1",
        observed_at_utc=NOW,
        capabilities=items,
    )
    return snapshot.model_copy(update={"canonical_hash": snapshot.calculate_hash()})


def test_explicit_floor_is_security_boundary_not_intent_router() -> None:
    parser = ExplicitConstraintParser()
    explicit = parser.parse(
        request_id="req-1",
        user_message="必须检索我上传的文档并且必须给出引用，不要联网。",
    )
    # NL semantics belong to the Semantic Router; the Floor is now a
    # security/protocol boundary and must not fabricate constraints.
    assert explicit.constraints == ()
    implicit = parser.parse(request_id="req-2", user_message="帮我算一下房贷哪个方案更划算")
    assert implicit.constraints == ()


def test_floor_does_not_turn_scope_restriction_into_rag_forbidden() -> None:
    floor = ExplicitConstraintParser().parse(
        request_id="req-clauses",
        user_message="必须引用。不要联网；不要改查其他知识库。",
    )
    assert floor.constraints == ()
    assert floor.resource_constraints.document_exclusive is False


def test_monotonic_merge_preserves_floor_and_detects_same_authority_conflict() -> None:
    required = _constraint("knowledge_retrieval", identifier="required", requirement=RequirementLevel.REQUIRED)
    optional = _constraint(
        "knowledge_retrieval",
        identifier="optional",
        requirement=RequirementLevel.OPTIONAL,
        authority=Authority.SEMANTIC_EXTRACTOR,
        strength=EnforcementStrength.INFERRED,
    )
    draft = merge_contract_draft(floor=_floor(required), extractor=_extractor(optional))
    assert effective_constraint_map(draft.constraints)["knowledge_retrieval"].requirement == RequirementLevel.REQUIRED

    forbidden = _constraint(
        "knowledge_retrieval",
        identifier="forbidden",
        requirement=RequirementLevel.NOT_NEEDED,
        permission=PermissionLevel.FORBIDDEN,
    )
    conflicted = merge_contract_draft(floor=_floor(required, forbidden), extractor=None)
    assert conflicted.conflicts[0].user_resolvable is True
    with pytest.raises(ValueError):
        seal_contract(conflicted, now=NOW)


def test_single_source_required_and_forbidden_is_a_recorded_conflict() -> None:
    contradictory = _constraint(
        "web_access",
        identifier="single-source-conflict",
        requirement=RequirementLevel.REQUIRED,
        permission=PermissionLevel.FORBIDDEN,
    )
    draft = merge_contract_draft(floor=_floor(contradictory), extractor=None)
    assert draft.conflicts[0].constraint_ids == ("single-source-conflict",)
    assert draft.conflicts[0].reason_code == ReasonCode.CONTRACT_PERMISSION_CONFLICT


def test_hard_policy_forbidden_blocks_lower_required_and_integrity_repairs_missing_floor() -> None:
    policy = _constraint(
        "web_access",
        identifier="policy",
        requirement=RequirementLevel.NOT_NEEDED,
        permission=PermissionLevel.FORBIDDEN,
        authority=Authority.SYSTEM_POLICY,
        strength=EnforcementStrength.HARD_POLICY,
    )
    user_required = _constraint(
        "web_access",
        identifier="user",
        requirement=RequirementLevel.REQUIRED,
    )
    conflicted = merge_contract_draft(floor=_floor(policy, user_required), extractor=None)
    assert any(item.reason_code == ReasonCode.CONTRACT_BLOCKED_BY_POLICY for item in conflicted.conflicts)

    valid_floor = _floor(user_required)
    empty_draft = merge_contract_draft(floor=_floor(), extractor=None)
    repaired, revision = repair_draft_from_floor(floor=valid_floor, draft=empty_draft, now=NOW)
    assert revision is not None
    assert repaired.constraints == (user_required,)
    assert revision.reason_codes == (ReasonCode.CONTRACT_INTEGRITY_REPAIRED,)


def test_scope_resolution_and_executor_preflight_never_expand_or_replace() -> None:
    requested = RequestedResourceScope(
        scope_id="doc-scope",
        requested_description="刚上传的文档",
        web_access=PermissionLevel.FORBIDDEN,
    )
    requested = requested.model_copy(update={"canonical_hash": requested.calculate_hash()})
    doc = ResolvedResourceRef(
        tenant_id="personal",
        knowledge_base_id="kb",
        document_id="doc-1",
        document_version=1,
        content_hash="sha256:v1",
    )
    resolved = resolve_resource_scope(
        requested=requested,
        authorized_candidates=(doc,),
        authorization_snapshot_id="auth-1",
        now=NOW,
    )
    assert resolved.resolution_status == ScopeResolutionStatus.RESOLVED
    assert executor_scope_preflight(
        resolved=resolved,
        current_resources=(doc,),
        authorization_snapshot_valid=True,
        tool_target_document_ids=("doc-1",),
    ) == (True, None)
    changed = doc.model_copy(update={"document_version": 2, "content_hash": "sha256:v2"})
    assert executor_scope_preflight(
        resolved=resolved,
        current_resources=(changed,),
        authorization_snapshot_valid=True,
    )[0] is False
    ambiguous = resolve_resource_scope(
        requested=requested,
        authorized_candidates=(doc, changed),
        authorization_snapshot_id="auth-2",
        now=NOW,
    )
    assert ambiguous.resolution_status == ScopeResolutionStatus.AMBIGUOUS
    assert ambiguous.resources == ()


def test_effect_lattice_uses_transitive_closure_and_declared_max_for_unknown() -> None:
    local = ToolEffects(network=NetworkEffect.INTERNAL)
    external = ToolEffects(network=NetworkEffect.EXTERNAL, mutation=MutationEffect.REVERSIBLE)
    closure = effect_closure(local, external)
    assert closure.network == NetworkEffect.EXTERNAL
    policy = EffectPolicy(maximum_allowed=ToolEffects(network=NetworkEffect.INTERNAL))
    assert effects_allowed(closure, policy) is False
    resolved = resolve_unknown_effects(
        ToolEffects(network=NetworkEffect.UNKNOWN),
        ToolEffects(network=NetworkEffect.EXTERNAL),
    )
    assert resolved.network == NetworkEffect.EXTERNAL


def test_reconciler_adds_required_step_and_does_not_claim_unavailable_complete() -> None:
    required = _constraint(
        "financial_calculation",
        identifier="calc",
        requirement=RequirementLevel.REQUIRED,
    )
    draft = merge_contract_draft(floor=_floor(required), extractor=_extractor(required))
    contract = seal_contract(draft, now=NOW)
    manifest = ToolManifest(
        tool_name="loan_amortization_compare",
        tool_version="1",
        capability="financial_calculation",
        declared_max_effects=ToolEffects(),
        deterministic=True,
        result_freshness_policy="immutable",
    )
    available = _snapshot(
        CapabilityAvailability(
            capability="financial_calculation",
            provider_or_tool=manifest.tool_name,
            status=RuntimeCapabilityStatus.AVAILABLE,
            checked_at_utc=NOW,
        )
    )
    strategy = reconcile_strategy(
        run_id="run-1",
        contract=contract,
        preliminary=None,
        scopes=(),
        runtime_snapshot=available,
        tool_manifests=(manifest,),
    )
    assert strategy.strategy_status == StrategyStatus.READY_DEGRADED
    assert strategy.steps[0].tool_name == manifest.tool_name
    assert ReasonCode.STRATEGY_RECONCILED in strategy.reason_codes

    unavailable = _snapshot(
        CapabilityAvailability(
            capability="financial_calculation",
            provider_or_tool=manifest.tool_name,
            status=RuntimeCapabilityStatus.UNAVAILABLE,
            checked_at_utc=NOW,
        )
    )
    blocked = reconcile_strategy(
        run_id="run-1",
        contract=contract,
        preliminary=None,
        scopes=(),
        runtime_snapshot=unavailable,
        tool_manifests=(manifest,),
    )
    assert blocked.strategy_status == StrategyStatus.UNFULFILLABLE
    assert blocked.steps == ()


def test_replay_and_cancellation_never_repeat_unknown_side_effect() -> None:
    manifest = ToolManifest(
        tool_name="external_write",
        tool_version="1",
        capability="external_action",
        declared_max_effects=ToolEffects(mutation=MutationEffect.IRREVERSIBLE),
        supports_idempotency=False,
        supports_status_query=False,
    )
    record = StepIdempotencyRecord(
        idempotency_key="key",
        request_id="req",
        run_id="run",
        sealed_contract_hash="contract",
        resolved_scope_hash="scope",
        step_id="step",
        tool_name=manifest.tool_name,
        tool_version="1",
        normalized_arguments_hash="args",
        effect_profile_hash="effects",
        state=StepExecutionState.UNKNOWN,
    )
    decision = decide_replay(record, identity_matches=True, freshness_valid=False, manifest=manifest)
    assert decision.action == ReplayAction.DO_NOT_RETRY
    assert decision.reason_code == ReasonCode.TOOL_RESULT_UNKNOWN

    deadline = MonotonicDeadline.start(1000, now_utc=NOW)
    cancellation = CancellationState(
        requested=True,
        cancel_requested_at_utc=NOW,
        execution_cutoff_at_utc=NOW,
        reason_code=ReasonCode.RUN_CANCELLED,
    )
    gate = runtime_gate(
        cancellation=cancellation,
        deadline=deadline,
        remaining_time_reserved_for_delivery=False,
    )
    assert gate.may_start_new_work is False
    assert gate.may_start_new_execution_round is False
