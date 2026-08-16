from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Iterable

from app.control_plane.contracts import merge_contract_draft, repair_draft_from_floor, seal_contract
from app.control_plane.enums import (
    Authority, EnforcementStrength, InvocationStatus, PermissionLevel,
    RequirementLevel, RuntimeCapabilityStatus,
)
from app.control_plane.floor import ExplicitConstraintParser
from app.control_plane.reconciler import reconcile_strategy
from app.control_plane.schemas import (
    CapabilityAvailability, CapabilityConstraint, ConstraintSource,
    PreliminaryStrategy, RuntimeCapabilitySnapshot, SemanticRequirementContract,
    ResolvedResourceScope, StrategyStatus, TaskRequirement,
)


class ControlPlaneBlocked(ValueError):
    def __init__(self, *, reason_codes: tuple[str, ...], audit: dict[str, Any]) -> None:
        super().__init__(",".join(reason_codes) or "CONTROL_PLANE_BLOCKED")
        self.reason_codes = reason_codes
        self.audit = audit


def semantic_contract_from_route(*, request_id: str, route: Any) -> SemanticRequirementContract:
    # Route may infer required work, but explicit deny permissions remain
    # exclusively owned by Floor/policy.
    raw_constraints: list[CapabilityConstraint] = []
    typed = dict(getattr(route, "capability_constraints", None) or {})
    resource = getattr(route, "resource_constraints", None)
    retrieval_scoped = bool(
        getattr(resource, "include_documents", None)
        or getattr(resource, "exclusive", False)
    )
    for capability, value in typed.items():
        if value == "forbidden":
            requirement = RequirementLevel.NOT_NEEDED
            permission = PermissionLevel.FORBIDDEN
        elif value == "required":
            requirement = RequirementLevel.REQUIRED
            permission = PermissionLevel.ALLOWED
        elif value == "optional":
            requirement = RequirementLevel.OPTIONAL
            permission = PermissionLevel.ALLOWED
        else:
            requirement = RequirementLevel.NOT_NEEDED
            permission = PermissionLevel.ALLOWED
        raw_constraints.append(
            CapabilityConstraint(
                capability=capability,
                requirement=requirement,
                permission=permission,
                scope_ref=(
                    "uploaded_documents"
                    if capability == "knowledge_retrieval"
                    and retrieval_scoped
                    else None
                ),
                source=ConstraintSource(
                    constraint_id=f"route:typed:{capability}",
                    authority=Authority.ROUTER_STRATEGY,
                    enforcement_strength=EnforcementStrength.INFERRED,
                    rule_id="semantic-route-typed-contract-v1",
                ),
            )
        )
    covered = {item.capability for item in raw_constraints}
    for index, capability in enumerate(
        getattr(route, "required_capabilities", None) or (),
        start=1,
    ):
        if capability in covered:
            continue
        raw_constraints.append(
            CapabilityConstraint(
                capability=capability,
                requirement=RequirementLevel.REQUIRED,
                permission=PermissionLevel.ALLOWED,
                scope_ref=(
                    "uploaded_documents"
                    if capability == "knowledge_retrieval"
                    and retrieval_scoped
                    else None
                ),
                source=ConstraintSource(
                    constraint_id=f"route:{index}:{capability}",
                    authority=Authority.ROUTER_STRATEGY,
                    enforcement_strength=EnforcementStrength.INFERRED,
                    rule_id="legacy-semantic-route-adapter-v1",
                ),
            )
        )
    constraints = tuple(raw_constraints)
    tasks = tuple(
        TaskRequirement(
            task_id=item.id, description=item.description, required=item.required,
            capabilities=tuple(item.capabilities), depends_on=tuple(item.depends_on),
            evidence_tool_names=tuple(item.evidence_tool_names), requires_citations=item.requires_citations,
        ) for item in route.task_requirements
    )
    value = SemanticRequirementContract(
        request_id=request_id, constraints=constraints, task_requirements=tasks,
        confidence=Decimal(str(route.confidence)), invocation_status=InvocationStatus.SUCCESS,
    )
    return value.model_copy(update={"canonical_hash": value.calculate_hash()})


def preliminary_from_route(*, request_id: str, route: Any) -> PreliminaryStrategy:
    contract = semantic_contract_from_route(request_id=request_id, route=route)
    value = PreliminaryStrategy(
        request_id=request_id, orchestration_mode=route.orchestration_mode,
        proposed_capabilities=tuple(route.required_capabilities),
        proposed_tasks=contract.task_requirements, confidence=Decimal(str(route.confidence)),
        invocation_status=InvocationStatus.SUCCESS,
    )
    return value.model_copy(update={"canonical_hash": value.calculate_hash()})


def production_control_preflight(
    *,
    request_id: str,
    run_id: str,
    user_message: str,
    route: Any,
    constraints: Any | None = None,
    scopes: Iterable[ResolvedResourceScope] = (),
) -> dict[str, Any]:
    """Reconcile the explicit floor with the resolved resource scopes.

    The caller parses the floor exactly once at the request boundary and passes
    it here; ``constraints=None`` is kept only for legacy callers/tests.
    """
    floor = (
        constraints
        if constraints is not None
        else ExplicitConstraintParser().parse(
            request_id=request_id,
            user_message=user_message,
        )
    )
    extractor = semantic_contract_from_route(request_id=request_id, route=route)
    draft = merge_contract_draft(floor=floor, extractor=extractor)
    draft, revision = repair_draft_from_floor(floor=floor, draft=draft)
    if draft.conflicts:
        codes = tuple(dict.fromkeys(item.reason_code.value for item in draft.conflicts))
        raise ControlPlaneBlocked(reason_codes=codes, audit={
            "floor_hash": floor.canonical_hash, "draft_hash": draft.canonical_hash,
            "conflicts": [item.model_dump(mode="json") for item in draft.conflicts],
        })
    sealed = seal_contract(draft)
    now = datetime.now(UTC)
    # The runtime snapshot must cover every capability in the effective
    # contract, not just the legacy required_capabilities list.  Typed
    # capability constraints (e.g. citation_validation=required) would
    # otherwise be treated as UNKNOWN and wrongly block the request.
    capabilities = tuple(
        dict.fromkeys(
            item.capability
            for item in sealed.constraints
            if item.requirement
            in {
                RequirementLevel.REQUIRED,
                RequirementLevel.OPTIONAL,
            }
        )
    )
    snapshot = RuntimeCapabilitySnapshot(
        run_id=run_id, observed_at_utc=now,
        capabilities=tuple(CapabilityAvailability(
            capability=capability, provider_or_tool="legacy-data-plane",
            status=RuntimeCapabilityStatus.AVAILABLE, checked_at_utc=now,
        ) for capability in capabilities),
    )
    snapshot = snapshot.model_copy(update={"canonical_hash": snapshot.calculate_hash()})
    strategy = reconcile_strategy(
        run_id=run_id, contract=sealed,
        preliminary=preliminary_from_route(request_id=request_id, route=route),
        scopes=tuple(scopes), runtime_snapshot=snapshot,
    )
    if strategy.strategy_status in {StrategyStatus.BLOCKED, StrategyStatus.UNFULFILLABLE}:
        raise ControlPlaneBlocked(
            reason_codes=tuple(code.value for code in strategy.reason_codes),
            audit={"floor_hash": floor.canonical_hash, "contract_hash": sealed.canonical_hash,
                "strategy_hash": strategy.canonical_hash},
        )
    return {
        "mode": "v2_execution",
        "floor_hash": floor.canonical_hash,
        "extractor_hash": extractor.canonical_hash,
        "sealed_contract_hash": sealed.canonical_hash,
        "strategy_hash": strategy.canonical_hash,
        "strategy_status": strategy.strategy_status.value,
        "reason_codes": [code.value for code in strategy.reason_codes],
    }
