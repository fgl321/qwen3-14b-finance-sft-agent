from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from app.control_plane.contracts import effective_constraint_map
from app.control_plane.effects import EffectPolicy, effects_allowed, resolve_unknown_effects
from app.control_plane.enums import (
    PermissionLevel,
    RequirementLevel,
    RuntimeCapabilityStatus,
    ScopeResolutionStatus,
    StrategyStatus,
)
from app.control_plane.reason_codes import ReasonCode
from app.control_plane.schemas import (
    CapabilityAvailability,
    EffectiveExecutionStrategy,
    PreliminaryStrategy,
    ResolvedResourceScope,
    RuntimeCapabilitySnapshot,
    SealedEffectiveContract,
    StrategyStep,
    ToolManifest,
)


def _availability_by_capability(
    snapshot: RuntimeCapabilitySnapshot,
) -> dict[str, RuntimeCapabilityStatus]:
    grouped: dict[str, list[RuntimeCapabilityStatus]] = defaultdict(list)
    for item in snapshot.capabilities:
        grouped[item.capability].append(item.status)
    priority = {
        RuntimeCapabilityStatus.AVAILABLE: 3,
        RuntimeCapabilityStatus.DEGRADED: 2,
        RuntimeCapabilityStatus.UNKNOWN: 1,
        RuntimeCapabilityStatus.UNAVAILABLE: 0,
    }
    return {
        capability: max(statuses, key=lambda status: priority[status])
        for capability, statuses in grouped.items()
    }


def _validate_dag(steps: Iterable[StrategyStep]) -> None:
    step_map = {step.step_id: step for step in steps}
    if len(step_map) != len(tuple(steps)):
        raise ValueError("duplicate strategy step_id")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("strategy DAG contains a cycle")
        if step_id in visited:
            return
        if step_id not in step_map:
            raise ValueError(f"unknown strategy dependency: {step_id}")
        visiting.add(step_id)
        for dependency in step_map[step_id].depends_on:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for identifier in step_map:
        visit(identifier)


def reconcile_strategy(
    *,
    run_id: str,
    contract: SealedEffectiveContract,
    preliminary: PreliminaryStrategy | None,
    scopes: Iterable[ResolvedResourceScope],
    runtime_snapshot: RuntimeCapabilitySnapshot,
    tool_manifests: Iterable[ToolManifest] = (),
    effect_policy: EffectPolicy | None = None,
) -> EffectiveExecutionStrategy:
    effective = effective_constraint_map(contract.constraints)
    scope_items = tuple(scopes)
    scope_by_id = {scope.scope_id: scope for scope in scope_items}
    availability = _availability_by_capability(runtime_snapshot)
    manifests_by_capability: dict[str, list[ToolManifest]] = defaultdict(list)
    for manifest in tool_manifests:
        manifests_by_capability[manifest.capability].append(manifest)

    proposed_tasks = preliminary.proposed_tasks if preliminary is not None else ()
    steps: list[StrategyStep] = []
    proposed_step_ids_by_task: dict[str, list[str]] = defaultdict(list)
    reasons: list[ReasonCode] = []
    blocked = False
    unfulfillable = False

    for task in proposed_tasks:
        for capability in task.capabilities:
            constraint = effective.get(capability)
            if constraint and constraint.permission == PermissionLevel.FORBIDDEN:
                reasons.append(ReasonCode.CONTRACT_PERMISSION_CONFLICT)
                continue
            manifest = next(iter(manifests_by_capability.get(capability, ())), None)
            expected_effects = manifest.declared_max_effects if manifest else None
            if expected_effects and effect_policy and not effects_allowed(expected_effects, effect_policy):
                reasons.append(ReasonCode.CONTRACT_PERMISSION_CONFLICT)
                continue
            step_id = f"proposed:{task.task_id}:{capability}"
            steps.append(
                StrategyStep(
                    step_id=step_id,
                    capability=capability,
                    task_ids=(task.task_id,),
                    depends_on=(),
                    tool_name=manifest.tool_name if manifest else (task.evidence_tool_names[0] if task.evidence_tool_names else None),
                    scope_hash=(scope_items[0].canonical_hash if scope_items and capability == "knowledge_retrieval" else None),
                    expected_effects=expected_effects,
                    required_outputs=(),
                )
            )
            proposed_step_ids_by_task[task.task_id].append(step_id)

    task_dependencies = {task.task_id: task.depends_on for task in proposed_tasks}
    steps = [
        step.model_copy(
            update={
                "depends_on": tuple(
                    dependency_step
                    for task_id in step.task_ids
                    for dependency_task in task_dependencies.get(task_id, ())
                    for dependency_step in proposed_step_ids_by_task.get(dependency_task, ())
                )
            }
        )
        for step in steps
    ]

    existing_capabilities = {step.capability for step in steps}
    for capability, constraint in effective.items():
        if constraint.requirement != RequirementLevel.REQUIRED:
            continue
        if constraint.permission == PermissionLevel.FORBIDDEN:
            blocked = True
            reasons.append(ReasonCode.CONTRACT_PERMISSION_CONFLICT)
            continue
        if constraint.scope_ref:
            scope = scope_by_id.get(constraint.scope_ref)
            if scope is None or scope.resolution_status != ScopeResolutionStatus.RESOLVED:
                blocked = True
                reasons.append(ReasonCode.SCOPE_RESOLUTION_FAILED)
                continue
        runtime_status = availability.get(capability, RuntimeCapabilityStatus.UNKNOWN)
        if runtime_status in {RuntimeCapabilityStatus.UNAVAILABLE, RuntimeCapabilityStatus.UNKNOWN}:
            unfulfillable = True
            reasons.append(ReasonCode.CAPABILITY_UNAVAILABLE)
            continue
        if runtime_status == RuntimeCapabilityStatus.DEGRADED:
            reasons.append(ReasonCode.CAPABILITY_DEGRADED)
        if capability in existing_capabilities:
            continue
        manifest = next(iter(manifests_by_capability.get(capability, ())), None)
        effects = manifest.declared_max_effects if manifest else None
        if effects and effect_policy:
            resolved_effects = resolve_unknown_effects(effects, manifest.declared_max_effects)
            if not effects_allowed(resolved_effects, effect_policy):
                blocked = True
                reasons.append(ReasonCode.CONTRACT_PERMISSION_CONFLICT)
                continue
        task_ids = tuple(
            task.task_id for task in contract.task_requirements if capability in task.capabilities
        )
        scope = scope_by_id.get(constraint.scope_ref) if constraint.scope_ref else None
        steps.append(
            StrategyStep(
                step_id=f"required:{capability}",
                capability=capability,
                task_ids=task_ids,
                tool_name=manifest.tool_name if manifest else None,
                scope_hash=scope.canonical_hash if scope else None,
                expected_effects=effects,
                required_outputs=(),
            )
        )
        reasons.append(ReasonCode.STRATEGY_RECONCILED)

    _validate_dag(tuple(steps))
    status = (
        StrategyStatus.BLOCKED
        if blocked
        else StrategyStatus.UNFULFILLABLE
        if unfulfillable
        else StrategyStatus.READY_DEGRADED
        if reasons
        else StrategyStatus.READY
    )
    strategy = EffectiveExecutionStrategy(
        run_id=run_id,
        sealed_contract_hash=contract.canonical_hash,
        resolved_scope_hashes=tuple(scope.canonical_hash for scope in scope_items),
        runtime_snapshot_hash=runtime_snapshot.canonical_hash or runtime_snapshot.calculate_hash(),
        preliminary_strategy_hash=(preliminary.canonical_hash if preliminary else None),
        steps=tuple(steps),
        strategy_status=status,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )
    return strategy.model_copy(update={"canonical_hash": strategy.calculate_hash()})
