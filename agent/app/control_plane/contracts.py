from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

from app.control_plane.canonical import content_hash
from app.control_plane.clock import utc_now
from app.control_plane.enums import Authority, EnforcementStrength, PermissionLevel, RequirementLevel
from app.control_plane.reason_codes import ReasonCode
from app.control_plane.schemas import (
    CapabilityConstraint,
    ConstraintConflict,
    ContractRevisionRecord,
    ExplicitRequirementFloor,
    MergedContractDraft,
    SealedEffectiveContract,
    SemanticRequirementContract,
)


_AUTHORITY_RANK = {
    (Authority.SYSTEM_POLICY, EnforcementStrength.HARD_POLICY): 700,
    (Authority.API_POLICY, EnforcementStrength.HARD_POLICY): 650,
    (Authority.USER_EXPLICIT, EnforcementStrength.EXPLICIT_CONSTRAINT): 600,
    (Authority.CALLER_EXPLICIT, EnforcementStrength.EXPLICIT_CONSTRAINT): 550,
    (Authority.SEMANTIC_EXTRACTOR, EnforcementStrength.INFERRED): 300,
    (Authority.API_POLICY, EnforcementStrength.DEFAULT): 200,
    (Authority.ROUTER_STRATEGY, EnforcementStrength.INFERRED): 100,
}


def constraint_rank(item: CapabilityConstraint) -> int:
    return _AUTHORITY_RANK.get(
        (item.source.authority, item.source.enforcement_strength),
        0,
    )


def _deduplicate_constraints(items: Iterable[CapabilityConstraint]) -> tuple[CapabilityConstraint, ...]:
    seen: set[str] = set()
    result: list[CapabilityConstraint] = []
    for item in items:
        key = item.source.constraint_id
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return tuple(result)


def detect_constraint_conflicts(
    constraints: Iterable[CapabilityConstraint],
) -> tuple[ConstraintConflict, ...]:
    grouped: dict[str, list[CapabilityConstraint]] = defaultdict(list)
    for item in constraints:
        grouped[item.capability].append(item)
    conflicts: list[ConstraintConflict] = []
    for capability, items in grouped.items():
        by_authority: dict[tuple[Authority, EnforcementStrength], list[CapabilityConstraint]] = defaultdict(list)
        for item in items:
            by_authority[(item.source.authority, item.source.enforcement_strength)].append(item)
        for authority_key, same_authority in by_authority.items():
            has_required = any(item.requirement == RequirementLevel.REQUIRED for item in same_authority)
            has_forbidden = any(item.permission == PermissionLevel.FORBIDDEN for item in same_authority)
            if has_required and has_forbidden:
                authority = authority_key[0]
                conflicts.append(
                    ConstraintConflict(
                        conflict_id=f"conflict:{capability}:{authority.value}",
                        capability=capability,
                        constraint_ids=tuple(item.source.constraint_id for item in same_authority),
                        conflict_type="same_authority_required_forbidden",
                        user_resolvable=authority in {Authority.USER_EXPLICIT, Authority.CALLER_EXPLICIT},
                        reason_code=ReasonCode.CONTRACT_PERMISSION_CONFLICT,
                    )
                )
        hard_forbidden = [
            item for item in items
            if item.permission == PermissionLevel.FORBIDDEN
            and item.source.enforcement_strength == EnforcementStrength.HARD_POLICY
        ]
        lower_required = [
            item for item in items
            if item.requirement == RequirementLevel.REQUIRED
            and any(constraint_rank(policy) > constraint_rank(item) for policy in hard_forbidden)
        ]
        if hard_forbidden and lower_required:
            involved = [*hard_forbidden, *lower_required]
            conflicts.append(
                ConstraintConflict(
                    conflict_id=f"conflict:{capability}:hard_policy",
                    capability=capability,
                    constraint_ids=tuple(item.source.constraint_id for item in involved),
                    conflict_type="higher_authority_policy_block",
                    user_resolvable=False,
                    reason_code=ReasonCode.CONTRACT_BLOCKED_BY_POLICY,
                )
            )
    return tuple(conflicts)


def merge_contract_draft(
    *,
    floor: ExplicitRequirementFloor,
    extractor: SemanticRequirementContract | None,
    draft_version: int = 1,
) -> MergedContractDraft:
    if extractor is not None and extractor.request_id != floor.request_id:
        raise ValueError("floor and extractor request_id mismatch")
    constraints = _deduplicate_constraints(
        [*floor.constraints, *((extractor.constraints if extractor else ()))]
    )
    task_by_id = {
        task.task_id: task
        for task in (extractor.task_requirements if extractor else ())
    }
    scope_refs = tuple(
        dict.fromkeys(item.scope_ref for item in constraints if item.scope_ref)
    )
    draft = MergedContractDraft(
        request_id=floor.request_id,
        draft_version=draft_version,
        constraints=constraints,
        task_requirements=tuple(task_by_id.values()),
        conflicts=detect_constraint_conflicts(constraints),
        requested_scope_refs=scope_refs,
        parent_hashes=tuple(
            value for value in (floor.canonical_hash, extractor.canonical_hash if extractor else None) if value
        ),
    )
    return draft.model_copy(update={"canonical_hash": draft.calculate_hash()})


def repair_draft_from_floor(
    *,
    floor: ExplicitRequirementFloor,
    draft: MergedContractDraft,
    now: datetime | None = None,
) -> tuple[MergedContractDraft, ContractRevisionRecord | None]:
    if floor.request_id != draft.request_id:
        raise ValueError("floor and draft request_id mismatch")
    present = {item.source.constraint_id for item in draft.constraints}
    missing = [item for item in floor.constraints if item.source.constraint_id not in present]
    if not missing:
        return draft, None
    repaired = draft.model_copy(
        update={
            "draft_version": draft.draft_version + 1,
            "constraints": _deduplicate_constraints([*draft.constraints, *missing]),
            "conflicts": detect_constraint_conflicts([*draft.constraints, *missing]),
            "canonical_hash": "",
        }
    )
    repaired = repaired.model_copy(update={"canonical_hash": repaired.calculate_hash()})
    record = ContractRevisionRecord(
        revision_id=f"revision:{draft.draft_version}:{repaired.draft_version}",
        request_id=draft.request_id,
        from_draft_hash=draft.canonical_hash,
        to_draft_hash=repaired.canonical_hash,
        changes=tuple(
            {"operation": "restore_floor_constraint", "constraint_id": item.source.constraint_id}
            for item in missing
        ),
        reason_codes=(ReasonCode.CONTRACT_INTEGRITY_REPAIRED,),
        repaired_at_utc=now or utc_now(),
        integrity_gate_version="contract-integrity-v1",
    )
    record = record.model_copy(update={"canonical_hash": record.calculate_hash()})
    return repaired, record


def seal_contract(
    draft: MergedContractDraft,
    *,
    contract_version: int = 1,
    now: datetime | None = None,
) -> SealedEffectiveContract:
    if draft.conflicts:
        raise ValueError("cannot seal a contract draft with unresolved conflicts")
    payload = {
        "schema_version": draft.schema_version,
        "contract_version": contract_version,
        "request_id": draft.request_id,
        "constraints": draft.constraints,
        "task_requirements": draft.task_requirements,
        "requested_scope_refs": draft.requested_scope_refs,
        "conflicts": (),
        "sealed_at_utc": now or utc_now(),
        "parent_draft_hash": draft.canonical_hash,
    }
    return SealedEffectiveContract(**payload, canonical_hash=content_hash(payload))


def effective_constraint_map(
    constraints: Iterable[CapabilityConstraint],
) -> dict[str, CapabilityConstraint]:
    """Return the highest-authority effective constraint per capability.

    Requirement is monotonic among constraints that are not dominated by a
    higher-priority forbidden policy. Permission never becomes allowed when a
    higher-ranked source forbids the capability.
    """

    grouped: dict[str, list[CapabilityConstraint]] = defaultdict(list)
    for item in constraints:
        grouped[item.capability].append(item)
    effective: dict[str, CapabilityConstraint] = {}
    for capability, items in grouped.items():
        ordered = sorted(items, key=constraint_rank, reverse=True)
        top = ordered[0]
        top_forbidden_rank = max(
            (constraint_rank(item) for item in items if item.permission == PermissionLevel.FORBIDDEN),
            default=-1,
        )
        required_candidates = [
            item for item in items
            if item.requirement == RequirementLevel.REQUIRED
            and constraint_rank(item) >= top_forbidden_rank
        ]
        permission = (
            PermissionLevel.FORBIDDEN
            if any(
                item.permission == PermissionLevel.FORBIDDEN
                and constraint_rank(item) == constraint_rank(top)
                for item in items
            )
            else top.permission
        )
        requirement = (
            RequirementLevel.REQUIRED
            if required_candidates
            else RequirementLevel.OPTIONAL
            if any(item.requirement == RequirementLevel.OPTIONAL for item in items)
            else RequirementLevel.NOT_NEEDED
        )
        effective[capability] = top.model_copy(
            update={"requirement": requirement, "permission": permission}
        )
    return effective
