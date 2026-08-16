from __future__ import annotations

from datetime import datetime
from typing import Iterable

from app.control_plane.clock import utc_now
from app.control_plane.enums import ScopeResolutionStatus
from app.control_plane.reason_codes import ReasonCode
from app.control_plane.schemas import (
    RequestedResourceScope,
    ResolvedResourceRef,
    ResolvedResourceScope,
)


def resolve_resource_scope(
    *,
    requested: RequestedResourceScope,
    authorized_candidates: Iterable[ResolvedResourceRef],
    authorization_snapshot_id: str,
    now: datetime | None = None,
) -> ResolvedResourceScope:
    candidates = tuple(authorized_candidates)
    status = (
        ScopeResolutionStatus.NOT_FOUND
        if not candidates
        else ScopeResolutionStatus.AMBIGUOUS
        if len(candidates) > 1
        else ScopeResolutionStatus.RESOLVED
    )
    resources = candidates if status == ScopeResolutionStatus.RESOLVED else ()
    scope = ResolvedResourceScope(
        scope_id=requested.scope_id,
        requested_scope_hash=requested.canonical_hash or requested.calculate_hash(),
        resources=resources,
        allowed_source_types=requested.allowed_source_types,
        forbidden_source_types=requested.forbidden_source_types,
        web_access=requested.web_access,
        authorization_snapshot_id=authorization_snapshot_id,
        resolved_at_utc=now or utc_now(),
        resolution_status=status,
    )
    return scope.model_copy(update={"canonical_hash": scope.calculate_hash()})


def executor_scope_preflight(
    *,
    resolved: ResolvedResourceScope,
    current_resources: Iterable[ResolvedResourceRef],
    authorization_snapshot_valid: bool,
    tool_target_document_ids: Iterable[str] = (),
) -> tuple[bool, ReasonCode | None]:
    if resolved.resolution_status != ScopeResolutionStatus.RESOLVED:
        return False, ReasonCode.SCOPE_EXECUTION_PRECONDITION_FAILED
    if not authorization_snapshot_valid:
        return False, ReasonCode.SCOPE_EXECUTION_PRECONDITION_FAILED
    expected = {
        (item.tenant_id, item.knowledge_base_id, item.document_id, item.document_version, item.content_hash)
        for item in resolved.resources
    }
    actual = {
        (item.tenant_id, item.knowledge_base_id, item.document_id, item.document_version, item.content_hash)
        for item in current_resources
    }
    if actual != expected:
        return False, ReasonCode.SCOPE_EXECUTION_PRECONDITION_FAILED
    allowed_ids = {item.document_id for item in resolved.resources}
    if not set(tool_target_document_ids).issubset(allowed_ids):
        return False, ReasonCode.SCOPE_EXECUTION_PRECONDITION_FAILED
    return True, None
