from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ReasonCode(StrEnum):
    CONTRACT_REQUIRED_CAPABILITY_MISSING = "CONTRACT_REQUIRED_CAPABILITY_MISSING"
    CONTRACT_PERMISSION_CONFLICT = "CONTRACT_PERMISSION_CONFLICT"
    CONTRACT_BLOCKED_BY_POLICY = "CONTRACT_BLOCKED_BY_POLICY"
    CONTRACT_INTEGRITY_REPAIRED = "CONTRACT_INTEGRITY_REPAIRED"
    CONTRACT_SEALED_MUTATION_DETECTED = "CONTRACT_SEALED_MUTATION_DETECTED"
    CONTRACT_SCHEMA_MIGRATION_REQUIRED = "CONTRACT_SCHEMA_MIGRATION_REQUIRED"
    CONTRACT_REQUIRED_FIELD_MISSING = "CONTRACT_REQUIRED_FIELD_MISSING"
    SCOPE_RESOLUTION_FAILED = "SCOPE_RESOLUTION_FAILED"
    SCOPE_EXECUTION_PRECONDITION_FAILED = "SCOPE_EXECUTION_PRECONDITION_FAILED"
    DOCUMENT_SCOPE_EMPTY = "DOCUMENT_SCOPE_EMPTY"
    DOCUMENT_SCOPE_AMBIGUOUS = "DOCUMENT_SCOPE_AMBIGUOUS"
    DOCUMENT_SCOPE_NOT_FOUND = "DOCUMENT_SCOPE_NOT_FOUND"
    DOCUMENT_SCOPE_CONFLICT = "DOCUMENT_SCOPE_CONFLICT"
    CITATION_SCOPE_VIOLATION = "CITATION_SCOPE_VIOLATION"
    CONTRACT_EXECUTION_VIOLATION = "CONTRACT_EXECUTION_VIOLATION"
    INSUFFICIENT_SCOPED_EVIDENCE = "INSUFFICIENT_SCOPED_EVIDENCE"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    CAPABILITY_DEGRADED = "CAPABILITY_DEGRADED"
    STRATEGY_RECONCILED = "STRATEGY_RECONCILED"
    STRATEGY_UNCLASSIFIED_CONTROL_STATE = "STRATEGY_UNCLASSIFIED_CONTROL_STATE"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOOL_RESULT_UNKNOWN = "TOOL_RESULT_UNKNOWN"
    RETRIEVAL_NO_EVIDENCE = "RETRIEVAL_NO_EVIDENCE"
    ROUTER_PROTOCOL_DEGRADED = "ROUTER_PROTOCOL_DEGRADED"
    EXTRACTOR_PROTOCOL_DEGRADED = "EXTRACTOR_PROTOCOL_DEGRADED"
    GUARD_PROTOCOL_DEGRADED = "GUARD_PROTOCOL_DEGRADED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    RUN_CANCELLED = "RUN_CANCELLED"
    LATE_RESULT_RECORDED = "LATE_RESULT_RECORDED"


ReasonCategory = Literal[
    "contract",
    "policy",
    "scope",
    "availability",
    "strategy",
    "execution",
    "retrieval",
    "delivery",
    "deadline",
    "budget",
    "audit",
]


class ReasonCodeDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ReasonCode
    category: ReasonCategory
    default_severity: Literal["info", "warning", "error", "critical"]
    retryable: bool
    task_impact: str
    health_impact: str
    delivery_impact: str
    owner_component: str
    introduced_version: str = "control-plane-v2"
    deprecated_by: ReasonCode | None = None


def _d(
    code: ReasonCode,
    category: ReasonCategory,
    severity: Literal["info", "warning", "error", "critical"],
    retryable: bool,
    task: str,
    health: str,
    delivery: str,
    owner: str,
) -> ReasonCodeDefinition:
    return ReasonCodeDefinition(
        code=code,
        category=category,
        default_severity=severity,
        retryable=retryable,
        task_impact=task,
        health_impact=health,
        delivery_impact=delivery,
        owner_component=owner,
    )


_REGISTRY = {
    ReasonCode.CONTRACT_REQUIRED_CAPABILITY_MISSING: _d(ReasonCode.CONTRACT_REQUIRED_CAPABILITY_MISSING, "contract", "error", False, "required outcome unmet", "none", "limitations", "contract_integrity"),
    ReasonCode.CONTRACT_PERMISSION_CONFLICT: _d(ReasonCode.CONTRACT_PERMISSION_CONFLICT, "contract", "error", False, "blocked or partial", "none", "not_generated or limitations", "contract_merge"),
    ReasonCode.CONTRACT_BLOCKED_BY_POLICY: _d(ReasonCode.CONTRACT_BLOCKED_BY_POLICY, "policy", "critical", False, "blocked", "none", "not_generated", "policy_gate"),
    ReasonCode.CONTRACT_INTEGRITY_REPAIRED: _d(ReasonCode.CONTRACT_INTEGRITY_REPAIRED, "contract", "warning", False, "none", "none", "none", "contract_integrity"),
    ReasonCode.CONTRACT_SEALED_MUTATION_DETECTED: _d(ReasonCode.CONTRACT_SEALED_MUTATION_DETECTED, "contract", "critical", False, "blocked or failed", "failed", "not_generated", "contract_integrity"),
    ReasonCode.CONTRACT_SCHEMA_MIGRATION_REQUIRED: _d(ReasonCode.CONTRACT_SCHEMA_MIGRATION_REQUIRED, "contract", "error", False, "new run required", "none", "not_generated", "contract_migration"),
    ReasonCode.CONTRACT_REQUIRED_FIELD_MISSING: _d(ReasonCode.CONTRACT_REQUIRED_FIELD_MISSING, "contract", "critical", False, "blocked", "failed", "not_generated", "contract_validation"),
    ReasonCode.SCOPE_RESOLUTION_FAILED: _d(ReasonCode.SCOPE_RESOLUTION_FAILED, "scope", "error", False, "blocked or partial", "none", "limitations", "scope_resolver"),
    ReasonCode.SCOPE_EXECUTION_PRECONDITION_FAILED: _d(ReasonCode.SCOPE_EXECUTION_PRECONDITION_FAILED, "scope", "error", False, "partial or failed", "none", "limitations", "executor_preflight"),
    ReasonCode.DOCUMENT_SCOPE_EMPTY: _d(ReasonCode.DOCUMENT_SCOPE_EMPTY, "scope", "warning", False, "blocked or partial", "none", "limitations", "scope_resolver"),
    ReasonCode.DOCUMENT_SCOPE_AMBIGUOUS: _d(ReasonCode.DOCUMENT_SCOPE_AMBIGUOUS, "scope", "warning", False, "blocked or partial", "none", "limitations", "scope_resolver"),
    ReasonCode.DOCUMENT_SCOPE_NOT_FOUND: _d(ReasonCode.DOCUMENT_SCOPE_NOT_FOUND, "scope", "warning", False, "blocked or partial", "none", "limitations", "scope_resolver"),
    ReasonCode.DOCUMENT_SCOPE_CONFLICT: _d(ReasonCode.DOCUMENT_SCOPE_CONFLICT, "scope", "warning", False, "blocked", "none", "not_generated", "scope_resolver"),
    ReasonCode.CITATION_SCOPE_VIOLATION: _d(ReasonCode.CITATION_SCOPE_VIOLATION, "delivery", "critical", False, "blocked or failed", "failed", "not_generated", "delivery_guard"),
    ReasonCode.CONTRACT_EXECUTION_VIOLATION: _d(ReasonCode.CONTRACT_EXECUTION_VIOLATION, "execution", "critical", False, "failed", "failed", "not_generated", "contract_integrity"),
    ReasonCode.INSUFFICIENT_SCOPED_EVIDENCE: _d(ReasonCode.INSUFFICIENT_SCOPED_EVIDENCE, "retrieval", "warning", False, "partial or failed", "none", "limitations", "rag"),
    ReasonCode.CAPABILITY_UNAVAILABLE: _d(ReasonCode.CAPABILITY_UNAVAILABLE, "availability", "error", True, "required outcome unmet", "degraded", "limitations", "capability_probe"),
    ReasonCode.CAPABILITY_DEGRADED: _d(ReasonCode.CAPABILITY_DEGRADED, "availability", "warning", True, "policy dependent", "degraded", "policy dependent", "capability_probe"),
    ReasonCode.STRATEGY_RECONCILED: _d(ReasonCode.STRATEGY_RECONCILED, "strategy", "info", False, "none", "none", "none", "strategy_reconciler"),
    ReasonCode.STRATEGY_UNCLASSIFIED_CONTROL_STATE: _d(ReasonCode.STRATEGY_UNCLASSIFIED_CONTROL_STATE, "strategy", "critical", False, "blocked", "failed", "not_generated", "strategy_reconciler"),
    ReasonCode.TOOL_EXECUTION_FAILED: _d(ReasonCode.TOOL_EXECUTION_FAILED, "execution", "error", True, "partial or failed", "policy dependent", "limitations", "tool_executor"),
    ReasonCode.TOOL_RESULT_UNKNOWN: _d(ReasonCode.TOOL_RESULT_UNKNOWN, "execution", "critical", False, "required outcome unmet", "degraded", "not_generated or limitations", "tool_reconciliation"),
    ReasonCode.RETRIEVAL_NO_EVIDENCE: _d(ReasonCode.RETRIEVAL_NO_EVIDENCE, "retrieval", "warning", False, "partial or failed", "none", "limitations", "rag"),
    ReasonCode.ROUTER_PROTOCOL_DEGRADED: _d(ReasonCode.ROUTER_PROTOCOL_DEGRADED, "strategy", "warning", True, "none if reconciled", "degraded", "none", "semantic_router"),
    ReasonCode.EXTRACTOR_PROTOCOL_DEGRADED: _d(ReasonCode.EXTRACTOR_PROTOCOL_DEGRADED, "contract", "warning", True, "floor only", "degraded", "none", "requirement_extractor"),
    ReasonCode.GUARD_PROTOCOL_DEGRADED: _d(ReasonCode.GUARD_PROTOCOL_DEGRADED, "delivery", "error", True, "none", "degraded", "guard_degraded or rejected", "delivery_guard"),
    ReasonCode.DEADLINE_EXCEEDED: _d(ReasonCode.DEADLINE_EXCEEDED, "deadline", "warning", False, "partial or failed", "none", "limitations", "runtime_policy"),
    ReasonCode.BUDGET_EXHAUSTED: _d(ReasonCode.BUDGET_EXHAUSTED, "budget", "warning", False, "partial or failed", "none", "limitations", "runtime_policy"),
    ReasonCode.RUN_CANCELLED: _d(ReasonCode.RUN_CANCELLED, "execution", "warning", False, "fact dependent", "none", "fact dependent", "run_controller"),
    ReasonCode.LATE_RESULT_RECORDED: _d(ReasonCode.LATE_RESULT_RECORDED, "audit", "info", False, "none", "none", "none", "run_audit"),
}

REASON_CODE_REGISTRY = MappingProxyType(_REGISTRY)


def reason_definition(code: ReasonCode | str) -> ReasonCodeDefinition:
    return REASON_CODE_REGISTRY[ReasonCode(code)]


def validate_reason_registry() -> None:
    missing = set(ReasonCode) - set(REASON_CODE_REGISTRY)
    extra = set(REASON_CODE_REGISTRY) - set(ReasonCode)
    if missing or extra:
        raise RuntimeError(f"reason registry mismatch: missing={missing}, extra={extra}")
