from __future__ import annotations

from enum import StrEnum


class RequirementLevel(StrEnum):
    NOT_NEEDED = "not_needed"
    OPTIONAL = "optional"
    REQUIRED = "required"


class PermissionLevel(StrEnum):
    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"


class Authority(StrEnum):
    SYSTEM_POLICY = "system_policy"
    API_POLICY = "api_policy"
    USER_EXPLICIT = "user_explicit"
    CALLER_EXPLICIT = "caller_explicit"
    SEMANTIC_EXTRACTOR = "semantic_extractor"
    ROUTER_STRATEGY = "router_strategy"


class EnforcementStrength(StrEnum):
    HARD_POLICY = "hard_policy"
    EXPLICIT_CONSTRAINT = "explicit_constraint"
    DEFAULT = "default"
    INFERRED = "inferred"


class InvocationStatus(StrEnum):
    SUCCESS = "success"
    REPAIRED = "repaired"
    PROTOCOL_FAILED = "protocol_failed"
    SERVICE_FAILED = "service_failed"
    TIMEOUT = "timeout"


class RuntimeCapabilityStatus(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ScopeResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


class NetworkEffect(StrEnum):
    NONE = "none"
    INTERNAL = "internal"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class DataEffect(StrEnum):
    NONE = "none"
    LOCAL_READ = "local_read"
    EXTERNAL_READ = "external_read"
    EXTERNAL_WRITE = "external_write"
    UNKNOWN = "unknown"


class MutationEffect(StrEnum):
    NONE = "none"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class SensitiveDataEffect(StrEnum):
    NONE = "none"
    PII_READ = "pii_read"
    PII_EGRESS = "pii_egress"
    UNKNOWN = "unknown"


class CostEffect(StrEnum):
    FREE = "free"
    METERED = "metered"
    UNKNOWN = "unknown"


class StrategyStatus(StrEnum):
    READY = "ready"
    READY_DEGRADED = "ready_degraded"
    BLOCKED = "blocked"
    UNFULFILLABLE = "unfulfillable"


class StepExecutionState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class CapabilityOutcomeStatus(StrEnum):
    SATISFIED = "satisfied"
    PARTIALLY_SATISFIED = "partially_satisfied"
    UNSATISFIED = "unsatisfied"
    NOT_REQUIRED = "not_required"


class TaskStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"


class SystemHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class DeliveryStatus(StrEnum):
    VALIDATED = "validated"
    VALIDATED_WITH_LIMITATIONS = "validated_with_limitations"
    GUARD_DEGRADED = "guard_degraded"
    REJECTED = "rejected"
    NOT_GENERATED = "not_generated"


class RequestRiskClass(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GuardState(StrEnum):
    PASSED = "passed"
    PASSED_WITH_LIMITATIONS = "passed_with_limitations"
    PROTOCOL_DEGRADED = "protocol_degraded"
    REJECTED = "rejected"
    NOT_RUN = "not_run"
