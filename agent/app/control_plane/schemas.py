from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.control_plane.canonical import content_hash
from app.control_plane.clock import require_utc
from app.control_plane.enums import (
    Authority,
    CapabilityOutcomeStatus,
    CostEffect,
    DataEffect,
    DeliveryStatus,
    EnforcementStrength,
    InvocationStatus,
    MutationEffect,
    NetworkEffect,
    PermissionLevel,
    RequirementLevel,
    RuntimeCapabilityStatus,
    ScopeResolutionStatus,
    SensitiveDataEffect,
    StepExecutionState,
    StrategyStatus,
    SystemHealth,
    TaskStatus,
)
from app.control_plane.reason_codes import ReasonCode


CONTROL_PLANE_SCHEMA_VERSION = "control-plane-schema-v1"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


class CanonicalModel(FrozenModel):
    def calculate_hash(
        self,
        *,
        exclude: set[str] | None = None,
        semantic_set_paths: tuple[str, ...] = (),
    ) -> str:
        excluded = {"canonical_hash", *(exclude or set())}
        return content_hash(
            self.model_dump(mode="python", exclude=excluded),
            semantic_set_paths=semantic_set_paths,
        )


class ConstraintSource(FrozenModel):
    constraint_id: str = Field(min_length=1, max_length=160)
    authority: Authority
    enforcement_strength: EnforcementStrength
    rule_id: str = Field(min_length=1, max_length=160)
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, ge=0)
    source_hash: str | None = None
    redacted_preview: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def validate_span(self) -> "ConstraintSource":
        if (self.source_start is None) != (self.source_end is None):
            raise ValueError("source_start and source_end must be provided together")
        if self.source_start is not None and self.source_end < self.source_start:
            raise ValueError("source_end must not precede source_start")
        return self


class CapabilityConstraint(FrozenModel):
    capability: str = Field(min_length=1, max_length=120)
    requirement: RequirementLevel = RequirementLevel.NOT_NEEDED
    permission: PermissionLevel = PermissionLevel.ALLOWED
    source: ConstraintSource
    scope_ref: str | None = Field(default=None, max_length=160)


class ResourceConstraints(FrozenModel):
    """Resource/source-level constraints orthogonal to capabilities.

    Capability constraints answer "may the system use retrieval at all";
    resource constraints answer "which documents/sources may it touch".
    They never overwrite each other.
    """

    document_exclusive: bool = False
    allow_other_documents: bool = True
    requested_title: str | None = Field(default=None, max_length=120)


class ExplicitRequirementFloor(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    request_id: str = Field(min_length=1, max_length=160)
    constraints: tuple[CapabilityConstraint, ...] = ()
    resource_constraints: ResourceConstraints = Field(
        default_factory=ResourceConstraints
    )
    extraction_status: Literal["completed"] = "completed"
    parser_version: str = Field(min_length=1, max_length=80)
    canonical_hash: str = ""


class TaskRequirement(FrozenModel):
    task_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z][A-Za-z0-9_:-]*$")
    description: str = Field(min_length=1, max_length=500)
    required: bool = True
    capabilities: tuple[str, ...] = Field(min_length=1, max_length=16)
    depends_on: tuple[str, ...] = Field(default=(), max_length=24)
    evidence_tool_names: tuple[str, ...] = Field(default=(), max_length=24)
    requires_citations: bool = False

    @model_validator(mode="after")
    def validate_no_self_dependency(self) -> "TaskRequirement":
        if self.task_id in self.depends_on:
            raise ValueError("task cannot depend on itself")
        return self


class SemanticRequirementContract(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    request_id: str
    constraints: tuple[CapabilityConstraint, ...] = ()
    task_requirements: tuple[TaskRequirement, ...] = ()
    confidence: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    ambiguities: tuple[str, ...] = ()
    invocation_status: InvocationStatus
    canonical_hash: str = ""


class PreliminaryStrategy(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    request_id: str
    orchestration_mode: Literal[
        "direct", "rag", "tool", "hybrid", "clarify", "unsupported"
    ]
    proposed_capabilities: tuple[str, ...] = ()
    proposed_tasks: tuple[TaskRequirement, ...] = ()
    confidence: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    invocation_status: InvocationStatus
    canonical_hash: str = ""


class ConstraintConflict(FrozenModel):
    conflict_id: str
    capability: str
    # A single explicit source may itself state required+forbidden (for
    # example, "must use the internet, but do not access it"). That is a real
    # conflict even though it has one provenance id.
    constraint_ids: tuple[str, ...] = Field(min_length=1)
    conflict_type: Literal[
        "same_authority_required_forbidden",
        "higher_authority_policy_block",
        "scope_conflict",
        "unclassified_constraint_conflict",
    ]
    user_resolvable: bool
    reason_code: ReasonCode


class MergedContractDraft(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    request_id: str
    draft_version: int = Field(default=1, ge=1)
    constraints: tuple[CapabilityConstraint, ...] = ()
    task_requirements: tuple[TaskRequirement, ...] = ()
    conflicts: tuple[ConstraintConflict, ...] = ()
    requested_scope_refs: tuple[str, ...] = ()
    parent_hashes: tuple[str, ...] = ()
    canonical_hash: str = ""


class ContractRevisionRecord(CanonicalModel):
    revision_id: str
    request_id: str
    from_draft_hash: str
    to_draft_hash: str
    changes: tuple[dict[str, Any], ...] = ()
    reason_codes: tuple[ReasonCode, ...] = ()
    repaired_at_utc: datetime
    integrity_gate_version: str
    canonical_hash: str = ""

    @model_validator(mode="after")
    def validate_time(self) -> "ContractRevisionRecord":
        require_utc(self.repaired_at_utc)
        return self


class SealedEffectiveContract(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    contract_version: int = Field(ge=1)
    request_id: str
    constraints: tuple[CapabilityConstraint, ...]
    task_requirements: tuple[TaskRequirement, ...]
    requested_scope_refs: tuple[str, ...] = ()
    conflicts: tuple[ConstraintConflict, ...] = ()
    sealed_at_utc: datetime
    canonical_hash: str = Field(min_length=1)
    parent_draft_hash: str

    @model_validator(mode="after")
    def validate_sealed(self) -> "SealedEffectiveContract":
        require_utc(self.sealed_at_utc)
        if self.conflicts:
            raise ValueError("a sealed contract cannot contain unresolved conflicts")
        if self.canonical_hash != self.calculate_hash():
            raise ValueError("sealed contract canonical_hash mismatch")
        return self


class RequestedResourceScope(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    scope_id: str
    source_constraint_ids: tuple[str, ...] = ()
    requested_description: str = Field(min_length=1, max_length=500)
    allowed_source_types: tuple[str, ...] = ()
    forbidden_source_types: tuple[str, ...] = ()
    web_access: PermissionLevel = PermissionLevel.ALLOWED
    freshness_requirement: str | None = Field(default=None, max_length=120)
    canonical_hash: str = ""


class ResolvedResourceRef(FrozenModel):
    tenant_id: str
    knowledge_base_id: str
    document_id: str
    document_version: int = Field(ge=1)
    content_hash: str


class ResolvedResourceScope(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    scope_id: str
    requested_scope_hash: str
    resources: tuple[ResolvedResourceRef, ...] = ()
    allowed_source_types: tuple[str, ...] = ()
    forbidden_source_types: tuple[str, ...] = ()
    web_access: PermissionLevel = PermissionLevel.ALLOWED
    authorization_snapshot_id: str
    resolved_at_utc: datetime
    canonical_hash: str = ""
    resolution_status: ScopeResolutionStatus

    @model_validator(mode="after")
    def validate_resolved_scope(self) -> "ResolvedResourceScope":
        require_utc(self.resolved_at_utc)
        if self.resolution_status == ScopeResolutionStatus.RESOLVED and not self.resources:
            raise ValueError("resolved scope must contain at least one resource")
        if self.resolution_status != ScopeResolutionStatus.RESOLVED and self.resources:
            raise ValueError("failed scope resolution cannot expose resources")
        return self


class CapabilityAvailability(FrozenModel):
    capability: str
    provider_or_tool: str
    status: RuntimeCapabilityStatus
    checked_at_utc: datetime
    reason_code: ReasonCode | None = None
    retryable: bool = False
    estimated_recovery_seconds: int | None = Field(default=None, ge=0)
    supported_degradations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_time(self) -> "CapabilityAvailability":
        require_utc(self.checked_at_utc)
        return self


class RuntimeCapabilitySnapshot(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    run_id: str
    observed_at_utc: datetime
    capabilities: tuple[CapabilityAvailability, ...] = ()
    canonical_hash: str = ""

    @model_validator(mode="after")
    def validate_unique_capabilities(self) -> "RuntimeCapabilitySnapshot":
        require_utc(self.observed_at_utc)
        keys = [(item.capability, item.provider_or_tool) for item in self.capabilities]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate capability/provider availability")
        return self


class ToolEffects(FrozenModel):
    network: NetworkEffect = NetworkEffect.NONE
    data: DataEffect = DataEffect.NONE
    mutation: MutationEffect = MutationEffect.NONE
    sensitive_data: SensitiveDataEffect = SensitiveDataEffect.NONE
    cost: CostEffect = CostEffect.FREE


class ToolManifest(FrozenModel):
    tool_name: str
    tool_version: str
    capability: str
    declared_max_effects: ToolEffects
    supports_idempotency: bool = False
    supports_status_query: bool = False
    supports_compensation: bool = False
    deterministic: bool = False
    result_freshness_policy: str = "no_reuse"


class ResolvedToolCall(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    step_id: str
    tool_name: str
    tool_version: str
    normalized_arguments_hash: str
    resolved_scope_hash: str
    resolved_call_effects: ToolEffects
    effect_profile_hash: str
    provider_idempotency_key: str | None = None
    canonical_hash: str = ""


class RuntimeBudget(FrozenModel):
    max_llm_calls: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    max_retrieval_queries: int = Field(ge=0)
    max_protocol_repairs: int = Field(ge=0)
    max_execution_rounds: int = Field(ge=0)
    token_budget: int | None = Field(default=None, ge=0)
    monetary_budget: Decimal | None = Field(default=None, ge=0)


class RuntimePolicyEnvelope(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    request_id: str
    hard_limits: RuntimeBudget
    soft_targets: RuntimeBudget
    reserved_delivery_budget_ms: int = Field(ge=0)
    created_at_utc: datetime
    deadline_at_utc: datetime
    original_budget_ms: int = Field(gt=0)
    source_authority: Authority
    canonical_hash: str = ""

    @model_validator(mode="after")
    def validate_deadline(self) -> "RuntimePolicyEnvelope":
        created = require_utc(self.created_at_utc)
        deadline = require_utc(self.deadline_at_utc)
        if deadline <= created:
            raise ValueError("deadline must be after creation time")
        if self.reserved_delivery_budget_ms > self.original_budget_ms:
            raise ValueError("reserved delivery budget exceeds total budget")
        return self


class StrategyStep(FrozenModel):
    step_id: str
    capability: str
    task_ids: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    tool_name: str | None = None
    scope_hash: str | None = None
    expected_effects: ToolEffects | None = None
    required_outputs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_no_self_dependency(self) -> "StrategyStep":
        if self.step_id in self.depends_on:
            raise ValueError("strategy step cannot depend on itself")
        return self


class EffectiveExecutionStrategy(CanonicalModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    run_id: str
    sealed_contract_hash: str
    resolved_scope_hashes: tuple[str, ...] = ()
    runtime_snapshot_hash: str
    preliminary_strategy_hash: str | None = None
    steps: tuple[StrategyStep, ...] = ()
    strategy_status: StrategyStatus
    reason_codes: tuple[ReasonCode, ...] = ()
    canonical_hash: str = ""


class StepIdempotencyRecord(FrozenModel):
    idempotency_key: str
    request_id: str
    run_id: str
    sealed_contract_hash: str
    resolved_scope_hash: str
    step_id: str
    tool_name: str
    tool_version: str
    normalized_arguments_hash: str
    effect_profile_hash: str
    state: StepExecutionState
    result_ref: str | None = None
    original_completed_at_utc: datetime | None = None
    freshness_validated: bool = False
    provider_idempotency_key: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "StepIdempotencyRecord":
        if self.original_completed_at_utc is not None:
            require_utc(self.original_completed_at_utc)
        if self.state == StepExecutionState.SUCCEEDED and not self.result_ref:
            raise ValueError("succeeded record requires result_ref")
        return self


class CancellationState(FrozenModel):
    requested: bool = False
    cancel_requested_at_utc: datetime | None = None
    execution_cutoff_at_utc: datetime | None = None
    reason_code: ReasonCode | None = None

    @model_validator(mode="after")
    def validate_cancellation(self) -> "CancellationState":
        if self.cancel_requested_at_utc is not None:
            require_utc(self.cancel_requested_at_utc)
        if self.execution_cutoff_at_utc is not None:
            require_utc(self.execution_cutoff_at_utc)
        if self.requested and self.cancel_requested_at_utc is None:
            raise ValueError("requested cancellation requires timestamp")
        return self


class LateResultRecord(FrozenModel):
    step_id: str
    received_at_utc: datetime
    result_ref: str
    side_effect_state: StepExecutionState
    eligible_for_future_reuse: bool

    @model_validator(mode="after")
    def validate_time(self) -> "LateResultRecord":
        require_utc(self.received_at_utc)
        return self


class CapabilityOutcome(FrozenModel):
    capability: str
    required: bool
    outcome: CapabilityOutcomeStatus
    actual_output_refs: tuple[str, ...] = ()
    runtime_status: RuntimeCapabilityStatus
    allowed_degradation_used: str | None = None
    reason_codes: tuple[ReasonCode, ...] = ()


class FinalRunStatus(FrozenModel):
    task_status: TaskStatus
    system_health: SystemHealth
    delivery_status: DeliveryStatus
    degraded_components: tuple[str, ...] = ()
    primary_reason_code: ReasonCode | None = None
    reason_codes: tuple[ReasonCode, ...] = ()
    legacy_overall_status: str


class CapabilitySatisfactionPolicy(FrozenModel):
    capability: str
    policy_version: str
    acceptable_runtime_statuses: tuple[RuntimeCapabilityStatus, ...]
    required_outputs: tuple[str, ...] = ()
    minimum_quality: dict[str, str | int | Decimal] = Field(default_factory=dict)
    allowed_degradations: tuple[str, ...] = ()


class RunAudit(FrozenModel):
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    request_id: str
    run_id: str
    trace_id: str
    parent_run_id: str | None = None
    replay_of_run_id: str | None = None

    floor_contract_hash: str
    extractor_contract_hash: str | None = None
    sealed_contract_hash: str
    resolved_scope_hashes: tuple[str, ...] = ()
    runtime_snapshot_hash: str
    strategy_hash: str
    runtime_revision: str
    schema_versions: dict[str, str]

    started_at_utc: datetime
    sealed_at_utc: datetime
    execution_started_at_utc: datetime | None = None
    completed_at_utc: datetime | None = None
    deadline_at_utc: datetime
    cancellation_state: CancellationState

    task_status: TaskStatus
    system_health: SystemHealth
    delivery_status: DeliveryStatus
    degraded_components: tuple[str, ...] = ()
    primary_reason_code: ReasonCode | None = None
    reason_codes: tuple[ReasonCode, ...] = ()
    budget_allocated: dict[str, str | int]
    budget_consumed: dict[str, str | int]
    idempotency_reuse_count: int = Field(ge=0)
    late_result_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_times(self) -> "RunAudit":
        for value in (
            self.started_at_utc,
            self.sealed_at_utc,
            self.execution_started_at_utc,
            self.completed_at_utc,
            self.deadline_at_utc,
        ):
            if value is not None:
                require_utc(value)
        return self


class ContractMigrationRecord(CanonicalModel):
    migration_id: str
    from_version: str
    to_version: str
    source_contract_hash: str
    target_draft_hash: str
    migrated_at_utc: datetime
    migration_hash: str = ""

    @model_validator(mode="after")
    def validate_time(self) -> "ContractMigrationRecord":
        require_utc(self.migrated_at_utc)
        return self


class ShadowControlPlaneResult(FrozenModel):
    request_id: str
    production_run_id: str
    shadow_revision: str
    shadow_floor_hash: str
    shadow_extractor_hash: str | None = None
    shadow_contract_hash: str | None = None
    shadow_strategy_hash: str | None = None
    status_prediction: FinalRunStatus | None = None
    sealed_contract: dict[str, Any] | None = None
    effective_strategy: dict[str, Any] | None = None
    diff_reason_codes: tuple[ReasonCode, ...] = ()
    side_effects_permitted: Literal[False] = False
    persisted_to_user_memory: Literal[False] = False


def build_step_idempotency_key(record: StepIdempotencyRecord) -> str:
    return content_hash(
        {
            "schema_version": CONTROL_PLANE_SCHEMA_VERSION,
            "request_id": record.request_id,
            "run_id": record.run_id,
            "sealed_contract_hash": record.sealed_contract_hash,
            "resolved_scope_hash": record.resolved_scope_hash,
            "step_id": record.step_id,
            "tool_name": record.tool_name,
            "tool_version": record.tool_version,
            "normalized_arguments_hash": record.normalized_arguments_hash,
            "effect_profile_hash": record.effect_profile_hash,
        }
    )
