from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.control_plane.clock import MonotonicDeadline
from app.control_plane.enums import StepExecutionState
from app.control_plane.reason_codes import ReasonCode
from app.control_plane.schemas import CancellationState, StepIdempotencyRecord, ToolManifest


class ReplayAction(StrEnum):
    REUSE = "reuse"
    WAIT = "wait"
    RETRY = "retry"
    STATUS_QUERY = "status_query"
    DO_NOT_RETRY = "do_not_retry"
    EXECUTE_NEW = "execute_new"


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    action: ReplayAction
    reason_code: ReasonCode | None = None
    counts_as_new_tool_call: bool = False


def decide_replay(
    record: StepIdempotencyRecord | None,
    *,
    identity_matches: bool,
    freshness_valid: bool,
    manifest: ToolManifest,
) -> ReplayDecision:
    if record is None or not identity_matches:
        return ReplayDecision(ReplayAction.EXECUTE_NEW, counts_as_new_tool_call=True)
    if record.state == StepExecutionState.SUCCEEDED:
        if freshness_valid:
            return ReplayDecision(ReplayAction.REUSE)
        if manifest.deterministic:
            return ReplayDecision(ReplayAction.EXECUTE_NEW, counts_as_new_tool_call=True)
        if manifest.supports_status_query:
            return ReplayDecision(ReplayAction.STATUS_QUERY)
        return ReplayDecision(ReplayAction.EXECUTE_NEW, counts_as_new_tool_call=True)
    if record.state in {StepExecutionState.PENDING, StepExecutionState.RUNNING}:
        return ReplayDecision(ReplayAction.WAIT)
    if record.state == StepExecutionState.FAILED_RETRYABLE:
        if manifest.supports_idempotency or manifest.declared_max_effects.mutation.value == "none":
            return ReplayDecision(
                ReplayAction.RETRY,
                ReasonCode.TOOL_EXECUTION_FAILED,
                counts_as_new_tool_call=True,
            )
        if manifest.supports_status_query:
            return ReplayDecision(ReplayAction.STATUS_QUERY, ReasonCode.TOOL_RESULT_UNKNOWN)
        return ReplayDecision(ReplayAction.DO_NOT_RETRY, ReasonCode.TOOL_RESULT_UNKNOWN)
    if record.state == StepExecutionState.UNKNOWN:
        return ReplayDecision(
            ReplayAction.STATUS_QUERY if manifest.supports_status_query else ReplayAction.DO_NOT_RETRY,
            ReasonCode.TOOL_RESULT_UNKNOWN,
        )
    if record.state == StepExecutionState.FAILED_TERMINAL:
        return ReplayDecision(ReplayAction.DO_NOT_RETRY, ReasonCode.TOOL_EXECUTION_FAILED)
    if record.state == StepExecutionState.CANCELLED:
        return ReplayDecision(ReplayAction.EXECUTE_NEW, ReasonCode.RUN_CANCELLED, True)
    return ReplayDecision(ReplayAction.DO_NOT_RETRY, ReasonCode.TOOL_RESULT_UNKNOWN)


@dataclass(frozen=True, slots=True)
class RuntimeGateDecision:
    may_start_new_work: bool
    may_start_new_execution_round: bool
    reason_code: ReasonCode | None = None


def runtime_gate(
    *,
    cancellation: CancellationState,
    deadline: MonotonicDeadline,
    remaining_time_reserved_for_delivery: bool,
) -> RuntimeGateDecision:
    if cancellation.requested:
        return RuntimeGateDecision(False, False, ReasonCode.RUN_CANCELLED)
    if deadline.expired():
        return RuntimeGateDecision(False, False, ReasonCode.DEADLINE_EXCEEDED)
    if remaining_time_reserved_for_delivery:
        return RuntimeGateDecision(False, False, ReasonCode.DEADLINE_EXCEEDED)
    return RuntimeGateDecision(True, True)


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    llm_calls: int = 0
    tool_calls: int = 0
    retrieval_queries: int = 0
    protocol_repairs: int = 0
    execution_rounds: int = 0
    tokens: int = 0


def budget_reason(
    usage: BudgetUsage,
    hard_limits: object,
) -> ReasonCode | None:
    pairs = (
        (usage.llm_calls, hard_limits.max_llm_calls),
        (usage.tool_calls, hard_limits.max_tool_calls),
        (usage.retrieval_queries, hard_limits.max_retrieval_queries),
        (usage.protocol_repairs, hard_limits.max_protocol_repairs),
        (usage.execution_rounds, hard_limits.max_execution_rounds),
    )
    if any(used >= limit for used, limit in pairs):
        return ReasonCode.BUDGET_EXHAUSTED
    if hard_limits.token_budget is not None and usage.tokens >= hard_limits.token_budget:
        return ReasonCode.BUDGET_EXHAUSTED
    return None
