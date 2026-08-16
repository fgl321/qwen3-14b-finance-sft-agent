from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent_graph.schemas.planner_schema import PlannerDecision
from app.agent_graph.schemas.tool_schema import (
    ToolResult,
    ToolTraceEntry,
)


AgentLoopStatus = Literal[
    "completed",
    "clarification_required",
    "review_required",
    "fallback",
]


class PlannerInvocationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1)
    invocation_index: int = Field(default=1, ge=1)
    completed_execution_rounds: int = Field(default=0, ge=0)
    target_execution_round: int = Field(default=1, ge=1)
    creates_execution_round: bool = False
    plan_repair: bool = False
    plan_attempt_in_round: int = Field(default=1, ge=1)

    action: str

    tool_names: list[str] = Field(default_factory=list)

    model: str | None = None
    finish_reason: str = ""

    usage: dict[str, Any] = Field(default_factory=dict)

    attempts: int = Field(default=1, ge=1)

    protocol_repaired: bool = False

    raw_tool_call_names: list[str] = Field(
        default_factory=list
    )

    plan_signature: str | None = None
    previous_plan_signature: str | None = None

    error: str | None = None


class PlanReviewInvocationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1)
    planner_invocation_index: int = Field(default=1, ge=1)
    target_execution_round: int = Field(default=1, ge=1)
    plan_attempt_in_round: int = Field(default=1, ge=1)
    plan_repair_count: int = Field(default=0, ge=0)

    verdict: str
    feedback: str = ""

    model: str | None = None
    finish_reason: str = ""

    usage: dict[str, Any] = Field(default_factory=dict)

    attempts: int = Field(default=1, ge=1)

    protocol_repaired: bool = False

    semantic_consistency_checked: bool = False

    verdict_normalized: bool = False

    error: str | None = None


class ToolReuseAudit(BaseModel):
    """
    Planner 再次请求一个已经成功完成的等价工具调用时，
    Python 运行时复用原结果所产生的审计记录。
    """

    model_config = ConfigDict(extra="forbid")

    requested_tool_call_id: str = Field(
        min_length=1,
        max_length=128,
    )

    reused_from_tool_call_id: str = Field(
        min_length=1,
        max_length=128,
    )

    tool_name: str = Field(
        min_length=1,
        max_length=128,
    )

    signature: str = Field(
        min_length=1,
        max_length=256,
    )

    round_index: int = Field(ge=1)




class NoProgressAudit(BaseModel):
    """
    某一工具执行轮没有产生任何新证据时的审计记录。

    它只记录工具名、数量和原因，不保存工具参数原值。
    """

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1)

    reason: str = Field(
        min_length=1,
        max_length=200,
    )

    tool_names: list[str] = Field(default_factory=list)

    executed_call_count: int = Field(default=0, ge=0)
    reused_call_count: int = Field(default=0, ge=0)

    consecutive_count: int = Field(ge=1)


class AgentLoopResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AgentLoopStatus

    final_decision: PlannerDecision

    agent_messages: list[dict[str, Any]] = Field(
        default_factory=list
    )

    tool_results: list[ToolResult] = Field(
        default_factory=list
    )

    tool_traces: list[ToolTraceEntry] = Field(
        default_factory=list
    )

    planner_invocations: list[
        PlannerInvocationAudit
    ] = Field(default_factory=list)

    review_invocations: list[
        PlanReviewInvocationAudit
    ] = Field(default_factory=list)

    agent_rounds: int = Field(default=0, ge=0)

    execution_round: int = Field(default=0, ge=0)

    completed_execution_rounds: int = Field(default=0, ge=0)

    target_execution_round: int = Field(default=1, ge=1)

    planner_invocation_count: int = Field(default=0, ge=0)

    plan_attempt_in_round: int = Field(default=0, ge=0)

    plan_repair_count: int = Field(default=0, ge=0)

    replan_count: int = Field(default=0, ge=0)

    execution_round_history: list[dict[str, Any]] = Field(
        default_factory=list
    )

    total_tool_calls: int = Field(default=0, ge=0)

    reused_tool_calls: list[ToolReuseAudit] = Field(
        default_factory=list
    )

    reused_tool_call_count: int = Field(
        default=0,
        ge=0,
    )

    no_progress_events: list[NoProgressAudit] = Field(
        default_factory=list
    )

    no_progress_round_count: int = Field(
        default=0,
        ge=0,
    )

    consecutive_no_progress_rounds: int = Field(
        default=0,
        ge=0,
    )

    max_consecutive_no_progress_rounds: int = Field(
        default=0,
        ge=0,
    )

    last_progress_round: int | None = Field(
        default=None,
        ge=1,
    )

    repeated_error_count: int = Field(default=0, ge=0)

    plan_revision_count: int = Field(default=0, ge=0)

    last_error_signature: str | None = None

    clarification_question: str | None = None

    finish_reason: str
