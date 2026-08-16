import pytest
from pydantic import ValidationError

from app.agent_graph.runtime.agent_limits import (
    AgentLimits,
    DEFAULT_AGENT_LIMITS,
)
from app.agent_graph.runtime.error_policy import (
    build_error_signature,
    build_tool_error,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
    ToolCallRequest,
)
from app.agent_graph.schemas.reviewer_schema import ReviewDecision
from app.agent_graph.schemas.synthesis_schema import OutputGuardResult
from app.agent_graph.schemas.tool_schema import ToolResult


def test_planner_decision_should_accept_tool_call() -> None:
    decision = PlannerDecision(
        action="call_tools",
        tool_calls=[
            ToolCallRequest(
                tool_name="yearly_expense_to_monthly",
                arguments={
                    "yearly_necessary_expense": 180000,
                },
            )
        ],
        decision_reason="需要先把年度支出换算为月度支出。",
        confidence="high",
    )

    assert decision.action == "call_tools"
    assert len(decision.tool_calls) == 1
    assert (
        decision.tool_calls[0].tool_name
        == "yearly_expense_to_monthly"
    )


def test_call_tools_should_require_tool_call() -> None:
    with pytest.raises(ValidationError):
        PlannerDecision(
            action="call_tools",
            tool_calls=[],
        )


def test_respond_should_reject_tool_calls() -> None:
    with pytest.raises(ValidationError):
        PlannerDecision(
            action="respond",
            tool_calls=[
                ToolCallRequest(
                    tool_name="emergency_fund_range",
                    arguments={},
                )
            ],
        )


def test_clarify_should_require_question() -> None:
    with pytest.raises(ValidationError):
        PlannerDecision(
            action="clarify",
        )

    decision = PlannerDecision(
        action="clarify",
        clarification_question="请提供家庭年度必要支出。",
        decision_reason="缺少完成计算所需的金额。",
    )

    assert decision.action == "clarify"
    assert decision.clarification_question is not None


def test_review_feedback_contract() -> None:
    approved = ReviewDecision(verdict="approve")
    assert approved.verdict == "approve"

    with pytest.raises(ValidationError):
        ReviewDecision(verdict="revise")

    revised = ReviewDecision(
        verdict="revise",
        feedback="当前工具参数缺少金额来源，请重新规划。",
    )

    assert revised.verdict == "revise"


def test_tool_result_contract() -> None:
    success_result = ToolResult(
        tool_call_id="call_001",
        tool_name="yearly_expense_to_monthly",
        success=True,
        output={
            "monthly_necessary_expense": "15000",
        },
    )

    assert success_result.success is True
    assert success_result.error is None

    error = build_tool_error(
        code="ARGUMENT_SCHEMA_ERROR",
        message="缺少 yearly_necessary_expense。",
    )

    failed_result = ToolResult(
        tool_call_id="call_002",
        tool_name="yearly_expense_to_monthly",
        success=False,
        error=error,
    )

    assert failed_result.success is False
    assert failed_result.error is not None
    assert failed_result.error.model_repairable is True


def test_error_signature_should_be_stable() -> None:
    error = build_tool_error(
        code="ARGUMENT_SCHEMA_ERROR",
        message="缺少 yearly_necessary_expense。",
    )

    signature_1 = build_error_signature(
        tool_name="yearly_expense_to_monthly",
        error=error,
    )

    signature_2 = build_error_signature(
        tool_name="yearly_expense_to_monthly",
        error=error,
    )

    assert signature_1 == signature_2
    assert "ARGUMENT_SCHEMA_ERROR" in signature_1


def test_default_agent_limits_should_be_valid() -> None:
    assert DEFAULT_AGENT_LIMITS.max_agent_rounds == 6
    assert DEFAULT_AGENT_LIMITS.max_total_tool_calls == 12
    assert (
        DEFAULT_AGENT_LIMITS.default_tool_timeout_seconds
        < DEFAULT_AGENT_LIMITS.total_run_timeout_seconds
    )

    with pytest.raises(ValueError):
        AgentLimits(max_agent_rounds=0)


def test_output_guard_rewrite_should_require_instructions() -> None:
    with pytest.raises(ValidationError):
        OutputGuardResult(
            verdict="rewrite",
            reason="回答存在未经证据支持的金额。",
        )

    result = OutputGuardResult(
        verdict="rewrite",
        reason="回答存在未经证据支持的金额。",
        rewrite_instructions="删除无工具依据的金额，只保留工具结果。",
    )

    assert result.verdict == "rewrite"


def test_agent_error_envelope_accepts_rag_stage() -> None:
    from app.agent_graph.schemas.error_schema import AgentErrorEnvelope

    envelope = AgentErrorEnvelope(
        code="RAG_SERVICE_UNAVAILABLE",
        category="unavailable",
        stage="rag",
        message="知识库检索服务暂不可用。",
        retryable=True,
    )
    assert envelope.stage == "rag"
    assert envelope.retryable is True
