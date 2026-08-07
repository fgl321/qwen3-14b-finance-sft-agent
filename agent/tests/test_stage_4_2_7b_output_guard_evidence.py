from __future__ import annotations

import asyncio
from typing import Any

from app.agent_graph.llm_output_guard import (
    LLMOutputGuard,
    OutputGuardRequest,
    _normalize_guard_payload,
    deterministic_output_flags,
)
from app.agent_graph.schemas.loop_schema import AgentLoopResult
from app.agent_graph.schemas.planner_schema import PlannerDecision
from app.agent_graph.schemas.synthesis_schema import (
    OutputGuardResult,
    SynthesisResult,
)
from app.agent_graph.runtime.error_policy import build_tool_error
from app.agent_graph.schemas.tool_schema import ToolResult


def _loop_result(
    tool_results: list[ToolResult],
) -> AgentLoopResult:
    return AgentLoopResult(
        status="completed",
        final_decision=PlannerDecision(
            action="respond",
            decision_reason="当前信息足以生成最终回答。",
        ),
        tool_results=tool_results,
        finish_reason="planner_finished",
    )


def test_direct_answer_without_tools_is_allowed() -> None:
    synthesis = SynthesisResult(
        answer="紧急备用金通常用于覆盖突发支出。",
        used_tool_call_ids=[],
    )

    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result([]),
        citations=[],
    )

    assert "missing_used_tool_call_ids" not in flags


def test_successful_tool_requires_real_used_tool_id() -> None:
    tool_result = ToolResult(
        tool_call_id="call_success_001",
        tool_name="yearly_expense_to_monthly",
        success=True,
        output={
            "monthly_necessary_expense": "15000",
        },
        duration_ms=1,
    )
    synthesis = SynthesisResult(
        answer="月度必要支出为1.5万元。",
        used_tool_call_ids=[],
    )

    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result([tool_result]),
        citations=[],
    )

    assert "missing_used_tool_call_ids" in flags


def test_valid_successful_tool_reference_passes_evidence_check() -> None:
    tool_result = ToolResult(
        tool_call_id="call_success_002",
        tool_name="yearly_expense_to_monthly",
        success=True,
        output={
            "monthly_necessary_expense": "15000",
        },
        duration_ms=1,
    )
    synthesis = SynthesisResult(
        answer="月度必要支出为1.5万元。",
        used_tool_call_ids=["call_success_002"],
    )

    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result([tool_result]),
        citations=[],
    )

    assert "missing_used_tool_call_ids" not in flags
    assert "invalid_used_tool_call_ids" not in flags


def test_failed_tool_cannot_be_replaced_by_successful_model_calculation() -> None:
    tool_result = ToolResult(
        tool_call_id="call_failed_001",
        tool_name="emergency_fund_range",
        success=False,
        output=None,
        error=build_tool_error(
            code="TOOL_TIMEOUT",
            message="工具执行超时。",
            details={},
        ),
        duration_ms=1,
    )
    synthesis = SynthesisResult(
        answer="计算结果为4.5万元至9万元。",
        used_tool_call_ids=[],
    )

    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result([tool_result]),
        citations=[],
    )

    assert (
        "successful_conclusion_after_tool_failure"
        in flags
    )



def test_failed_tool_can_be_reported_without_fake_success() -> None:
    tool_result = ToolResult(
        tool_call_id="call_failed_002",
        tool_name="emergency_fund_range",
        success=False,
        output=None,
        error=build_tool_error(
            code="TOOL_TIMEOUT",
            message="工具执行超时。",
            details={},
        ),
        duration_ms=1,
    )
    synthesis = SynthesisResult(
        answer=(
            "工具执行失败，本次无法得到可靠的"
            "紧急备用金计算结果。"
        ),
        used_tool_call_ids=[],
    )

    flags = deterministic_output_flags(
        synthesis,
        loop_result=_loop_result([tool_result]),
        citations=[],
    )

    assert (
        "successful_conclusion_after_tool_failure"
        not in flags
    )

def test_string_null_is_normalized_to_none() -> None:
    payload = _normalize_guard_payload(
        {
            "verdict": "pass",
            "reason": "检查通过。",
            "risk_flags": [],
            "rewrite_instructions": "null",
        }
    )

    result = OutputGuardResult.model_validate(payload)

    assert result.rewrite_instructions is None
    assert (
        result.model_dump(mode="json")[
            "rewrite_instructions"
        ]
        is None
    )


class _UnexpectedLLMClient:
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        thinking_enabled: bool = False,
        max_completion_tokens: int = 800,
    ) -> dict[str, Any]:
        raise AssertionError(
            "确定性证据检查失败时不应继续调用 LLM Guard。"
        )


def test_deterministic_evidence_problem_blocks_before_llm() -> None:
    guard = LLMOutputGuard(
        llm_client=_UnexpectedLLMClient()
    )
    tool_result = ToolResult(
        tool_call_id="call_success_003",
        tool_name="yearly_expense_to_monthly",
        success=True,
        output={
            "monthly_necessary_expense": "15000",
        },
        duration_ms=1,
    )

    invocation = asyncio.run(
        guard.guard(
            OutputGuardRequest(
                request_id="request_guard_001",
                run_id="run_guard_001",
                user_message="换算月度支出。",
                loop_result=_loop_result([tool_result]),
                synthesis=SynthesisResult(
                    answer="月度必要支出为1.5万元。",
                    used_tool_call_ids=[],
                ),
                citations=[],
            )
        )
    )

    assert invocation.result.verdict == "rewrite"
    assert (
        "missing_used_tool_call_ids"
        in invocation.result.risk_flags
    )
