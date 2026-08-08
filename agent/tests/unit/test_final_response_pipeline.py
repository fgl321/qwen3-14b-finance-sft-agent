import pytest

from app.agent_graph.final_response_pipeline import (
    FinalResponsePipeline,
    FinalResponseRequest,
)
from app.agent_graph.llm_output_guard import (
    OutputGuardInvocationResult,
)
from app.agent_graph.llm_synthesizer import (
    SynthesisInvocationResult,
)
from app.agent_graph.runtime.agent_limits import (
    AgentLimits,
)
from app.agent_graph.schemas.loop_schema import (
    AgentLoopResult,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
)
from app.agent_graph.schemas.synthesis_schema import (
    OutputGuardResult,
    SynthesisResult,
)
from app.agent_graph.schemas.tool_schema import ToolResult


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeSynthesizer:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    async def synthesize(self, request):
        self.requests.append(request)
        return self.results.pop(0)


class FakeGuard:
    def __init__(self, results):
        self.results = list(results)

    async def guard(self, request):
        return self.results.pop(0)


def loop_result(status="completed"):
    clarification = (
        "请提供年度必要支出。"
        if status == "clarification_required"
        else None
    )

    action = (
        "clarify"
        if status == "clarification_required"
        else "respond"
    )

    return AgentLoopResult(
        status=status,
        final_decision=PlannerDecision(
            action=action,
            clarification_question=clarification,
            decision_reason="测试。",
            confidence="high",
            plan_version=1,
        ),
        tool_results=[
            ToolResult(
                tool_call_id="call_1",
                tool_name="emergency_fund_range",
                success=True,
                output={
                    "min_amount": "45000.00",
                    "max_amount": "90000.00",
                },
            )
        ],
        agent_rounds=1,
        total_tool_calls=1,
        clarification_question=clarification,
        finish_reason="test",
    )


def synthesis(answer):
    return SynthesisInvocationResult(
        result=SynthesisResult(
            answer=answer,
            used_tool_call_ids=["call_1"],
            used_citation_ids=[],
            uncertainty=None,
            disclaimer_required=True,
        )
    )


def guard(verdict, rewrite=None):
    return OutputGuardInvocationResult(
        result=OutputGuardResult(
            verdict=verdict,
            reason="测试检查结果。",
            risk_flags=[],
            rewrite_instructions=rewrite,
        )
    )


def request(status="completed"):
    return FinalResponseRequest(
        request_id="request_test",
        run_id="run_test",
        user_message="请计算紧急备用金。",
        loop_result=loop_result(status),
    )


@pytest.mark.anyio
async def test_pipeline_should_return_passed_answer():
    pipeline = FinalResponsePipeline(
        synthesizer=FakeSynthesizer(
            [
                synthesis(
                    "备用金建议为4.5万至9万元。"
                )
            ]
        ),
        output_guard=FakeGuard(
            [guard("pass")]
        ),
    )

    result = await pipeline.run(request())

    assert result.status == "completed"
    assert "4.5万" in result.answer
    assert "不构成" in result.answer
    assert result.output_rewrites == 0


@pytest.mark.anyio
async def test_pipeline_should_rewrite_once():
    synthesizer = FakeSynthesizer(
        [
            synthesis("第一版回答。"),
            synthesis(
                "修正后备用金为4.5万至9万元。"
            ),
        ]
    )

    pipeline = FinalResponsePipeline(
        synthesizer=synthesizer,
        output_guard=FakeGuard(
            [
                guard(
                    "rewrite",
                    "必须使用工具返回的正确金额。",
                ),
                guard("pass"),
            ]
        ),
    )

    result = await pipeline.run(request())

    assert result.status == "completed"
    assert result.output_rewrites == 1

    assert (
        synthesizer.requests[1].rewrite_instructions
        == "必须使用工具返回的正确金额。"
    )


@pytest.mark.anyio
async def test_pipeline_should_bypass_for_clarification():
    pipeline = FinalResponsePipeline(
        synthesizer=FakeSynthesizer([]),
        output_guard=FakeGuard([]),
    )

    result = await pipeline.run(
        request("clarification_required")
    )

    assert (
        result.status
        == "clarification_required"
    )

    assert (
        result.answer
        == "请提供年度必要支出。"
    )


@pytest.mark.anyio
async def test_second_rewrite_should_return_last_draft():
    limits = AgentLimits(
        max_output_rewrites=1
    )

    pipeline = FinalResponsePipeline(
        synthesizer=FakeSynthesizer(
            [
                synthesis("第一版。"),
                synthesis("第二版。"),
            ]
        ),
        output_guard=FakeGuard(
            [
                guard(
                    "rewrite",
                    "第一次修改。",
                ),
                guard(
                    "rewrite",
                    "仍然存在错误。",
                ),
            ]
        ),
        limits=limits,
    )

    result = await pipeline.run(request())

    assert result.status == "completed"
    assert result.answer.startswith("第二版。")

    assert (
        result.finish_reason
        == "max_output_rewrites_exceeded"
    )

    assert result.output_rewrites == 1
