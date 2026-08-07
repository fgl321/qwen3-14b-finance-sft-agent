import json

import pytest

from app.agent_graph.llm_output_guard import (
    LLMOutputGuard,
    OutputGuardRequest,
    deterministic_output_flags,
)
from app.agent_graph.schemas.loop_schema import (
    AgentLoopResult,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
)
from app.agent_graph.schemas.synthesis_schema import (
    SynthesisResult,
)
from app.agent_graph.schemas.tool_schema import ToolResult


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat(self, **kwargs):
        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def guard_response(
    verdict,
    *,
    reason="检查完成。",
    flags=None,
    rewrite=None,
):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "guard_1",
                    "type": "function",
                    "function": {
                        "name": (
                            "submit_output_guard_result"
                        ),
                        "arguments": json.dumps(
                            {
                                "verdict": verdict,
                                "reason": reason,
                                "risk_flags": (
                                    flags or []
                                ),
                                "rewrite_instructions": (
                                    rewrite
                                ),
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        },
        "model": "deepseek-test",
        "finish_reason": "tool_calls",
        "usage": {},
    }


def loop_result():
    return AgentLoopResult(
        status="completed",
        final_decision=PlannerDecision(
            action="respond",
            decision_reason="完成。",
            confidence="high",
            plan_version=3,
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
        agent_rounds=3,
        total_tool_calls=1,
        finish_reason="planner_finished",
    )


def request(answer):
    return OutputGuardRequest(
        request_id="request_test",
        run_id="run_test",
        user_message="请计算紧急备用金。",
        loop_result=loop_result(),
        synthesis=SynthesisResult(
            answer=answer,
            used_tool_call_ids=["call_1"],
            used_citation_ids=[],
            uncertainty=None,
            disclaimer_required=True,
        ),
    )


def test_deterministic_guard_should_detect_unsafe_text():
    synthesis = SynthesisResult(
        answer="这是零风险收益，建议贷款投资。",
        used_tool_call_ids=[],
        used_citation_ids=[],
        uncertainty=None,
        disclaimer_required=False,
    )

    flags = deterministic_output_flags(
        synthesis
    )

    assert "guaranteed_return" in flags
    assert "leverage_encouragement" in flags


@pytest.mark.anyio
async def test_should_parse_pass():
    guard = LLMOutputGuard(
        llm_client=FakeClient(
            [guard_response("pass")]
        )
    )

    result = await guard.guard(
        request(
            "紧急备用金建议为4.5万至9万元。"
        )
    )

    assert result.result.verdict == "pass"


@pytest.mark.anyio
async def test_deterministic_issue_should_request_rewrite():
    guard = LLMOutputGuard(
        llm_client=FakeClient([])
    )

    # <think> 标签已经由 SynthesisResult 在更前面拦截。
    # 这里专门验证 Output Guard 对金融风险表述的检查。
    result = await guard.guard(
        request(
            "紧急备用金稳赚，建议贷款投资。"
        )
    )

    assert result.result.verdict == "rewrite"

    assert "guaranteed_return" in (
        result.result.risk_flags
    )

    assert "leverage_encouragement" in (
        result.result.risk_flags
    )

@pytest.mark.anyio
async def test_guard_failure_should_fallback():
    guard = LLMOutputGuard(
        llm_client=FakeClient(
            [ConnectionError("测试异常")]
        )
    )

    result = await guard.guard(
        request(
            "紧急备用金建议为4.5万至9万元。"
        )
    )

    assert result.result.verdict == "fallback"
    assert result.error == "ConnectionError"
