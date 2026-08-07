import json

import pytest

from app.agent_graph.llm_synthesizer import (
    LLMAnswerSynthesizer,
    SynthesisRequest,
)
from app.agent_graph.schemas.loop_schema import (
    AgentLoopResult,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
)
from app.agent_graph.schemas.tool_schema import ToolResult


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def model_result(arguments):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "submit_1",
                    "type": "function",
                    "function": {
                        "name": (
                            "submit_synthesis_result"
                        ),
                        "arguments": (
                            json.dumps(
                                arguments,
                                ensure_ascii=False,
                            )
                            if isinstance(arguments, dict)
                            else arguments
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
            decision_reason="工具结果齐全。",
            confidence="high",
            plan_version=3,
        ),
        tool_results=[
            ToolResult(
                tool_call_id="call_monthly",
                tool_name=(
                    "yearly_expense_to_monthly"
                ),
                success=True,
                output={
                    "monthly_necessary_expense": (
                        "15000.00"
                    )
                },
            ),
            ToolResult(
                tool_call_id="call_emergency",
                tool_name="emergency_fund_range",
                success=True,
                output={
                    "min_amount": "45000.00",
                    "max_amount": "90000.00",
                },
            ),
        ],
        agent_rounds=3,
        total_tool_calls=2,
        finish_reason="planner_finished",
    )


def request():
    return SynthesisRequest(
        request_id="request_test",
        run_id="run_test",
        user_message="请计算紧急备用金。",
        loop_result=loop_result(),
    )


@pytest.mark.anyio
async def test_should_parse_synthesis_result():
    client = FakeClient(
        [
            model_result(
                {
                    "answer": (
                        "月度必要支出为1.5万元，"
                        "紧急备用金建议为4.5万至9万元。"
                    ),
                    "used_tool_call_ids": [
                        "call_monthly",
                        "call_emergency",
                    ],
                    "used_citation_ids": [],
                    "uncertainty": None,
                    "disclaimer_required": True,
                }
            )
        ]
    )

    synthesizer = LLMAnswerSynthesizer(
        llm_client=client
    )

    result = await synthesizer.synthesize(
        request()
    )

    assert result.result is not None
    assert "4.5万" in result.result.answer

    assert result.result.used_tool_call_ids == [
        "call_monthly",
        "call_emergency",
    ]


@pytest.mark.anyio
async def test_unknown_tool_id_should_repair():
    client = FakeClient(
        [
            model_result(
                {
                    "answer": "错误答案。",
                    "used_tool_call_ids": [
                        "unknown_call"
                    ],
                    "used_citation_ids": [],
                    "uncertainty": None,
                    "disclaimer_required": True,
                }
            ),
            model_result(
                {
                    "answer": (
                        "备用金范围为4.5万至9万元。"
                    ),
                    "used_tool_call_ids": [
                        "call_emergency"
                    ],
                    "used_citation_ids": [],
                    "uncertainty": None,
                    "disclaimer_required": True,
                }
            ),
        ]
    )

    synthesizer = LLMAnswerSynthesizer(
        llm_client=client
    )

    result = await synthesizer.synthesize(
        request()
    )

    assert result.result is not None
    assert result.attempts == 2
    assert result.protocol_repaired is True


@pytest.mark.anyio
async def test_think_tag_should_repair():
    client = FakeClient(
        [
            model_result(
                {
                    "answer": (
                        "<think>内部思考</think>答案"
                    ),
                        "used_tool_call_ids": [
                            "call_monthly",
                            "call_emergency",
                        ],
                    "used_citation_ids": [],
                    "uncertainty": None,
                    "disclaimer_required": False,
                }
            ),
            model_result(
                {
                    "answer": "安全答案。",
                        "used_tool_call_ids": [
                            "call_monthly",
                            "call_emergency",
                        ],
                    "used_citation_ids": [],
                    "uncertainty": None,
                    "disclaimer_required": False,
                }
            ),
        ]
    )

    synthesizer = LLMAnswerSynthesizer(
        llm_client=client
    )

    result = await synthesizer.synthesize(
        request()
    )

    assert result.result is not None
    assert result.result.answer == "安全答案。"
    assert result.attempts == 2


@pytest.mark.anyio
async def test_client_failure_should_return_error():
    synthesizer = LLMAnswerSynthesizer(
        llm_client=FakeClient(
            [ConnectionError("测试异常")]
        )
    )

    result = await synthesizer.synthesize(
        request()
    )

    assert result.result is None
    assert result.error == "ConnectionError"

    assert "测试异常" not in str(
        result.model_dump()
    )
