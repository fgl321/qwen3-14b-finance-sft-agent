import json

import pytest

from app.agent_graph.llm_plan_reviewer import (
    LLMPlanReviewer,
    PlanReviewPolicy,
    PlanReviewRequest,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
    ToolCallRequest,
)
from app.tools.runtime_registry import (
    build_production_tool_registry,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeDeepSeekClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def build_review_call(
    *,
    verdict,
    feedback="",
    arguments_override=None,
):
    arguments = (
        arguments_override
        if arguments_override is not None
        else {
            "verdict": verdict,
            "feedback": feedback,
        }
    )

    return {
        "id": "review_call",
        "type": "function",
        "function": {
            "name": "review_plan_decision",
            "arguments": (
                json.dumps(arguments, ensure_ascii=False)
                if isinstance(arguments, dict)
                else arguments
            ),
        },
    }


def build_response(*, tool_calls=None, content=None):
    return {
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
        },
        "model": "deepseek-test",
        "finish_reason": "tool_calls",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


def build_multi_tool_decision():
    return PlannerDecision(
        action="call_tools",
        tool_calls=[
            ToolCallRequest(
                tool_call_id="call_1",
                tool_name="yearly_expense_to_monthly",
                arguments={
                    "yearly_necessary_expense": 180000
                },
            ),
            ToolCallRequest(
                tool_call_id="call_2",
                tool_name="life_insurance_gap",
                arguments={
                    "annual_necessary_expense": 180000,
                    "coverage_years": 10,
                },
            ),
        ],
        decision_reason="执行两个独立计算。",
        confidence="medium",
        needs_review=True,
        plan_version=1,
    )


def build_request(decision=None):
    return PlanReviewRequest(
        request_id="request_test",
        run_id="run_test",
        user_message="请进行家庭保障计算。",
        decision=(
            decision or build_multi_tool_decision()
        ),
        route_context={
            "complexity": "high",
            "risk_level": "medium",
        },
    )


def test_policy_should_review_multi_tool_plan():
    registry = build_production_tool_registry()

    policy = PlanReviewPolicy(
        registry=registry
    )

    assert policy.should_review(
        decision=build_multi_tool_decision(),
        route_context={},
        repeated_error_count=0,
    ) is True


@pytest.mark.anyio
async def test_should_parse_approve():
    client = FakeDeepSeekClient(
        [
            build_response(
                tool_calls=[
                    build_review_call(
                        verdict="approve"
                    )
                ]
            )
        ]
    )

    reviewer = LLMPlanReviewer(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await reviewer.review(
        build_request()
    )

    assert result.decision.verdict == "approve"
    assert result.attempts == 1
    assert result.protocol_repaired is False


@pytest.mark.anyio
async def test_invalid_arguments_should_repair():
    client = FakeDeepSeekClient(
        [
            build_response(
                tool_calls=[
                    build_review_call(
                        verdict="approve",
                        arguments_override="{invalid-json",
                    )
                ]
            ),
            build_response(
                tool_calls=[
                    build_review_call(
                        verdict="revise",
                        feedback="请拆分依赖工具。",
                    )
                ]
            ),
        ]
    )

    reviewer = LLMPlanReviewer(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await reviewer.review(
        build_request()
    )

    assert result.decision.verdict == "revise"
    assert result.attempts == 2
    assert result.protocol_repaired is True


@pytest.mark.anyio
async def test_should_support_plain_json_content():
    client = FakeDeepSeekClient(
        [
            build_response(
                content=json.dumps(
                    {
                        "verdict": "clarify",
                        "feedback": (
                            "请提供家庭年度必要支出。"
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        ]
    )

    reviewer = LLMPlanReviewer(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await reviewer.review(
        build_request()
    )

    assert result.decision.verdict == "clarify"


@pytest.mark.anyio
async def test_client_error_should_reject():
    client = FakeDeepSeekClient(
        [
            ConnectionError("测试连接异常")
        ]
    )

    reviewer = LLMPlanReviewer(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await reviewer.review(
        build_request()
    )

    assert result.decision.verdict == "reject"
    assert result.error == "ConnectionError"

    assert "测试连接异常" not in str(
        result.model_dump()
    )
