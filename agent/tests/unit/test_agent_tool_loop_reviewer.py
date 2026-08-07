import json

import pytest

from app.agent_graph.agent_loop import AgentToolLoop
from app.agent_graph.llm_plan_reviewer import (
    LLMPlanReviewer,
)
from app.agent_graph.llm_task_planner import (
    LLMTaskPlanner,
    PlannerRequest,
)
from app.agent_graph.runtime.agent_limits import (
    AgentLimits,
)
from app.tools.runtime_registry import (
    build_production_tool_registry,
)
from app.tools.tool_executor import (
    ProductionToolExecutor,
    ToolExecutionContext,
)


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


def tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                ensure_ascii=False,
            ),
        },
    }


def model_result(tool_calls):
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        },
        "model": "deepseek-test",
        "finish_reason": "tool_calls",
        "usage": {},
    }


def review_result(verdict, feedback=""):
    return model_result(
        [
            tool_call(
                "review_call",
                "review_plan_decision",
                {
                    "verdict": verdict,
                    "feedback": feedback,
                },
            )
        ]
    )


def build_runtime(
    planner_responses,
    reviewer_responses,
    *,
    limits=None,
):
    registry = build_production_tool_registry()

    planner = LLMTaskPlanner(
        llm_client=FakeClient(
            planner_responses
        ),
        registry=registry,
    )

    reviewer = LLMPlanReviewer(
        llm_client=FakeClient(
            reviewer_responses
        ),
        registry=registry,
    )

    executor = ProductionToolExecutor(
        registry=registry
    )

    return AgentToolLoop(
        planner=planner,
        reviewer=reviewer,
        executor=executor,
        limits=limits or AgentLimits(),
    )


def request():
    return PlannerRequest(
        request_id="request_test",
        run_id="run_test",
        user_message="请进行家庭保障计算。",
        route_context={
            "complexity": "medium",
            "risk_level": "medium",
        },
        allowed_tool_groups=frozenset(
            {"financial_calculation"}
        ),
        remaining_tool_calls=12,
    )


def context():
    return ToolExecutionContext(
        request_id="request_test",
        run_id="run_test",
        user_id="user_test",
        allowed_tool_groups=frozenset(
            {"financial_calculation"}
        ),
        remaining_tool_calls=12,
    )


@pytest.mark.anyio
async def test_approved_multi_tool_plan_should_execute():
    loop = build_runtime(
        planner_responses=[
            model_result(
                [
                    tool_call(
                        "call_yearly",
                        "yearly_expense_to_monthly",
                        {
                            "yearly_necessary_expense": 180000
                        },
                    ),
                    tool_call(
                        "call_life",
                        "life_insurance_gap",
                        {
                            "annual_necessary_expense": 180000,
                            "coverage_years": 10,
                        },
                    ),
                ]
            ),
            model_result(
                [
                    tool_call(
                        "finish",
                        "planner_finish",
                        {
                            "reason": "工具结果齐全。",
                            "confidence": "high",
                            "needs_review": False,
                        },
                    )
                ]
            ),
        ],
        reviewer_responses=[
            review_result("approve")
        ],
    )

    result = await loop.run(
        request=request(),
        execution_context=context(),
    )

    assert result.status == "completed"
    assert result.total_tool_calls == 2

    assert len(result.review_invocations) == 1

    assert (
        result.review_invocations[0].verdict
        == "approve"
    )

    assert all(
        item.success
        for item in result.tool_results
    )


@pytest.mark.anyio
async def test_reviewer_revision_should_return_to_planner():
    loop = build_runtime(
        planner_responses=[
            model_result(
                [
                    tool_call(
                        "call_yearly_bad_parallel",
                        "yearly_expense_to_monthly",
                        {
                            "yearly_necessary_expense": 180000
                        },
                    ),
                    tool_call(
                        "call_emergency_bad_parallel",
                        "emergency_fund_range",
                        {
                            "monthly_necessary_expense": 15000,
                            "min_months": 3,
                            "max_months": 6,
                        },
                    ),
                ]
            ),
            model_result(
                [
                    tool_call(
                        "call_yearly_fixed",
                        "yearly_expense_to_monthly",
                        {
                            "yearly_necessary_expense": 180000
                        },
                    )
                ]
            ),
            model_result(
                [
                    tool_call(
                        "finish",
                        "planner_finish",
                        {
                            "reason": "第一步执行完成。",
                            "confidence": "high",
                            "needs_review": False,
                        },
                    )
                ]
            ),
        ],
        reviewer_responses=[
            review_result(
                "revise",
                "两个工具存在依赖，先执行年度转月度工具。",
            ),
            review_result("approve"),
        ],
    )

    result = await loop.run(
        request=request(),
        execution_context=context(),
    )

    assert result.status == "completed"

    assert result.plan_revision_count == 1

    assert len(result.review_invocations) == 2

    assert (
        result.review_invocations[0].verdict
        == "revise"
    )

    assert (
        result.review_invocations[1].verdict
        == "approve"
    )

    assert result.total_tool_calls == 1

    assert (
        result.tool_results[0].tool_name
        == "yearly_expense_to_monthly"
    )


@pytest.mark.anyio
async def test_reviewer_can_request_clarification():
    loop = build_runtime(
        planner_responses=[
            model_result(
                [
                    tool_call(
                        "call_1",
                        "yearly_expense_to_monthly",
                        {
                            "yearly_necessary_expense": 180000
                        },
                    ),
                    tool_call(
                        "call_2",
                        "life_insurance_gap",
                        {
                            "annual_necessary_expense": 180000
                        },
                    ),
                ]
            )
        ],
        reviewer_responses=[
            review_result(
                "clarify",
                "请确认需要计算哪位家庭成员的寿险缺口。",
            )
        ],
    )

    result = await loop.run(
        request=request(),
        execution_context=context(),
    )

    assert (
        result.status
        == "clarification_required"
    )

    assert result.total_tool_calls == 0

    assert (
        result.clarification_question
        == "请确认需要计算哪位家庭成员的寿险缺口。"
    )


@pytest.mark.anyio
async def test_revision_limit_should_fallback():
    limits = AgentLimits(
        max_agent_rounds=6,
        max_total_tool_calls=12,
        max_parallel_tool_calls=4,
        max_same_error_count=2,
        max_plan_revisions=2,
        max_output_rewrites=1,
        default_tool_timeout_seconds=10,
        total_run_timeout_seconds=120,
    )

    invalid_plan = model_result(
        [
            tool_call(
                "call_1",
                "yearly_expense_to_monthly",
                {
                    "yearly_necessary_expense": 180000
                },
            ),
            tool_call(
                "call_2",
                "emergency_fund_range",
                {
                    "monthly_necessary_expense": 15000
                },
            ),
        ]
    )

    loop = build_runtime(
        planner_responses=[
            invalid_plan,
            invalid_plan,
            invalid_plan,
        ],
        reviewer_responses=[
            review_result(
                "revise",
                "请拆分依赖工具。",
            ),
            review_result(
                "revise",
                "仍然没有拆分依赖工具。",
            ),
            review_result(
                "revise",
                "计划仍然不符合要求。",
            ),
        ],
        limits=limits,
    )

    result = await loop.run(
        request=request(),
        execution_context=context(),
    )

    assert result.status == "fallback"

    assert (
        result.finish_reason
        == "max_plan_revisions_exceeded"
    )

    assert result.plan_revision_count == 2

    assert result.total_tool_calls == 0
