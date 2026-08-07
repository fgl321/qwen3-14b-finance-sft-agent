import asyncio
import json

import pytest

from app.agent_graph.agent_loop import AgentToolLoop
from app.agent_graph.llm_task_planner import (
    LLMTaskPlanner,
    PlannerRequest,
)
from app.agent_graph.runtime.agent_limits import AgentLimits
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


class FakeDeepSeekClient:
    def __init__(self, responses, delay=0.0):
        self.responses = list(responses)
        self.calls = []
        self.delay = delay

    async def chat(self, **kwargs):
        self.calls.append(kwargs)

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if not self.responses:
            raise RuntimeError("没有剩余 Fake 响应。")

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def build_tool_call(
    *,
    call_id,
    name,
    arguments,
):
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


def build_result(
    *,
    tool_calls=None,
    content=None,
    finish_reason="tool_calls",
):
    return {
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
        },
        "model": "deepseek-test",
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


def build_request(
    *,
    message=(
        "家庭年度必要支出18万元，"
        "请计算3到6个月紧急备用金。"
    ),
    remaining_tool_calls=12,
):
    return PlannerRequest(
        request_id="request_test",
        run_id="run_test",
        user_message=message,
        allowed_tool_groups=frozenset(
            {"financial_calculation"}
        ),
        remaining_tool_calls=remaining_tool_calls,
    )


def build_context(
    *,
    remaining_tool_calls=12,
):
    return ToolExecutionContext(
        request_id="request_test",
        run_id="run_test",
        tenant_id="default",
        user_id="user_test",
        role="user",
        allowed_tool_groups=frozenset(
            {"financial_calculation"}
        ),
        remaining_tool_calls=remaining_tool_calls,
    )


def build_loop(client, *, limits=None):
    registry = build_production_tool_registry()

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=registry,
    )

    executor = ProductionToolExecutor(
        registry=registry,
        limits=limits or AgentLimits(),
    )

    return AgentToolLoop(
        planner=planner,
        executor=executor,
        limits=limits or AgentLimits(),
    )


@pytest.mark.anyio
async def test_complete_two_tool_chain():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="call_yearly",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "yearly_necessary_expense": 180000,
                        },
                    )
                ]
            ),
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="call_emergency",
                        name="emergency_fund_range",
                        arguments={
                            "monthly_necessary_expense": 15000,
                            "min_months": 3,
                            "max_months": 6,
                        },
                    )
                ]
            ),
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="control_finish",
                        name="planner_finish",
                        arguments={
                            "reason": "工具结果已经齐全。",
                            "confidence": "high",
                            "needs_review": False,
                        },
                    )
                ]
            ),
        ]
    )

    loop = build_loop(client)

    result = await loop.run(
        request=build_request(),
        execution_context=build_context(),
    )

    assert result.status == "completed"
    assert result.finish_reason == "planner_finished"

    assert result.agent_rounds == 3
    assert result.total_tool_calls == 2

    assert len(result.tool_results) == 2

    assert all(
        tool_result.success
        for tool_result in result.tool_results
    )

    assert (
        result.tool_results[0].output[
            "monthly_necessary_expense"
        ]
        == "15000.00"
    )

    assert (
        result.tool_results[1].output["min_amount"]
        == "45000.00"
    )

    assert (
        result.tool_results[1].output["max_amount"]
        == "90000.00"
    )


@pytest.mark.anyio
async def test_argument_error_should_be_repaired():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="call_bad",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "monthly_expense": 180000,
                        },
                    )
                ]
            ),
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="call_fixed",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "yearly_necessary_expense": 180000,
                        },
                    )
                ]
            ),
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="control_finish",
                        name="planner_finish",
                        arguments={
                            "reason": "参数已经修复并执行成功。",
                            "confidence": "high",
                            "needs_review": False,
                        },
                    )
                ]
            ),
        ]
    )

    loop = build_loop(client)

    result = await loop.run(
        request=build_request(),
        execution_context=build_context(),
    )

    assert result.status == "completed"
    assert result.total_tool_calls == 2

    assert result.tool_results[0].success is False

    assert (
        result.tool_results[0].error.code
        == "ARGUMENT_SCHEMA_ERROR"
    )

    assert (
        result.tool_results[0].error.model_repairable
        is True
    )

    assert result.tool_results[1].success is True

    second_call_messages = client.calls[1][
        "messages"
    ]

    assert any(
        message.get("role") == "tool"
        and "ARGUMENT_SCHEMA_ERROR"
        in str(message.get("content"))
        for message in second_call_messages
    )


@pytest.mark.anyio
async def test_should_request_clarification():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="control_clarify",
                        name=(
                            "planner_request_clarification"
                        ),
                        arguments={
                            "question": (
                                "请提供家庭年度必要支出。"
                            ),
                            "reason": "缺少计算所需金额。",
                        },
                    )
                ]
            )
        ]
    )

    loop = build_loop(client)

    result = await loop.run(
        request=build_request(
            message="请帮我算紧急备用金。"
        ),
        execution_context=build_context(),
    )

    assert (
        result.status
        == "clarification_required"
    )

    assert (
        result.clarification_question
        == "请提供家庭年度必要支出。"
    )

    assert result.total_tool_calls == 0


@pytest.mark.anyio
async def test_should_honor_planner_fallback():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="control_fallback",
                        name="planner_fallback",
                        arguments={
                            "reason": "现有工具无法完成。",
                        },
                    )
                ]
            )
        ]
    )

    loop = build_loop(client)

    result = await loop.run(
        request=build_request(),
        execution_context=build_context(),
    )

    assert result.status == "fallback"

    assert (
        result.finish_reason
        == "planner_requested_fallback"
    )


@pytest.mark.anyio
async def test_same_error_twice_should_stop_loop():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="unknown_1",
                        name="unknown_finance_tool",
                        arguments={},
                    )
                ]
            ),
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="unknown_2",
                        name="unknown_finance_tool",
                        arguments={},
                    )
                ]
            ),
        ]
    )

    loop = build_loop(client)

    result = await loop.run(
        request=build_request(),
        execution_context=build_context(),
    )

    assert result.status == "fallback"

    assert (
        result.finish_reason
        == "repeated_tool_error_limit"
    )

    assert result.repeated_error_count == 2

    assert len(result.tool_results) == 2

    assert all(
        item.error.code == "TOOL_NOT_FOUND"
        for item in result.tool_results
    )


@pytest.mark.anyio
async def test_tool_budget_should_stop_execution():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="call_budget",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "yearly_necessary_expense": 180000,
                        },
                    )
                ]
            )
        ]
    )

    loop = build_loop(client)

    result = await loop.run(
        request=build_request(
            remaining_tool_calls=0
        ),
        execution_context=build_context(
            remaining_tool_calls=0
        ),
    )

    assert result.status == "fallback"

    assert (
        result.finish_reason
        == "tool_budget_exceeded"
    )

    assert result.total_tool_calls == 0
    assert result.tool_results == []


@pytest.mark.anyio
async def test_multi_tool_plan_should_require_review():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="call_1",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "yearly_necessary_expense": 180000,
                        },
                    ),
                    build_tool_call(
                        call_id="call_2",
                        name="life_insurance_gap",
                        arguments={
                            "annual_necessary_expense": 180000,
                        },
                    ),
                ]
            )
        ]
    )

    loop = build_loop(client)

    result = await loop.run(
        request=build_request(),
        execution_context=build_context(),
    )

    assert result.status == "review_required"

    assert (
        result.finish_reason
        == "plan_review_required"
    )

    # Reviewer 通过前不能执行工具。
    assert result.total_tool_calls == 0
    assert result.tool_results == []


@pytest.mark.anyio
async def test_total_loop_timeout_should_fallback():
    limits = AgentLimits(
        max_agent_rounds=3,
        max_total_tool_calls=3,
        max_parallel_tool_calls=1,
        max_same_error_count=2,
        max_plan_revisions=2,
        max_output_rewrites=1,
        default_tool_timeout_seconds=0.01,
        total_run_timeout_seconds=0.02,
    )

    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="late_call",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "yearly_necessary_expense": 180000,
                        },
                    )
                ]
            )
        ],
        delay=0.2,
    )

    loop = build_loop(
        client,
        limits=limits,
    )

    result = await loop.run(
        request=build_request(),
        execution_context=build_context(),
    )

    assert result.status == "fallback"

    assert (
        result.finish_reason
        == "agent_run_timeout"
    )

    assert result.total_tool_calls == 0
