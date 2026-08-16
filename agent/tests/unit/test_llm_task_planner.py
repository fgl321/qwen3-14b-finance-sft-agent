import json

import pytest

from app.agent_graph.llm_task_planner import (
    LLMTaskPlanner,
    PlannerRequest,
    build_tool_feedback_message,
)
from app.agent_graph.runtime.error_policy import build_tool_error
from app.agent_graph.schemas.tool_schema import ToolResult
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

        if not self.responses:
            raise RuntimeError("没有剩余 Fake 响应。")

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


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


def build_request(**overrides):
    payload = {
        "request_id": "request_test",
        "run_id": "run_test",
        "user_message": (
            "家庭年度必要支出18万元，"
            "请计算3到6个月紧急备用金。"
        ),
        "allowed_tool_groups": frozenset(
            {"financial_calculation"}
        ),
        "agent_round": 1,
        "remaining_tool_calls": 12,
    }

    payload.update(overrides)

    return PlannerRequest(**payload)


@pytest.mark.anyio
async def test_should_parse_business_tool_call():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="call_001",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "yearly_necessary_expense": 180000,
                        },
                    )
                ]
            )
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "call_tools"
    assert len(result.decision.tool_calls) == 1

    tool_call = result.decision.tool_calls[0]

    assert tool_call.tool_call_id == "call_001"
    assert (
        tool_call.tool_name
        == "yearly_expense_to_monthly"
    )

    assert tool_call.arguments == {
        "yearly_necessary_expense": 180000
    }


@pytest.mark.anyio
async def test_structured_plan_control_builds_executable_dag_message():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="control_plan",
                        name="planner_submit_tool_plan",
                        arguments={
                            "steps": [
                                {
                                    "step_id": "monthly",
                                    "tool_name": "yearly_expense_to_monthly",
                                    "arguments": {
                                        "yearly_necessary_expense": 180000
                                    },
                                    "depends_on": [],
                                },
                                {
                                    "step_id": "reserve",
                                    "tool_name": "emergency_fund_range",
                                    "arguments": {
                                        "monthly_necessary_expense": {
                                            "$ref": {
                                                "step_id": "monthly",
                                                "path": [
                                                    "monthly_necessary_expense"
                                                ],
                                            }
                                        },
                                        "min_months": 3,
                                        "max_months": 6,
                                    },
                                    "depends_on": ["monthly"],
                                },
                            ],
                            "reason": "计算备用金工具链。",
                            "confidence": "high",
                            "needs_review": False,
                        },
                    )
                ]
            )
        ]
    )
    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "call_tools"
    assert result.decision.tool_calls[1].depends_on == ["monthly"]
    emitted_ids = {
        item["id"] for item in result.assistant_message["tool_calls"]
    }
    assert emitted_ids == {
        item.tool_call_id for item in result.decision.tool_calls
    }


@pytest.mark.anyio
async def test_repair_rejects_semantically_identical_plan():
    same_call = build_tool_call(
        call_id="new_id",
        name="yearly_expense_to_monthly",
        arguments={"yearly_necessary_expense": 180000},
    )
    client = FakeDeepSeekClient(
        [build_result(tool_calls=[same_call]), build_result(tool_calls=[same_call])]
    )
    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )
    request = build_request(
        review_feedback="use a structurally different dependency plan",
        previous_plan={
            "action": "call_tools",
            "tool_calls": [
                {
                    "tool_call_id": "old_id",
                    "tool_name": "yearly_expense_to_monthly",
                    "arguments": {"yearly_necessary_expense": 180000},
                    "step_id": None,
                    "depends_on": [],
                }
            ],
        },
    )

    result = await planner.plan(request)

    assert result.decision.action == "fallback"
    assert result.error.startswith("PlannerProtocolError:")
    assert result.attempts == 2


@pytest.mark.anyio
async def test_should_parse_clarification_control():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="control_001",
                        name=(
                            "planner_request_clarification"
                        ),
                        arguments={
                            "question": (
                                "请提供家庭年度必要支出。"
                            ),
                            "reason": (
                                "缺少完成计算所需的金额。"
                            ),
                        },
                    )
                ]
            )
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(
        build_request(
            user_message="帮我算紧急备用金。"
        )
    )

    assert result.decision.action == "clarify"

    assert (
        result.decision.clarification_question
        == "请提供家庭年度必要支出。"
    )


@pytest.mark.anyio
async def test_should_parse_finish_control():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="control_finish",
                        name="planner_finish",
                        arguments={
                            "reason": (
                                "所需工具结果已经齐全。"
                            ),
                            "confidence": "high",
                            "needs_review": False,
                        },
                    )
                ]
            )
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "respond"
    assert result.decision.confidence == "high"
    assert result.decision.needs_review is False


@pytest.mark.anyio
async def test_should_parse_fallback_control():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="control_fallback",
                        name="planner_fallback",
                        arguments={
                            "reason": (
                                "现有工具无法完成该请求。"
                            )
                        },
                    )
                ]
            )
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "fallback"
    assert result.decision.confidence == "low"


@pytest.mark.anyio
async def test_invalid_json_should_repair_once():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="bad_call",
                        name="yearly_expense_to_monthly",
                        arguments="{invalid-json",
                    )
                ]
            ),
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="good_call",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "yearly_necessary_expense": 180000,
                        },
                    )
                ]
            ),
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
        max_protocol_repairs=1,
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "call_tools"
    assert result.attempts == 2
    assert result.protocol_repaired is True

    assert len(client.calls) == 2

    second_messages = client.calls[1]["messages"]

    assert any(
        "PlannerProtocolError"
        in str(message.get("content", ""))
        for message in second_messages
    )


@pytest.mark.anyio
async def test_mixed_control_and_business_should_repair():
    client = FakeDeepSeekClient(
        [
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="business_call",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "yearly_necessary_expense": 180000,
                        },
                    ),
                    build_tool_call(
                        call_id="finish_call",
                        name="planner_finish",
                        arguments={
                            "reason": "已经完成。",
                            "confidence": "high",
                            "needs_review": False,
                        },
                    ),
                ]
            ),
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="business_retry",
                        name="yearly_expense_to_monthly",
                        arguments={
                            "yearly_necessary_expense": 180000,
                        },
                    )
                ]
            ),
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "call_tools"
    assert result.attempts == 2
    assert result.protocol_repaired is True


@pytest.mark.anyio
async def test_plain_content_should_recover_as_respond():
    client = FakeDeepSeekClient(
        [
            build_result(
                content="现有信息已经足够。",
                finish_reason="stop",
            ),
            build_result(
                content="仍然直接回答。",
                finish_reason="stop",
            ),
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "respond"
    assert result.decision.tool_calls == []
    assert result.decision.needs_review is True
    assert result.protocol_repaired is True
    assert result.attempts == 2

    # Planner 文本不能直接作为最终回答。
    assert (
        "现有信息已经足够"
        not in result.decision.decision_reason
    )


@pytest.mark.anyio
async def test_plain_content_retry_accepts_protocol_response():
    client = FakeDeepSeekClient(
        [
            build_result(
                content="现有信息已经足够。",
                finish_reason="stop",
            ),
            build_result(
                tool_calls=[
                    build_tool_call(
                        call_id="call_2",
                        name="planner_finish",
                        arguments={
                            "reason": "信息已经足够。",
                            "confidence": "high",
                            "needs_review": False,
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "respond"
    assert result.decision.needs_review is False
    assert result.attempts == 2
    assert result.protocol_repaired is True


@pytest.mark.anyio
async def test_content_json_should_be_supported():
    client = FakeDeepSeekClient(
        [
            build_result(
                content=json.dumps(
                    {
                        "action": "clarify",
                        "tool_calls": [],
                        "clarification_question": (
                            "请提供家庭年度必要支出。"
                        ),
                        "decision_reason": (
                            "缺少计算所需金额。"
                        ),
                        "confidence": "medium",
                        "needs_review": False,
                        "plan_version": 1,
                    },
                    ensure_ascii=False,
                ),
                finish_reason="stop",
            )
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "clarify"


@pytest.mark.anyio
async def test_client_exception_should_safe_fallback():
    client = FakeDeepSeekClient(
        [
            ConnectionError("测试连接失败"),
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=build_production_tool_registry(),
    )

    result = await planner.plan(build_request())

    assert result.decision.action == "fallback"
    assert result.error == "ConnectionError"

    # 原始错误内容不进入 Planner 结果。
    assert "测试连接失败" not in str(
        result.model_dump()
    )


def test_should_filter_business_tools():
    planner = LLMTaskPlanner(
        llm_client=FakeDeepSeekClient([]),
        registry=build_production_tool_registry(),
    )

    definitions = planner.get_planner_tool_definitions(
        allowed_tool_names=frozenset(
            {"yearly_expense_to_monthly"}
        )
    )

    names = [
        item["function"]["name"]
        for item in definitions
    ]

    assert "yearly_expense_to_monthly" in names

    assert "emergency_fund_range" not in names
    assert "life_insurance_gap" not in names

    assert "planner_request_clarification" in names
    assert "planner_finish" in names
    assert "planner_fallback" in names


def test_build_messages_should_include_context_and_agent_messages():
    planner = LLMTaskPlanner(
        llm_client=FakeDeepSeekClient([]),
        registry=build_production_tool_registry(),
    )

    messages = planner.build_messages(
        build_request(
            history_messages=[
                {
                    "role": "user",
                    "content": "之前的问题",
                },
                {
                    "role": "assistant",
                    "content": "之前的回答",
                },
            ],
            context_summary="年度必要支出为18万元。",
            agent_messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        build_tool_call(
                            call_id="call_previous",
                            name=(
                                "yearly_expense_to_monthly"
                            ),
                            arguments={
                                "yearly_necessary_expense": 180000,
                            },
                        )
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_previous",
                    "content": (
                        '{"success":true,'
                        '"output":{"monthly":15000}}'
                    ),
                },
            ],
        )
    )

    assert messages[0]["role"] == "system"

    assert any(
        "<context_data>" in str(
            message.get("content", "")
        )
        for message in messages
    )

    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id")
        == "call_previous"
        for message in messages
    )


def test_build_tool_feedback_message():
    success_message = build_tool_feedback_message(
        ToolResult(
            tool_call_id="call_success",
            tool_name="yearly_expense_to_monthly",
            success=True,
            output={
                "monthly_necessary_expense": "15000.00",
            },
        )
    )

    assert success_message["role"] == "tool"
    assert (
        success_message["tool_call_id"]
        == "call_success"
    )

    success_payload = json.loads(
        success_message["content"]
    )

    assert success_payload["success"] is True

    failure_message = build_tool_feedback_message(
        ToolResult(
            tool_call_id="call_failure",
            tool_name="yearly_expense_to_monthly",
            success=False,
            error=build_tool_error(
                code="ARGUMENT_SCHEMA_ERROR",
                message=(
                    "缺少 yearly_necessary_expense。"
                ),
            ),
        )
    )

    failure_payload = json.loads(
        failure_message["content"]
    )

    assert failure_payload["success"] is False

    assert (
        failure_payload["error"]["code"]
        == "ARGUMENT_SCHEMA_ERROR"
    )

    assert (
        failure_payload["error"]["model_repairable"]
        is True
    )
