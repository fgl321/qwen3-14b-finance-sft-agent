from __future__ import annotations

import asyncio
import json
from typing import Any

from app.agent_graph.llm_task_planner import (
    LLMTaskPlanner,
    PlannerRequest,
)


class _SequencedPlannerClient:
    def __init__(
        self,
        responses: list[dict[str, Any]],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        thinking_enabled: bool = False,
        max_completion_tokens: int = 1024,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "thinking_enabled": thinking_enabled,
                "max_completion_tokens": (
                    max_completion_tokens
                ),
            }
        )

        if not self._responses:
            raise AssertionError(
                "测试没有提供足够的 Planner 响应。"
            )

        return self._responses.pop(0)


class _FakeRegistry:
    def get_llm_tool_definitions(
        self,
        *,
        allowed_tool_names: frozenset[str] | None = None,
        allowed_tool_groups: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": (
                        "yearly_expense_to_monthly"
                    ),
                    "description": "年度必要支出转月度必要支出。",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "yearly_necessary_expense": {
                                "type": "number",
                            }
                        },
                        "required": [
                            "yearly_necessary_expense"
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "emergency_fund_range",
                    "description": "计算紧急备用金范围。",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "monthly_necessary_expense": {
                                "type": "number",
                            },
                            "min_months": {
                                "type": "integer",
                            },
                            "max_months": {
                                "type": "integer",
                            },
                        },
                        "required": [
                            "monthly_necessary_expense",
                            "min_months",
                            "max_months",
                        ],
                    },
                },
            },
        ]

        if allowed_tool_names is None:
            return definitions

        return [
            definition
            for definition in definitions
            if definition["function"]["name"]
            in allowed_tool_names
        ]

    def get(self, tool_name: str) -> None:
        return None


def _tool_call(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    tool_call_id: str,
) -> dict[str, Any]:
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(
                arguments,
                ensure_ascii=False,
            ),
        },
    }


def _planner_response(
    *,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        },
        "model": "deepseek-test",
        "finish_reason": "tool_calls",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    }


def _request() -> PlannerRequest:
    return PlannerRequest(
        request_id="req_test",
        run_id="run_test",
        user_message=(
            "我的家庭年度必要支出是18万元，"
            "请计算3到6个月的紧急备用金。"
        ),
        allowed_tool_names=frozenset(
            {
                "yearly_expense_to_monthly",
                "emergency_fund_range",
            }
        ),
        remaining_tool_calls=12,
    )


def test_respond_with_tool_required_reason_should_trigger_repair() -> None:
    client = _SequencedPlannerClient(
        [
            _planner_response(
                tool_calls=[
                    _tool_call(
                        tool_name="planner_finish",
                        tool_call_id="call_finish_1",
                        arguments={
                            "reason": (
                                "当前轮需要先调用 "
                                "yearly_expense_to_monthly，"
                                "再调用 emergency_fund_range。"
                            ),
                            "confidence": "high",
                            "needs_review": False,
                        },
                    )
                ]
            ),
            _planner_response(
                tool_calls=[
                    _tool_call(
                        tool_name=(
                            "yearly_expense_to_monthly"
                        ),
                        tool_call_id="call_business_1",
                        arguments={
                            "yearly_necessary_expense": 180000
                        },
                    )
                ]
            ),
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=_FakeRegistry(),  # type: ignore[arg-type]
        max_protocol_repairs=1,
    )

    result = asyncio.run(
        planner.plan(_request())
    )

    assert result.decision.action == "call_tools"
    assert result.attempts == 2
    assert result.protocol_repaired is True
    assert result.raw_tool_call_names == [
        "yearly_expense_to_monthly"
    ]

    repair_message = (
        client.calls[1]["messages"][-1]["content"]
    )

    assert "结构化动作与 decision_reason 互相冲突" in (
        repair_message
    )


def test_repaired_business_tool_call_should_be_accepted() -> None:
    client = _SequencedPlannerClient(
        [
            _planner_response(
                tool_calls=[
                    _tool_call(
                        tool_name="planner_finish",
                        tool_call_id="call_finish_2",
                        arguments={
                            "reason": (
                                "应先调用 emergency_fund_range。"
                            ),
                            "confidence": "medium",
                            "needs_review": False,
                        },
                    )
                ]
            ),
            _planner_response(
                tool_calls=[
                    _tool_call(
                        tool_name="emergency_fund_range",
                        tool_call_id="call_business_2",
                        arguments={
                            "monthly_necessary_expense": 15000,
                            "min_months": 3,
                            "max_months": 6,
                        },
                    )
                ]
            ),
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=_FakeRegistry(),  # type: ignore[arg-type]
        max_protocol_repairs=1,
    )

    result = asyncio.run(
        planner.plan(_request())
    )

    assert result.decision.action == "call_tools"
    assert [
        call.tool_name
        for call in result.decision.tool_calls
    ] == ["emergency_fund_range"]
    assert result.error is None


def test_normal_respond_without_pending_tool_should_pass() -> None:
    client = _SequencedPlannerClient(
        [
            _planner_response(
                tool_calls=[
                    _tool_call(
                        tool_name="planner_finish",
                        tool_call_id="call_finish_3",
                        arguments={
                            "reason": (
                                "已有信息足以回答一般概念问题，"
                                "不需要额外工具结果。"
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
        registry=_FakeRegistry(),  # type: ignore[arg-type]
        max_protocol_repairs=1,
    )

    result = asyncio.run(
        planner.plan(_request())
    )

    assert result.decision.action == "respond"
    assert result.attempts == 1
    assert result.protocol_repaired is False
    assert result.raw_tool_call_names == [
        "planner_finish"
    ]


def test_second_inconsistent_decision_should_fallback() -> None:
    inconsistent_response = _planner_response(
        tool_calls=[
            _tool_call(
                tool_name="planner_finish",
                tool_call_id="call_finish_bad",
                arguments={
                    "reason": (
                        "当前轮应先执行 "
                        "yearly_expense_to_monthly。"
                    ),
                    "confidence": "high",
                    "needs_review": False,
                },
            )
        ]
    )

    client = _SequencedPlannerClient(
        [
            inconsistent_response,
            inconsistent_response,
        ]
    )

    planner = LLMTaskPlanner(
        llm_client=client,
        registry=_FakeRegistry(),  # type: ignore[arg-type]
        max_protocol_repairs=1,
    )

    result = asyncio.run(
        planner.plan(_request())
    )

    assert result.decision.action == "fallback"
    assert result.attempts == 2
    assert result.protocol_repaired is True
    assert result.error is not None
    assert "PlannerDecisionConsistencyError" in (
        result.error
    )
    assert "yearly_expense_to_monthly" in (
        result.error
    )


def test_reason_mentions_tool_historically_but_no_pending_action_should_not_false_positive() -> None:
    client = _SequencedPlannerClient(
        [
            _planner_response(
                tool_calls=[
                    _tool_call(
                        tool_name="planner_finish",
                        tool_call_id="call_finish_4",
                        arguments={
                            "reason": (
                                "已经调用 "
                                "yearly_expense_to_monthly，"
                                "并取得真实月度支出；"
                                "emergency_fund_range "
                                "也已执行完成，结果已经齐全。"
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
        registry=_FakeRegistry(),  # type: ignore[arg-type]
        max_protocol_repairs=1,
    )

    result = asyncio.run(
        planner.plan(_request())
    )

    assert result.decision.action == "respond"
    assert result.attempts == 1
    assert result.protocol_repaired is False
