from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.agent_graph.llm_task_planner import (
    LLMTaskPlanner,
    PlannerRequest,
)
from app.agent_graph.production_nodes import (
    prepare_production_run_node,
)
from app.agent_graph.production_service import (
    ProductionFinanceGraphService,
)
from app.agent_graph.prompts.planner_prompt import (
    build_execution_policy_prompt,
)
from app.agent_graph.schemas.planner_schema import (
    normalize_execution_policy,
)
from app.api.routes.chat_graph_v2 import (
    ProductionChatRequest,
)


class _CaptureGraph:
    def __init__(self) -> None:
        self.graph_input: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None

    async def ainvoke(
        self,
        graph_input: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.graph_input = dict(graph_input)
        self.config = dict(config)
        return dict(graph_input)


class _FakeRegistry:
    def get_llm_tool_definitions(
        self,
        *,
        allowed_tool_names: frozenset[str] | None = None,
        allowed_tool_groups: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        definition = {
            "type": "function",
            "function": {
                "name": "yearly_expense_to_monthly",
                "description": "将年度必要支出换算为月度必要支出。",
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
        }

        if (
            allowed_tool_names is not None
            and "yearly_expense_to_monthly"
            not in allowed_tool_names
        ):
            return []

        if (
            allowed_tool_groups is not None
            and "financial_calculation"
            not in allowed_tool_groups
        ):
            return []

        return [definition]

    def get(self, tool_name: str) -> None:
        return None


class _SingleResponseClient:
    def __init__(
        self,
        response: dict[str, Any],
    ) -> None:
        self.response = response
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
        return self.response


def _tool_call_response(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    tool_call_id: str,
) -> dict[str, Any]:
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
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
            ],
        },
        "model": "deepseek-test",
        "finish_reason": "tool_calls",
        "usage": {},
    }


@pytest.mark.parametrize(
    "policy",
    [
        "direct_allowed",
        "auto",
        "prefer_tool",
        "require_tool",
    ],
)
def test_execution_policy_normalization(
    policy: str,
) -> None:
    assert normalize_execution_policy(policy) == policy


def test_execution_policy_default_is_auto() -> None:
    assert normalize_execution_policy(None) == "auto"

    payload = ProductionChatRequest(
        user_message="什么是紧急备用金？",
        user_id="user_001",
        thread_id="thread_001",
    )

    assert payload.execution_policy == "auto"


def test_invalid_execution_policy_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_execution_policy("always_call_everything")

    with pytest.raises(ValidationError):
        ProductionChatRequest(
            user_message="什么是紧急备用金？",
            user_id="user_001",
            thread_id="thread_001",
            execution_policy="invalid",  # type: ignore[arg-type]
        )


def test_service_passes_execution_policy_to_graph() -> None:
    graph = _CaptureGraph()
    service = ProductionFinanceGraphService(
        graph=graph
    )

    result = asyncio.run(
        service.run(
            user_message="请计算紧急备用金。",
            user_id="user_001",
            thread_id="thread_001",
            execution_policy="prefer_tool",
        )
    )

    assert result["execution_policy"] == "prefer_tool"
    assert graph.graph_input is not None
    assert (
        graph.graph_input["execution_policy"]
        == "prefer_tool"
    )
    assert graph.config is not None
    assert (
        graph.config["metadata"]["execution_policy"]
        == "prefer_tool"
    )


def test_prepare_node_keeps_execution_policy() -> None:
    result = prepare_production_run_node(
        {
            "user_message": "什么是紧急备用金？",
            "user_id": "user_001",
            "thread_id": "thread_001",
            "execution_policy": "direct_allowed",
        }
    )

    assert result["execution_policy"] == "direct_allowed"
    assert result["graph_version"].startswith("stage_4_2_8")


@pytest.mark.parametrize(
    ("policy", "expected_text"),
    [
        (
            "direct_allowed",
            "不要求为了形式而调用工具",
        ),
        (
            "auto",
            "自主决定",
        ),
        (
            "prefer_tool",
            "应优先调用工具",
        ),
        (
            "require_tool",
            "必须调用对应工具",
        ),
    ],
)
def test_each_policy_has_distinct_prompt(
    policy: str,
    expected_text: str,
) -> None:
    prompt = build_execution_policy_prompt(policy)
    assert f"当前执行策略：{policy}" in prompt
    assert expected_text in prompt


def test_auto_policy_can_accept_direct_finish() -> None:
    client = _SingleResponseClient(
        _tool_call_response(
            tool_name="planner_finish",
            tool_call_id="call_finish_auto",
            arguments={
                "reason": (
                    "问题信息完整且风险较低，"
                    "当前可以直接生成回答。"
                ),
                "confidence": "high",
                "needs_review": False,
            },
        )
    )
    planner = LLMTaskPlanner(
        llm_client=client,
        registry=_FakeRegistry(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        planner.plan(
            PlannerRequest(
                request_id="req_auto_direct",
                run_id="run_auto_direct",
                user_message="什么是紧急备用金？",
                execution_policy="auto",
            )
        )
    )

    assert result.decision.action == "respond"
    assert result.decision.tool_calls == []


def test_auto_policy_can_accept_model_selected_tool() -> None:
    client = _SingleResponseClient(
        _tool_call_response(
            tool_name="yearly_expense_to_monthly",
            tool_call_id="call_auto_tool",
            arguments={
                "yearly_necessary_expense": 180000,
            },
        )
    )
    planner = LLMTaskPlanner(
        llm_client=client,
        registry=_FakeRegistry(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        planner.plan(
            PlannerRequest(
                request_id="req_auto_tool",
                run_id="run_auto_tool",
                user_message=(
                    "我的年度必要支出是18万元，"
                    "请换算为每月支出。"
                ),
                execution_policy="auto",
            )
        )
    )

    assert result.decision.action == "call_tools"
    assert [
        item.tool_name
        for item in result.decision.tool_calls
    ] == ["yearly_expense_to_monthly"]


def test_direct_allowed_does_not_force_tool_call() -> None:
    client = _SingleResponseClient(
        _tool_call_response(
            tool_name="planner_finish",
            tool_call_id="call_finish_direct",
            arguments={
                "reason": "当前策略允许直接回答。",
                "confidence": "medium",
                "needs_review": False,
            },
        )
    )
    planner = LLMTaskPlanner(
        llm_client=client,
        registry=_FakeRegistry(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        planner.plan(
            PlannerRequest(
                request_id="req_direct",
                run_id="run_direct",
                user_message="解释紧急备用金的作用。",
                execution_policy="direct_allowed",
            )
        )
    )

    assert result.decision.action == "respond"

    system_messages = [
        item["content"]
        for item in client.calls[0]["messages"]
        if item.get("role") == "system"
    ]

    assert any(
        "当前执行策略：direct_allowed" in content
        for content in system_messages
    )


def test_require_tool_policy_is_visible_to_model() -> None:
    client = _SingleResponseClient(
        _tool_call_response(
            tool_name="yearly_expense_to_monthly",
            tool_call_id="call_required_tool",
            arguments={
                "yearly_necessary_expense": 180000,
            },
        )
    )
    planner = LLMTaskPlanner(
        llm_client=client,
        registry=_FakeRegistry(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        planner.plan(
            PlannerRequest(
                request_id="req_required",
                run_id="run_required",
                user_message="把18万元年支出换算成月支出。",
                execution_policy="require_tool",
            )
        )
    )

    assert result.decision.action == "call_tools"

    system_messages = [
        item["content"]
        for item in client.calls[0]["messages"]
        if item.get("role") == "system"
    ]

    assert any(
        "当前执行策略：require_tool" in content
        and "必须调用对应工具" in content
        for content in system_messages
    )
