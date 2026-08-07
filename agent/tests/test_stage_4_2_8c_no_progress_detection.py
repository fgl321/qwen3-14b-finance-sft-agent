from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from app.agent_graph.agent_loop import AgentToolLoop
from app.agent_graph.llm_task_planner import (
    PlannerInvocationResult,
    PlannerRequest,
)
from app.agent_graph.runtime.agent_limits import AgentLimits
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
    ToolCallRequest,
)
from app.tools.tool_executor import (
    ProductionToolExecutor,
    ToolExecutionContext,
)


class AmountInput(BaseModel):
    amount: Decimal


class FakeRegistry:
    def __init__(self, handler: Any) -> None:
        self.spec = SimpleNamespace(
            name="stable_calculation",
            description="测试计算工具",
            input_model=AmountInput,
            handler=handler,
            allowed_roles=frozenset({"user"}),
            tool_group="financial_calculation",
            side_effect=False,
            idempotent=True,
            timeout_seconds=1.0,
            max_infrastructure_retries=0,
        )

    def get(self, name: str):
        if name == self.spec.name:
            return self.spec
        return None

    def names(self) -> tuple[str, ...]:
        return (self.spec.name,)


class SequencePlanner:
    def __init__(self, decisions: list[PlannerDecision]) -> None:
        self.decisions = list(decisions)

    async def plan(
        self,
        request: PlannerRequest,
    ) -> PlannerInvocationResult:
        decision = self.decisions.pop(0)

        if decision.action == "call_tools":
            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": json.dumps(
                                call.arguments,
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for call in decision.tool_calls
                ],
            }
        else:
            assistant_message = {}

        return PlannerInvocationResult(
            decision=decision,
            assistant_message=assistant_message,
            finish_reason=(
                "tool_calls"
                if decision.action == "call_tools"
                else "planner_finish"
            ),
            raw_tool_call_names=[
                call.tool_name
                for call in decision.tool_calls
            ],
        )


def _call(call_id: str, amount: Any, version: int) -> PlannerDecision:
    return PlannerDecision(
        action="call_tools",
        tool_calls=[
            ToolCallRequest(
                tool_call_id=call_id,
                tool_name="stable_calculation",
                arguments={"amount": amount},
            )
        ],
        decision_reason="执行测试工具。",
        confidence="high",
        plan_version=version,
    )


def _finish(version: int) -> PlannerDecision:
    return PlannerDecision(
        action="respond",
        decision_reason="已有结果足够。",
        confidence="high",
        plan_version=version,
    )


def _request(tool_budget: int = 6) -> PlannerRequest:
    return PlannerRequest(
        request_id="request_no_progress_001",
        run_id="run_no_progress_001",
        user_message="请完成测试计算。",
        execution_policy="auto",
        remaining_tool_calls=tool_budget,
    )


def _context(tool_budget: int = 6) -> ToolExecutionContext:
    return ToolExecutionContext(
        request_id="request_no_progress_001",
        run_id="run_no_progress_001",
        allowed_tool_groups=frozenset(
            {"financial_calculation"}
        ),
        remaining_tool_calls=tool_budget,
    )


def test_reused_only_rounds_trigger_no_progress_fallback() -> None:
    handler_call_count = 0

    def handler(amount: Decimal) -> dict[str, str]:
        nonlocal handler_call_count
        handler_call_count += 1
        return {"amount": format(amount, "f")}

    planner = SequencePlanner(
        [
            _call("call_original", 180000, 1),
            _call("call_duplicate_1", "180000.00", 2),
            _call("call_duplicate_2", 180000.0, 3),
            _finish(4),
        ]
    )

    loop = AgentToolLoop(
        planner=planner,
        executor=ProductionToolExecutor(
            registry=FakeRegistry(handler)
        ),
        limits=AgentLimits(
            max_agent_rounds=5,
            max_total_tool_calls=2,
            max_consecutive_no_progress_rounds=2,
        ),
    )

    result = asyncio.run(
        loop.run(
            request=_request(tool_budget=2),
            execution_context=_context(tool_budget=2),
        )
    )

    assert result.status == "fallback"
    assert result.finish_reason == "no_progress_loop_detected"
    assert handler_call_count == 1
    assert result.total_tool_calls == 1
    assert result.reused_tool_call_count == 2
    assert result.no_progress_round_count == 2
    assert result.consecutive_no_progress_rounds == 2
    assert result.max_consecutive_no_progress_rounds == 2
    assert result.last_progress_round == 1
    assert len(result.no_progress_events) == 2
    assert result.no_progress_events[0].round_index == 2
    assert result.no_progress_events[1].round_index == 3


def test_single_reused_round_then_finish_is_allowed() -> None:
    handler_call_count = 0

    def handler(amount: Decimal) -> dict[str, str]:
        nonlocal handler_call_count
        handler_call_count += 1
        return {"amount": format(amount, "f")}

    loop = AgentToolLoop(
        planner=SequencePlanner(
            [
                _call("call_original", 180000, 1),
                _call("call_duplicate", "180000.00", 2),
                _finish(3),
            ]
        ),
        executor=ProductionToolExecutor(
            registry=FakeRegistry(handler)
        ),
        limits=AgentLimits(
            max_agent_rounds=4,
            max_total_tool_calls=1,
            max_consecutive_no_progress_rounds=2,
        ),
    )

    result = asyncio.run(
        loop.run(
            request=_request(tool_budget=1),
            execution_context=_context(tool_budget=1),
        )
    )

    assert result.status == "completed"
    assert result.finish_reason == "planner_finished"
    assert handler_call_count == 1
    assert result.no_progress_round_count == 1
    assert result.consecutive_no_progress_rounds == 1
    assert result.max_consecutive_no_progress_rounds == 1


def test_new_success_resets_consecutive_no_progress_counter() -> None:
    handler_call_count = 0

    def handler(amount: Decimal) -> dict[str, str]:
        nonlocal handler_call_count
        handler_call_count += 1
        return {"amount": format(amount, "f")}

    loop = AgentToolLoop(
        planner=SequencePlanner(
            [
                _call("call_100", 100, 1),
                _call("call_100_duplicate", "100.00", 2),
                _call("call_200", 200, 3),
                _call("call_200_duplicate", "200.0", 4),
                _finish(5),
            ]
        ),
        executor=ProductionToolExecutor(
            registry=FakeRegistry(handler)
        ),
        limits=AgentLimits(
            max_agent_rounds=6,
            max_total_tool_calls=2,
            max_consecutive_no_progress_rounds=2,
        ),
    )

    result = asyncio.run(
        loop.run(
            request=_request(tool_budget=2),
            execution_context=_context(tool_budget=2),
        )
    )

    assert result.status == "completed"
    assert handler_call_count == 2
    assert result.total_tool_calls == 2
    assert result.reused_tool_call_count == 2
    assert result.no_progress_round_count == 2
    assert result.max_consecutive_no_progress_rounds == 1
    assert result.consecutive_no_progress_rounds == 1
    assert result.last_progress_round == 3


def test_first_error_is_progress_repeated_errors_are_not() -> None:
    handler_call_count = 0

    def handler(amount: Decimal) -> dict[str, str]:
        nonlocal handler_call_count
        handler_call_count += 1
        raise ValueError("固定领域错误。")

    loop = AgentToolLoop(
        planner=SequencePlanner(
            [
                _call("call_fail_1", 100, 1),
                _call("call_fail_2", 100, 2),
                _call("call_fail_3", 100, 3),
                _finish(4),
            ]
        ),
        executor=ProductionToolExecutor(
            registry=FakeRegistry(handler)
        ),
        limits=AgentLimits(
            max_agent_rounds=5,
            max_total_tool_calls=4,
            max_same_error_count=10,
            max_consecutive_no_progress_rounds=2,
        ),
    )

    result = asyncio.run(
        loop.run(
            request=_request(tool_budget=4),
            execution_context=_context(tool_budget=4),
        )
    )

    assert result.status == "fallback"
    assert result.finish_reason == "no_progress_loop_detected"
    assert handler_call_count == 3
    assert result.no_progress_round_count == 2
    assert result.last_progress_round == 1
    assert result.repeated_error_count == 3


def test_no_progress_limit_must_be_positive() -> None:
    with pytest.raises(ValueError):
        AgentLimits(
            max_consecutive_no_progress_rounds=0,
        )
