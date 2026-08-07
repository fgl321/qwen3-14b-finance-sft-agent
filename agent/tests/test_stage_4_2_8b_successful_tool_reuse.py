from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

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
    def __init__(
        self,
        handler: Any,
        *,
        idempotent: bool = True,
        side_effect: bool = False,
    ) -> None:
        self.spec = SimpleNamespace(
            name="stable_calculation",
            description="测试计算工具",
            input_model=AmountInput,
            handler=handler,
            allowed_roles=frozenset({"user"}),
            tool_group="financial_calculation",
            side_effect=side_effect,
            idempotent=idempotent,
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
    def __init__(
        self,
        decisions: list[PlannerDecision],
    ) -> None:
        self.decisions = list(decisions)

    async def plan(
        self,
        request: PlannerRequest,
    ) -> PlannerInvocationResult:
        decision = self.decisions.pop(0)

        if decision.action == "call_tools":
            tool_calls = [
                {
                    "id": item.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": item.tool_name,
                        "arguments": json.dumps(
                            item.arguments,
                            ensure_ascii=False,
                        ),
                    },
                }
                for item in decision.tool_calls
            ]

            assistant_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
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
                item.tool_name
                for item in decision.tool_calls
            ],
        )


def _call_decision(
    call_id: str,
    amount: Any,
    *,
    version: int,
) -> PlannerDecision:
    return PlannerDecision(
        action="call_tools",
        tool_calls=[
            ToolCallRequest(
                tool_call_id=call_id,
                tool_name="stable_calculation",
                arguments={"amount": amount},
            )
        ],
        decision_reason="执行确定性计算工具。",
        confidence="high",
        plan_version=version,
    )


def _finish_decision(version: int) -> PlannerDecision:
    return PlannerDecision(
        action="respond",
        decision_reason="现有成功工具结果已经足够。",
        confidence="high",
        plan_version=version,
    )


def _request(tool_budget: int) -> PlannerRequest:
    return PlannerRequest(
        request_id="request_reuse_001",
        run_id="run_reuse_001",
        user_message="请完成测试计算。",
        execution_policy="auto",
        remaining_tool_calls=tool_budget,
    )


def _context(tool_budget: int) -> ToolExecutionContext:
    return ToolExecutionContext(
        request_id="request_reuse_001",
        run_id="run_reuse_001",
        allowed_tool_groups=frozenset(
            {"financial_calculation"}
        ),
        remaining_tool_calls=tool_budget,
    )


def test_equivalent_validated_arguments_share_signature() -> None:
    registry = FakeRegistry(lambda amount: {"amount": amount})
    executor = ProductionToolExecutor(registry=registry)

    signature_one = executor.build_reuse_signature(
        ToolCallRequest(
            tool_call_id="call_one",
            tool_name="stable_calculation",
            arguments={"amount": 180000},
        ),
        context=_context(1),
    )

    signature_two = executor.build_reuse_signature(
        ToolCallRequest(
            tool_call_id="call_two",
            tool_name="stable_calculation",
            arguments={"amount": "180000.00"},
        ),
        context=_context(1),
    )

    assert signature_one is not None
    assert signature_one == signature_two


def test_different_arguments_do_not_share_signature() -> None:
    registry = FakeRegistry(lambda amount: {"amount": amount})
    executor = ProductionToolExecutor(registry=registry)

    signature_one = executor.build_reuse_signature(
        ToolCallRequest(
            tool_call_id="call_one",
            tool_name="stable_calculation",
            arguments={"amount": 180000},
        ),
        context=_context(1),
    )

    signature_two = executor.build_reuse_signature(
        ToolCallRequest(
            tool_call_id="call_two",
            tool_name="stable_calculation",
            arguments={"amount": 120000},
        ),
        context=_context(1),
    )

    assert signature_one is not None
    assert signature_two is not None
    assert signature_one != signature_two


def test_side_effect_or_non_idempotent_tool_is_not_reused() -> None:
    side_effect_executor = ProductionToolExecutor(
        registry=FakeRegistry(
            lambda amount: {"amount": amount},
            side_effect=True,
        )
    )
    non_idempotent_executor = ProductionToolExecutor(
        registry=FakeRegistry(
            lambda amount: {"amount": amount},
            idempotent=False,
        )
    )

    call = ToolCallRequest(
        tool_call_id="call_one",
        tool_name="stable_calculation",
        arguments={"amount": 180000},
    )

    assert (
        side_effect_executor.build_reuse_signature(
            call,
            context=_context(1),
        )
        is None
    )

    assert (
        non_idempotent_executor.build_reuse_signature(
            call,
            context=_context(1),
        )
        is None
    )


def test_successful_equivalent_call_is_executed_once() -> None:
    handler_call_count = 0

    def handler(amount: Decimal) -> dict[str, str]:
        nonlocal handler_call_count
        handler_call_count += 1
        return {
            "amount": format(amount, "f"),
        }

    executor = ProductionToolExecutor(
        registry=FakeRegistry(handler)
    )

    planner = SequencePlanner(
        [
            _call_decision(
                "call_original",
                180000,
                version=1,
            ),
            _call_decision(
                "call_duplicate",
                "180000.00",
                version=2,
            ),
            _finish_decision(version=3),
        ]
    )

    loop = AgentToolLoop(
        planner=planner,
        executor=executor,
        limits=AgentLimits(
            max_agent_rounds=4,
            max_total_tool_calls=1,
        ),
    )

    result = asyncio.run(
        loop.run(
            request=_request(tool_budget=1),
            execution_context=_context(tool_budget=1),
        )
    )

    assert result.status == "completed"
    assert handler_call_count == 1
    assert result.total_tool_calls == 1
    assert result.reused_tool_call_count == 1
    assert len(result.tool_results) == 1
    assert len(result.tool_traces) == 1
    assert len(result.reused_tool_calls) == 1

    reuse_audit = result.reused_tool_calls[0]

    assert (
        reuse_audit.requested_tool_call_id
        == "call_duplicate"
    )
    assert (
        reuse_audit.reused_from_tool_call_id
        == "call_original"
    )

    tool_messages = [
        message
        for message in result.agent_messages
        if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 2
    assert (
        tool_messages[1]["tool_call_id"]
        == "call_duplicate"
    )

    reused_payload = json.loads(
        tool_messages[1]["content"]
    )

    assert reused_payload["success"] is True
    assert reused_payload["reused"] is True
    assert (
        reused_payload["reused_from_tool_call_id"]
        == "call_original"
    )


def test_failed_call_is_not_cached_for_reuse() -> None:
    handler_call_count = 0

    def handler(amount: Decimal) -> dict[str, str]:
        nonlocal handler_call_count
        handler_call_count += 1

        if handler_call_count == 1:
            raise ValueError("第一次执行失败。")

        return {
            "amount": format(amount, "f"),
        }

    executor = ProductionToolExecutor(
        registry=FakeRegistry(handler)
    )

    planner = SequencePlanner(
        [
            _call_decision(
                "call_failed",
                180000,
                version=1,
            ),
            _call_decision(
                "call_retry",
                "180000.00",
                version=2,
            ),
            _finish_decision(version=3),
        ]
    )

    loop = AgentToolLoop(
        planner=planner,
        executor=executor,
        limits=AgentLimits(
            max_agent_rounds=4,
            max_total_tool_calls=2,
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
    assert result.reused_tool_call_count == 0
    assert len(result.tool_results) == 2
    assert result.tool_results[0].success is False
    assert result.tool_results[1].success is True


def test_legacy_executor_without_signature_method_still_runs() -> None:
    class LegacyExecutor:
        def __init__(self) -> None:
            self.execute_count = 0

        async def execute_many(
            self,
            tool_calls,
            *,
            context,
        ):
            from app.agent_graph.schemas.tool_schema import (
                ToolResult,
                ToolTraceEntry,
            )
            from app.tools.tool_executor import (
                ToolExecutionOutcome,
            )

            outcomes = []

            for tool_call in tool_calls:
                self.execute_count += 1

                result = ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=tool_call.tool_name,
                    success=True,
                    output={"amount": "180000"},
                    duration_ms=0,
                )
                trace = ToolTraceEntry(
                    tool_call_id=tool_call.tool_call_id,
                    tool_name=tool_call.tool_name,
                    status="succeeded",
                    arguments_summary={},
                    output_summary={},
                    duration_ms=0,
                )
                outcomes.append(
                    ToolExecutionOutcome(
                        result=result,
                        trace=trace,
                    )
                )

            return outcomes

    executor = LegacyExecutor()
    planner = SequencePlanner(
        [
            _call_decision(
                "legacy_call",
                180000,
                version=1,
            ),
            _finish_decision(version=2),
        ]
    )

    loop = AgentToolLoop(
        planner=planner,
        executor=executor,  # type: ignore[arg-type]
        limits=AgentLimits(
            max_agent_rounds=3,
            max_total_tool_calls=1,
        ),
    )

    result = asyncio.run(
        loop.run(
            request=_request(tool_budget=1),
            execution_context=_context(tool_budget=1),
        )
    )

    assert result.status == "completed"
    assert executor.execute_count == 1
    assert result.total_tool_calls == 1
    assert result.reused_tool_call_count == 0
