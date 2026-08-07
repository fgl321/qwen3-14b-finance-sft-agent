from __future__ import annotations

import asyncio
from decimal import Decimal

from app.agent_graph.schemas.planner_schema import ToolCallRequest
from app.tools.runtime_registry import build_production_tool_registry
from app.tools.tool_executor import (
    ProductionToolExecutor,
    ToolExecutionContext,
)


def test_empty_allowed_names_does_not_block_allowed_group() -> None:
    registry = build_production_tool_registry()
    executor = ProductionToolExecutor(registry=registry)

    outcome = asyncio.run(
        executor.execute_one(
            ToolCallRequest(
                tool_call_id="call_yearly_001",
                tool_name="yearly_expense_to_monthly",
                arguments={
                    "yearly_necessary_expense": 180000,
                },
            ),
            context=ToolExecutionContext(
                request_id="request_001",
                run_id="run_001",
                allowed_tool_names=frozenset(),
                allowed_tool_groups=frozenset(
                    {"financial_calculation"}
                ),
            ),
        )
    )

    assert outcome.result.success is True
    # 工具输出采用可序列化 Decimal 字符串；
    # "15000"、"15000.0"、"15000.00" 数值等价。
    assert Decimal(
        str(
            outcome.result.output[
                "monthly_necessary_expense"
            ]
        )
    ) == Decimal("15000")


def test_nonempty_allowed_names_is_still_enforced() -> None:
    registry = build_production_tool_registry()
    executor = ProductionToolExecutor(registry=registry)

    outcome = asyncio.run(
        executor.execute_one(
            ToolCallRequest(
                tool_call_id="call_yearly_002",
                tool_name="yearly_expense_to_monthly",
                arguments={
                    "yearly_necessary_expense": 180000,
                },
            ),
            context=ToolExecutionContext(
                request_id="request_002",
                run_id="run_002",
                allowed_tool_names=frozenset(
                    {"emergency_fund_range"}
                ),
                allowed_tool_groups=frozenset(
                    {"financial_calculation"}
                ),
            ),
        )
    )

    assert outcome.result.success is False
    assert outcome.result.error is not None
    assert outcome.result.error.code == "PERMISSION_DENIED"
