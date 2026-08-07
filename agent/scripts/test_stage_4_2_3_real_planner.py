from __future__ import annotations

import asyncio
import json

from app.agent_graph.llm_task_planner import (
    LLMTaskPlanner,
    PlannerRequest,
)
from app.core.config import get_settings
from app.llm.deepseek_client import DeepSeekClient
from app.tools.runtime_registry import (
    build_production_tool_registry,
)


async def main() -> None:
    settings = get_settings()

    client = DeepSeekClient(settings)

    try:
        planner = LLMTaskPlanner(
            llm_client=client,
            registry=build_production_tool_registry(),
            max_completion_tokens=1024,
            max_protocol_repairs=1,
        )

        result = await planner.plan(
            PlannerRequest(
                request_id="real_planner_request",
                run_id="real_planner_run",
                user_message=(
                    "我的家庭年度必要支出是18万元，"
                    "请帮我计算3到6个月的紧急备用金。"
                ),
                route_context={
                    "capabilities": [
                        "financial_calculation"
                    ],
                    "complexity": "medium",
                    "risk_level": "low",
                    "allowed_tool_groups": [
                        "financial_calculation"
                    ],
                },
                allowed_tool_groups=frozenset(
                    {"financial_calculation"}
                ),
                agent_round=1,
                remaining_tool_calls=12,
            )
        )

        print(
            json.dumps(
                result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )

        if result.decision.action != "call_tools":
            raise AssertionError(
                "真实 Planner 第一轮没有选择 call_tools。"
            )

        selected_names = [
            call.tool_name
            for call in result.decision.tool_calls
        ]

        if "yearly_expense_to_monthly" not in selected_names:
            raise AssertionError(
                "真实 Planner 第一轮没有先调用 "
                "yearly_expense_to_monthly。"
            )

        if "emergency_fund_range" in selected_names:
            raise AssertionError(
                "真实 Planner 在尚未得到月度支出前，"
                "提前调用了 emergency_fund_range。"
            )

        print(
            "\nStage 4.2.3 real planner first-round test passed."
        )

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
