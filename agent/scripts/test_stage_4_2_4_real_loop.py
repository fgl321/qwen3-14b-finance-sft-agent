from __future__ import annotations

import asyncio
import json

from app.agent_graph.agent_loop import AgentToolLoop
from app.agent_graph.llm_task_planner import (
    LLMTaskPlanner,
    PlannerRequest,
)
from app.core.config import get_settings
from app.llm.deepseek_client import DeepSeekClient
from app.tools.runtime_registry import (
    build_production_tool_registry,
)
from app.tools.tool_executor import (
    ProductionToolExecutor,
    ToolExecutionContext,
)


async def main() -> None:
    settings = get_settings()

    client = DeepSeekClient(settings)

    try:
        registry = build_production_tool_registry()

        planner = LLMTaskPlanner(
            llm_client=client,
            registry=registry,
            max_completion_tokens=1024,
            max_protocol_repairs=1,
        )

        executor = ProductionToolExecutor(
            registry=registry
        )

        loop = AgentToolLoop(
            planner=planner,
            executor=executor,
        )

        result = await loop.run(
            request=PlannerRequest(
                request_id="real_loop_request",
                run_id="real_loop_run",
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
                remaining_tool_calls=12,
            ),
            execution_context=ToolExecutionContext(
                request_id="real_loop_request",
                run_id="real_loop_run",
                tenant_id="default",
                user_id="real_loop_user",
                role="user",
                allowed_tool_groups=frozenset(
                    {"financial_calculation"}
                ),
                allow_side_effects=False,
                remaining_tool_calls=12,
            ),
        )

        print(
            json.dumps(
                result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )

        if result.status != "completed":
            raise AssertionError(
                f"真实工具循环没有完成：{result.status}，"
                f"finish_reason={result.finish_reason}"
            )

        successful_results = [
            item
            for item in result.tool_results
            if item.success
        ]

        tool_names = [
            item.tool_name
            for item in successful_results
        ]

        if tool_names != [
            "yearly_expense_to_monthly",
            "emergency_fund_range",
        ]:
            raise AssertionError(
                f"实际工具调用顺序不正确：{tool_names}"
            )

        monthly_result = successful_results[0].output

        if (
            monthly_result[
                "monthly_necessary_expense"
            ]
            != "15000.00"
        ):
            raise AssertionError(
                "年度支出换算结果不正确。"
            )

        emergency_result = successful_results[1].output

        if emergency_result["min_amount"] != "45000.00":
            raise AssertionError(
                "紧急备用金最小值不正确。"
            )

        if emergency_result["max_amount"] != "90000.00":
            raise AssertionError(
                "紧急备用金最大值不正确。"
            )

        print(
            "\nStage 4.2.4 real agent tool loop passed."
        )

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
