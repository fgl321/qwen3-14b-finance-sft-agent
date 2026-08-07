from __future__ import annotations

import asyncio
import json

from app.agent_graph.llm_plan_reviewer import (
    LLMPlanReviewer,
    PlanReviewRequest,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
    ToolCallRequest,
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
        registry = build_production_tool_registry()

        reviewer = LLMPlanReviewer(
            llm_client=client,
            registry=registry,
        )

        # 这是一个故意构造的错误并行计划：
        # 第二个工具依赖第一个工具的计算结果，
        # 因此不能在同一轮并行。
        decision = PlannerDecision(
            action="call_tools",
            tool_calls=[
                ToolCallRequest(
                    tool_call_id="call_yearly",
                    tool_name=(
                        "yearly_expense_to_monthly"
                    ),
                    arguments={
                        "yearly_necessary_expense": 180000
                    },
                ),
                ToolCallRequest(
                    tool_call_id="call_emergency",
                    tool_name="emergency_fund_range",
                    arguments={
                        "monthly_necessary_expense": 15000,
                        "min_months": 3,
                        "max_months": 6,
                    },
                ),
            ],
            decision_reason=(
                "并行执行年度换算和备用金计算。"
            ),
            confidence="medium",
            needs_review=True,
            plan_version=1,
        )

        result = await reviewer.review(
            PlanReviewRequest(
                request_id="real_reviewer_request",
                run_id="real_reviewer_run",
                user_message=(
                    "家庭年度必要支出18万元，"
                    "请计算3到6个月紧急备用金。"
                ),
                decision=decision,
                route_context={
                    "complexity": "medium",
                    "risk_level": "low",
                },
            )
        )

        print(
            json.dumps(
                result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )

        if result.decision.verdict != "revise":
            raise AssertionError(
                "Reviewer 没有识别出工具依赖错误："
                f"{result.decision.verdict}"
            )

        print(
            "\nStage 4.2.5 real reviewer test passed."
        )

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
