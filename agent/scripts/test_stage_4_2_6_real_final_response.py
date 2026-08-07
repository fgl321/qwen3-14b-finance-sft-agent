from __future__ import annotations

import asyncio
import json

from app.agent_graph.agent_loop import AgentToolLoop
from app.agent_graph.final_response_pipeline import (
    FinalResponsePipeline,
    FinalResponseRequest,
)
from app.agent_graph.llm_output_guard import (
    LLMOutputGuard,
)
from app.agent_graph.llm_synthesizer import (
    LLMAnswerSynthesizer,
)
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
        )

        executor = ProductionToolExecutor(
            registry=registry
        )

        loop = AgentToolLoop(
            planner=planner,
            executor=executor,
        )

        user_message = (
            "我的家庭年度必要支出是18万元，"
            "请帮我计算3到6个月的紧急备用金。"
        )

        loop_result = await loop.run(
            request=PlannerRequest(
                request_id="real_final_request",
                run_id="real_final_run",
                user_message=user_message,
                route_context={
                    "complexity": "medium",
                    "risk_level": "low",
                },
                allowed_tool_groups=frozenset(
                    {"financial_calculation"}
                ),
                remaining_tool_calls=12,
            ),
            execution_context=ToolExecutionContext(
                request_id="real_final_request",
                run_id="real_final_run",
                user_id="real_final_user",
                role="user",
                allowed_tool_groups=frozenset(
                    {"financial_calculation"}
                ),
                remaining_tool_calls=12,
            ),
        )

        pipeline = FinalResponsePipeline(
            synthesizer=LLMAnswerSynthesizer(
                llm_client=client
            ),
            output_guard=LLMOutputGuard(
                llm_client=client
            ),
        )

        final_result = await pipeline.run(
            FinalResponseRequest(
                request_id="real_final_request",
                run_id="real_final_run",
                user_message=user_message,
                loop_result=loop_result,
            )
        )

        print(
            json.dumps(
                final_result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )

        if final_result.status != "completed":
            raise AssertionError(
                "最终响应流水线没有完成："
                f"{final_result.status}，"
                f"{final_result.finish_reason}"
            )

        min_value_present = any(
            value in final_result.answer
            for value in (
                "4.5万",
                "45000",
                "45,000",
            )
        )

        max_value_present = any(
            value in final_result.answer
            for value in (
                "9万",
                "90000",
                "90,000",
            )
        )

        if not min_value_present:
            raise AssertionError(
                "最终回答没有包含备用金最小值。"
            )

        if not max_value_present:
            raise AssertionError(
                "最终回答没有包含备用金最大值。"
            )

        if (
            final_result.guard is None
            or final_result.guard.verdict != "pass"
        ):
            raise AssertionError(
                "最终回答没有通过 Output Guard。"
            )

        print(
            "\nStage 4.2.6 real final response passed."
        )

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
