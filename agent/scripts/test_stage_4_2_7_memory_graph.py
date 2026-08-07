from __future__ import annotations

import asyncio
import json

from app.agent_graph.production_service import (
    ProductionFinanceGraphService,
)
from app.core.config import get_settings
from app.llm.deepseek_client import (
    DeepSeekClient,
)


async def main() -> None:
    settings = get_settings()

    client = DeepSeekClient(settings)

    try:
        service = (
            ProductionFinanceGraphService
            .from_llm_client(
                llm_client=client
            )
        )

        user_message = (
            "我的家庭年度必要支出是18万元，"
            "请计算3到6个月的紧急备用金。"
        )

        result = await service.run(
            request_id=(
                "stage_4_2_7_request"
            ),
            run_id="stage_4_2_7_run",
            user_message=user_message,
            user_id="stage_4_2_7_user",
            thread_id=(
                "stage_4_2_7_thread"
            ),
            tenant_id="default",
            route_context={
                "complexity": "medium",
                "risk_level": "low",
            },
            allowed_tool_groups=[
                "financial_calculation"
            ],
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

        if result.get("status") != "completed":
            raise AssertionError(
                "生产主图没有完成："
                f"{result.get('status')}，"
                f"{result.get('finish_reason')}"
            )

        if (
            result.get("finish_reason")
            != "output_guard_passed"
        ):
            raise AssertionError(
                "最终回答没有通过 Output Guard。"
            )

        answer = str(
            result.get("final_answer") or ""
        )

        if not any(
            value in answer
            for value in (
                "4.5万",
                "45000",
                "45,000",
            )
        ):
            raise AssertionError(
                "最终回答缺少最小备用金金额。"
            )

        if not any(
            value in answer
            for value in (
                "9万",
                "90000",
                "90,000",
            )
        ):
            raise AssertionError(
                "最终回答缺少最大备用金金额。"
            )

        checkpoint_state = (
            await service
            .get_checkpoint_state(
                user_id=(
                    "stage_4_2_7_user"
                ),
                thread_id=(
                    "stage_4_2_7_thread"
                ),
                tenant_id="default",
            )
        )

        if (
            checkpoint_state.get(
                "request_id"
            )
            != "stage_4_2_7_request"
        ):
            raise AssertionError(
                "Checkpointer 没有保存当前请求状态。"
            )

        if (
            checkpoint_state.get(
                "final_answer"
            )
            != result.get("final_answer")
        ):
            raise AssertionError(
                "Checkpoint 最终回答不一致。"
            )

        print(
            "\nStage 4.2.7A production "
            "memory graph passed."
        )

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
