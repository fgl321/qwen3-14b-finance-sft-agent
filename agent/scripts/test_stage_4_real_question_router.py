from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.agent_graph.deepseek_router_adapter import (
    build_hybrid_question_router,
)
from app.core.config import get_settings
from app.llm.deepseek_client import DeepSeekClient


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "明确概念问题",
        "question": "什么是紧急备用金？",
        "expected_router": "hard_rule",
        "expected_capabilities": [
            "general_explanation",
        ],
    },
    {
        "name": "模糊语义问题",
        "question": (
            "最近总觉得钱放着也不安心，"
            "这种情况一般先考虑什么？"
        ),
        "expected_router": "llm_semantic_router",
        "expected_capabilities": None,
    },
    {
        "name": "明确多能力问题",
        "question": (
            "请根据我上传的保单，"
            "计算寿险保障缺口，"
            "并给出调整方案。"
        ),
        "expected_router": "hard_rule",
        "expected_capabilities": [
            "knowledge_retrieval",
            "financial_calculation",
            "complex_reasoning",
        ],
    },
]


async def run_case(
    *,
    router: Any,
    index: int,
    case: dict[str, Any],
) -> None:
    question = str(case["question"])
    expected_router = str(
        case["expected_router"]
    )
    expected_capabilities = case.get(
        "expected_capabilities"
    )

    result = await router.route(question)
    result_dict = result.to_dict()

    print()
    print("=" * 80)
    print(f"测试 {index}：{case['name']}")
    print("-" * 80)
    print(f"用户问题：{question}")
    print()
    print("路由结果：")
    print(
        json.dumps(
            result_dict,
            ensure_ascii=False,
            indent=2,
        )
    )

    if result.router != expected_router:
        raise AssertionError(
            "路由来源不符合预期："
            f"expected={expected_router}, "
            f"actual={result.router}"
        )

    if result.used_fallback:
        validation_error = ""

        if result.llm_result is not None:
            validation_error = (
                result.llm_result.validation_error
            )

        raise AssertionError(
            "真实路由调用触发了 fallback："
            f"reason={result.reason}, "
            f"validation_error={validation_error}"
        )

    actual_capabilities = [
        capability.value
        for capability in result.capabilities
    ]

    if (
        expected_capabilities is not None
        and actual_capabilities
        != expected_capabilities
    ):
        raise AssertionError(
            "能力列表不符合预期："
            f"expected={expected_capabilities}, "
            f"actual={actual_capabilities}"
        )

    print()
    print("该测试通过。")


async def main() -> None:
    settings = get_settings()

    if not settings.deepseek_api_key:
        raise RuntimeError(
            "没有读取到 DEEPSEEK_API_KEY。"
            "请检查项目根目录下的 .env 文件。"
        )

    print("=" * 80)
    print("Stage 4.1 真实 DeepSeek 问题路由测试")
    print("=" * 80)
    print(f"模型：{settings.deepseek_model}")
    print(f"接口地址：{settings.deepseek_base_url}")

    llm_client = DeepSeekClient(settings)

    try:
        router = build_hybrid_question_router(
            llm_client=llm_client,
            timeout_seconds=30.0,
            max_completion_tokens=512,
        )

        for index, case in enumerate(
            TEST_CASES,
            start=1,
        ):
            await run_case(
                router=router,
                index=index,
                case=case,
            )

        print()
        print("=" * 80)
        print(
            "Stage 4.1 真实问题路由测试全部通过"
        )
        print("=" * 80)

    finally:
        await llm_client.close()


if __name__ == "__main__":
    asyncio.run(main())
