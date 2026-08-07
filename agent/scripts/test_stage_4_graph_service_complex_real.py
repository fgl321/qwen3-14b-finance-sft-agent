import asyncio
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent_graph.service import FinanceAgentGraphService


def _pick_debug_fields(result: dict) -> dict:
    """
    只打印关键字段，避免终端输出太乱。
    """
    usage = result.get("usage") or {}
    agent_result = result.get("agent_result") or {}

    return {
        "request_id": result.get("request_id"),
        "question_capabilities": result.get("question_capabilities"),
        "question_router": result.get("question_router"),
        "question_router_confidence": result.get(
            "question_router_confidence"
        ),
        "question_router_reason": result.get(
            "question_router_reason"
        ),
        "question_router_used_fallback": result.get(
            "question_router_used_fallback"
        ),
        "question_router_matched_rules": result.get(
            "question_router_matched_rules"
        ),
        "execution_plan": result.get("execution_plan"),
        "finish_reason": result.get("finish_reason"),
        "fallback_used": result.get("fallback_used"),
        "needs_general_finance_fallback": result.get(
            "needs_general_finance_fallback"
        ),
        "executed_tools": result.get("executed_tools"),
        "usage_keys": list(usage.keys()),
        "agent_result_keys": list(agent_result.keys()),
        "quality_gate": result.get("quality_gate"),
        "error": result.get("error"),
        "final_answer": result.get("final_answer"),
    }


async def main() -> None:
    service = FinanceAgentGraphService()

    user_message = (
        "我家年度必要支出是18万元，"
        "请帮我换算成月度必要支出，"
        "并计算3到6个月紧急备用金范围。"
    )

    result = await service.run(
        user_message=user_message,
        user_id="stage4_complex_user_001",
        thread_id="stage4_complex_thread_001",
        request_id="stage4-graph-service-complex-real-001",
        tenant_id="tenant_001",
        knowledge_base_id="kb_finance_basic",
        history_messages=[],
    )

    debug_result = _pick_debug_fields(result)

    print("=" * 80)
    print("Stage 4.1 Graph Service 复杂问题真实调用测试")
    print("=" * 80)
    print("用户问题：")
    print(user_message)

    print("=" * 80)
    print("关键返回字段")
    print("=" * 80)

    print(
        json.dumps(
            debug_result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("=" * 80)
    print("字段检查")
    print("=" * 80)

    required_fields = [
        "question_capabilities",
        "question_router",
        "question_router_confidence",
        "question_router_reason",
        "execution_plan",
        "final_answer",
        "finish_reason",
    ]

    for field in required_fields:
        value = result.get(field)
        status = "OK" if value else "MISSING"
        print(f"{field}: {status}")

    print("=" * 80)
    print("复杂路径检查")
    print("=" * 80)

    execution_plan = result.get("execution_plan") or []
    capabilities = result.get("question_capabilities") or []

    print(
        "是否进入 finance_agent 执行计划：",
        "YES" if "finance_agent" in execution_plan else "NO",
    )

    print(
        "是否包含 financial_calculation 能力：",
        "YES" if "financial_calculation" in capabilities else "NO",
    )

    if result.get("error"):
        print("=" * 80)
        print("检测到 error 字段")
        print("=" * 80)
        print(result["error"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        print("=" * 80)
        print("脚本执行失败")
        print("=" * 80)
        traceback.print_exc()
