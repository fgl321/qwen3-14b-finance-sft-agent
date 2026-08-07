import asyncio
import json
import traceback

from app.agent_graph.service import FinanceAgentGraphService


def _pick_debug_fields(result: dict) -> dict:
    """
    只打印关键字段，避免终端输出太乱。
    """
    usage = result.get("usage") or {}

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
        "usage_keys": list(usage.keys()),
        "error": result.get("error"),
        "final_answer": result.get("final_answer"),
    }


async def main() -> None:
    service = FinanceAgentGraphService()

    result = await service.run(
        user_message="什么是紧急备用金？",
        user_id="stage4_real_user_001",
        thread_id="stage4_real_thread_001",
        request_id="stage4-graph-service-real-001",
        tenant_id="tenant_001",
        knowledge_base_id="kb_finance_basic",
        history_messages=[],
    )

    debug_result = _pick_debug_fields(result)

    print("=" * 80)
    print("Stage 4.1 Graph Service 真实调用测试")
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
