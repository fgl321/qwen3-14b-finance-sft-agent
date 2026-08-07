import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# 避免 Windows 系统代理影响本地服务
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")


from app.agent_graph.service import get_finance_agent_graph_service  # noqa: E402


NO_EVIDENCE_PATTERNS = (
    "当前知识库中没有找到足够依据",
    "知识库中没有找到足够依据",
    "不能基于知识库给出确定回答",
)


async def main() -> None:
    print("========== Stage 3 Graph Service Test ==========")

    service = get_finance_agent_graph_service()

    request_id = f"stage3-graph-service-{uuid4()}"

    result = await service.run(
        request_id=request_id,
        user_id="stage3_user_001",
        thread_id="stage3_thread_001",
        tenant_id="tenant_001",
        knowledge_base_id="kb_finance_basic",
        user_message="请用一句话解释什么是紧急备用金，不要推荐具体投资产品。",
        history_messages=[],
    )

    print("\n========== result keys ==========")
    print(sorted(result.keys()))

    print("\n========== fallback_used ==========")
    print(result.get("fallback_used"))

    print("\n========== final_answer ==========")
    print(result.get("final_answer"))

    print("\n========== quality_gate ==========")
    print(result.get("quality_gate"))

    if result.get("error"):
        print("\n========== error ==========")
        print(result["error"])
        raise RuntimeError(result["error"])

    final_answer = result.get("final_answer") or ""

    assert result.get("request_id") == request_id
    assert final_answer.strip(), "final_answer 不能为空"
    assert "紧急备用金" in final_answer, "回答中应该解释紧急备用金"

    for pattern in NO_EVIDENCE_PATTERNS:
        assert pattern not in final_answer, (
            "Graph Service 没有正确使用 LangGraph 质量门控兜底。"
        )

    print("\nStage 3 Graph Service Test Passed")


if __name__ == "__main__":
    asyncio.run(main())
