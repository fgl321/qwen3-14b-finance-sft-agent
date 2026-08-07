import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# 避免 Windows 系统代理污染本地 Redis / Qdrant / Postgres / FastAPI 访问
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")


from app.agent_graph.graph import build_finance_agent_graph  # noqa: E402


async def main() -> None:
    print("========== Stage 3 LangGraph Minimal Test ==========")

    graph = build_finance_agent_graph()

    request_id = f"stage3-minimal-{uuid4()}"

    result = await graph.ainvoke(
        {
            "request_id": request_id,
            "user_id": "stage3_user_001",
            "thread_id": "stage3_thread_001",
            "tenant_id": "tenant_001",
            "knowledge_base_id": "kb_finance_basic",
            "user_message": "请用一句话解释什么是紧急备用金，不要推荐具体投资产品。",
            "history_messages": [],
        }
    )

    print("\n========== graph result keys ==========")
    print(sorted(result.keys()))

    print("\n========== final_answer ==========")
    print(result.get("final_answer"))

    print("\n========== usage ==========")
    print(result.get("usage"))

    print("\n========== executed_tools ==========")
    print(result.get("executed_tools"))

    if result.get("error"):
        print("\n========== error ==========")
        print(result["error"])
        raise RuntimeError(result["error"])

    assert result.get("request_id") == request_id
    assert result.get("final_answer"), "final_answer 不能为空"
    assert isinstance(result.get("usage"), dict), "usage 必须是 dict"

    print("\nStage 3 LangGraph Minimal Test Passed")


if __name__ == "__main__":
    asyncio.run(main())
