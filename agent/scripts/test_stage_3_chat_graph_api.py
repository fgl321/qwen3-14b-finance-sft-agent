import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

import httpx


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# 避免 Windows 系统代理污染 127.0.0.1 请求
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")


NO_EVIDENCE_PATTERNS = (
    "当前知识库中没有找到足够依据",
    "知识库中没有找到足够依据",
    "不能基于知识库给出确定回答",
)


async def main() -> None:
    print("========== Stage 3 Chat Graph API Test ==========")

    request_id = f"stage3-chat-graph-api-{uuid4()}"

    payload = {
        "request_id": request_id,
        "user_id": "stage3_graph_api_user_001",
        "thread_id": "stage3_graph_api_thread_001",
        "tenant_id": "tenant_001",
        "knowledge_base_id": "kb_finance_basic",
        "message": "请用一句话解释什么是紧急备用金，不要推荐具体投资产品。",
        "history_messages": [],
    }

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        timeout=120.0,
        trust_env=False,
    ) as client:
        response = await client.post(
            "/api/chat/graph",
            json=payload,
        )

    print("\n========== status_code ==========")
    print(response.status_code)

    print("\n========== response text ==========")
    print(response.text)

    assert response.status_code == 200

    data = response.json()

    print("\n========== response keys ==========")
    print(sorted(data.keys()))

    print("\n========== fallback_used ==========")
    print(data.get("fallback_used"))

    print("\n========== answer ==========")
    print(data.get("answer"))

    print("\n========== quality_gate ==========")
    print(data.get("quality_gate"))

    answer = data.get("answer") or ""

    assert data.get("request_id") == request_id
    assert answer.strip(), "answer 不能为空"
    assert "紧急备用金" in answer, "回答中应该解释紧急备用金"

    for pattern in NO_EVIDENCE_PATTERNS:
        assert pattern not in answer, (
            "Graph API 没有正确使用 LangGraph 质量门控兜底。"
        )

    print("\nStage 3 Chat Graph API Test Passed")


if __name__ == "__main__":
    asyncio.run(main())
