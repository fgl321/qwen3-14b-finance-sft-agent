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


def extract_answer(data: dict) -> str:
    """
    兼容不同接口返回结构。

    可能是：
    1. {"answer": "..."}
    2. {"final_answer": "..."}
    3. {"data": {"answer": "..."}}
    4. {"data": {"final_answer": "..."}}
    """
    if not isinstance(data, dict):
        return ""

    if isinstance(data.get("answer"), str):
        return data["answer"]

    if isinstance(data.get("final_answer"), str):
        return data["final_answer"]

    inner_data = data.get("data")

    if isinstance(inner_data, dict):
        if isinstance(inner_data.get("answer"), str):
            return inner_data["answer"]

        if isinstance(inner_data.get("final_answer"), str):
            return inner_data["final_answer"]

    return ""


async def post_with_compatible_payload(
    client: httpx.AsyncClient,
    path: str,
    payload: dict,
) -> httpx.Response:
    """
    兼容旧 /api/chat 可能使用 message 或 user_message 字段的问题。

    先用 message 请求。
    如果接口返回 422，再换成 user_message 请求一次。
    """
    response = await client.post(path, json=payload)

    if response.status_code != 422:
        return response

    alt_payload = dict(payload)
    alt_payload["user_message"] = alt_payload.pop("message")

    return await client.post(path, json=alt_payload)


async def main() -> None:
    print("========== Stage 3 API Regression Test ==========")

    base_payload = {
        "request_id": f"stage3-api-regression-{uuid4()}",
        "user_id": "stage3_api_regression_user_001",
        "thread_id": "stage3_api_regression_thread_001",
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
        legacy_response = await post_with_compatible_payload(
            client=client,
            path="/api/chat",
            payload=base_payload,
        )

        graph_response = await post_with_compatible_payload(
            client=client,
            path="/api/chat/graph",
            payload=base_payload,
        )

    print("\n========== legacy /api/chat status_code ==========")
    print(legacy_response.status_code)

    print("\n========== legacy /api/chat response text ==========")
    print(legacy_response.text)

    print("\n========== graph /api/chat/graph status_code ==========")
    print(graph_response.status_code)

    print("\n========== graph /api/chat/graph response text ==========")
    print(graph_response.text)

    assert legacy_response.status_code == 200, (
        "旧 /api/chat 接口异常，说明新增 Graph 代码可能影响了旧接口。"
    )

    assert graph_response.status_code == 200, (
        "新 /api/chat/graph 接口异常。"
    )

    legacy_data = legacy_response.json()
    graph_data = graph_response.json()

    legacy_answer = extract_answer(legacy_data)
    graph_answer = extract_answer(graph_data)

    print("\n========== legacy answer ==========")
    print(legacy_answer)

    print("\n========== graph answer ==========")
    print(graph_answer)

    assert legacy_answer.strip(), "旧 /api/chat answer 不能为空"
    assert graph_answer.strip(), "新 /api/chat/graph answer 不能为空"

    assert graph_data.get("fallback_used") is True, (
        "新 /api/chat/graph 应该触发 LangGraph 质量门控兜底。"
    )

    for pattern in NO_EVIDENCE_PATTERNS:
        assert pattern not in graph_answer, (
            "新 /api/chat/graph 没有正确使用质量门控兜底。"
        )

    print("\nStage 3 API Regression Test Passed")


if __name__ == "__main__":
    asyncio.run(main())
