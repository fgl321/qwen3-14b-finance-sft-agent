from __future__ import annotations

import json
import uuid

import httpx


BASE_URL = "http://127.0.0.1:8000"


def chat(
    *,
    client: httpx.Client,
    message: str,
    thread_id: str,
) -> dict:
    response = client.post(
        "/api/chat",
        json={
            "message": message,
            "user_id": "u001",
            "thread_id": thread_id,
            "tenant_id": "tenant_001",
            "knowledge_base_id": "kb_finance_basic",
        },
    )

    print("status_code:", response.status_code)
    response.raise_for_status()

    data = response.json()

    print("\n========== answer ==========")
    print(data["answer"])

    print("\n========== usage.short_memory ==========")
    print(json.dumps(data["usage"].get("short_memory"), ensure_ascii=False, indent=2))

    return data


def main() -> None:
    thread_id = f"memory_case_{uuid.uuid4()}"

    with httpx.Client(
        base_url=BASE_URL,
        timeout=180,
        trust_env=False,
    ) as client:
        health = client.get("/health")
        health.raise_for_status()

        memory_health = client.get("/health/memory")
        print("memory_health:", memory_health.status_code, memory_health.text)
        memory_health.raise_for_status()

        first = chat(
            client=client,
            thread_id=thread_id,
            message=(
                "请记住这个家庭信息：家庭年度必要支出是18万元，"
                "房贷余额80万元，已有可用资产25万元。你先简单回复已记录。"
            ),
        )

        first_memory = first["usage"].get("short_memory") or {}

        if first_memory.get("history_message_count") != 0:
            raise AssertionError("第一次对话不应该有历史消息。")

        second = chat(
            client=client,
            thread_id=thread_id,
            message="刚才我说的家庭年度必要支出是多少？",
        )

        second_memory = second["usage"].get("short_memory") or {}

        if second_memory.get("history_message_count", 0) <= 0:
            raise AssertionError("第二次对话应该读取到历史消息。")

        answer = second["answer"]

        if "18" not in answer and "180000" not in answer and "180,000" not in answer:
            raise AssertionError("第二次回答没有记住年度必要支出 18 万元。")

    print()
    print("=" * 80)
    print("短期记忆接口测试通过")
    print("=" * 80)


if __name__ == "__main__":
    main()
