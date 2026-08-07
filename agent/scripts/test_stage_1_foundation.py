from __future__ import annotations

import json
import uuid

import httpx

from app.rag.rag_audit import RagCitationAuditor
from app.tools.tool_audit import ToolCallAuditor


BASE_URL = "http://127.0.0.1:8000"


def print_json(title: str, data: dict) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def request(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
    timeout: int = 120,
) -> dict:
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        response = client.request(
            method,
            f"{BASE_URL}{path}",
            json=json_body,
            params=params,
        )

    print(method, path, "status_code:", response.status_code)

    if response.status_code != 200:
        print(response.text)

    response.raise_for_status()
    return response.json()


def test_local_rag_audit() -> None:
    auditor = RagCitationAuditor()

    ok_result = auditor.audit(
        answer="寿险缺口公式如下 [1]。",
        executed_tools=[
            {
                "tool_name": "search_knowledge_base",
                "ok": True,
                "result": {
                    "citations": [
                        {
                            "citation_id": 1,
                            "document_id": "doc_001",
                        }
                    ]
                },
            }
        ],
    )

    assert ok_result.citation_consistent is True

    mismatch_result = auditor.audit(
        answer="寿险缺口公式如下 [1]。",
        executed_tools=[],
    )

    assert mismatch_result.citation_consistent is False
    assert mismatch_result.issue == "answer_contains_citation_but_rag_tool_not_called"

    print("local_rag_audit_ok")


def test_local_tool_audit() -> None:
    auditor = ToolCallAuditor()

    result = auditor.audit(
        executed_tools=[
            {
                "tool_name": "yearly_expense_to_monthly",
                "ok": True,
                "result": {
                    "monthly_necessary_expense": "15000",
                },
            },
            {
                "tool_name": "life_insurance_gap",
                "ok": False,
                "error": "缺少必要参数",
            },
        ]
    )

    assert result.total_tool_calls == 2
    assert result.successful_tool_calls == 1
    assert result.failed_tool_calls == 1
    assert result.has_calculation_tool is True
    assert "life_insurance_gap" in result.failed_tool_names

    print("local_tool_audit_ok")


def test_memory_management_api(user_id: str) -> None:
    tenant_id = "default"

    delete_data = request(
        "DELETE",
        "/api/memory/facts",
        params={
            "user_id": user_id,
            "tenant_id": tenant_id,
        },
    )

    assert delete_data["ok"] is True

    upsert_data = request(
        "POST",
        "/api/memory/facts",
        json_body={
            "user_id": user_id,
            "tenant_id": tenant_id,
            "fact_type": "family_finance",
            "fact_key": "annual_necessary_expense",
            "fact_value": {
                "amount": 180000,
                "currency": "CNY",
                "unit": "year",
                "original_text": "家庭年度必要支出18万元",
            },
            "confidence": 0.99,
            "source_thread_id": "stage_1_memory_api",
        },
    )

    assert upsert_data["ok"] is True
    assert upsert_data["fact"]["fact_value"]["amount"] == 180000

    list_data = request(
        "GET",
        "/api/memory/facts",
        params={
            "user_id": user_id,
            "tenant_id": tenant_id,
        },
    )

    assert list_data["ok"] is True
    assert list_data["count"] >= 1

    delete_one_data = request(
        "DELETE",
        "/api/memory/facts/family_finance/annual_necessary_expense",
        params={
            "user_id": user_id,
            "tenant_id": tenant_id,
        },
    )

    assert delete_one_data["ok"] is True
    assert delete_one_data["deleted_count"] >= 1

    print("memory_management_api_ok")


def test_chat_long_memory_and_audit(user_id: str) -> None:
    tenant_id = "default"
    thread_id_1 = f"stage_1_thread_A_{uuid.uuid4()}"
    thread_id_2 = f"stage_1_thread_B_{uuid.uuid4()}"

    first_data = request(
        "POST",
        "/api/chat",
        json_body={
            "user_id": user_id,
            "tenant_id": tenant_id,
            "thread_id": thread_id_1,
            "message": (
                "请记录一下我的家庭信息：夫妻两人分别35岁和33岁，孩子6岁，"
                "家庭年度必要支出18万元，房贷余额80万元，已有可用资产25万元，"
                "丈夫定寿30万元，妻子无寿险，风险偏好稳健。"
            ),
        },
    )

    print_json("first_chat_response", first_data)

    first_usage = first_data.get("usage", {})
    first_long_memory = first_usage.get("long_memory", {})
    first_rag_audit = first_usage.get("rag_audit", {})
    first_tool_audit = first_usage.get("tool_audit", {})

    assert first_long_memory.get("enabled") is True
    assert first_long_memory.get("direct_return") is True
    assert first_long_memory.get("saved_count", 0) >= 8
    assert first_rag_audit.get("citation_consistent") is True
    assert first_tool_audit.get("total_tool_calls") == 0
    assert "已记录" in first_data.get("answer", "")

    second_data = request(
        "POST",
        "/api/chat",
        json_body={
            "user_id": user_id,
            "tenant_id": tenant_id,
            "thread_id": thread_id_2,
            "message": "我刚才记录的家庭年度必要支出是多少？另外房贷余额是多少？",
        },
    )

    print_json("second_chat_response", second_data)

    second_usage = second_data.get("usage", {})
    second_long_memory = second_usage.get("long_memory", {})
    second_rag_audit = second_usage.get("rag_audit", {})
    second_tool_audit = second_usage.get("tool_audit", {})

    assert second_long_memory.get("enabled") is True
    assert second_long_memory.get("loaded") is True
    assert second_long_memory.get("saved_count") == 0
    assert second_long_memory.get("direct_return") is False
    assert "citation_consistent" in second_rag_audit
    assert "total_tool_calls" in second_tool_audit

    answer = second_data.get("answer", "")

    assert "18" in answer or "180000" in answer
    assert "80" in answer or "800000" in answer

    print("chat_long_memory_and_audit_ok")


def main() -> None:
    user_id = f"stage_1_user_{uuid.uuid4()}"

    print()
    print("=" * 80)
    print("Stage 1 Foundation Test Started")
    print("=" * 80)

    health = request("GET", "/health")
    assert health["status"] == "ok"

    memory_health = request("GET", "/health/memory")
    assert memory_health["status"] == "ok"

    test_local_rag_audit()
    test_local_tool_audit()
    test_memory_management_api(user_id)
    test_chat_long_memory_and_audit(user_id)

    print()
    print("=" * 80)
    print("Stage 1 Foundation Test Passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
