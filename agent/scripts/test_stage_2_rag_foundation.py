from __future__ import annotations

import json
import uuid

import httpx

from app.rag.rag_audit import RagCitationAuditor
from app.rag.rag_quality_audit import RagQualityAuditor
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


def test_local_rag_quality_audit() -> None:
    executed_tools = [
        {
            "tool_name": "search_knowledge_base",
            "ok": True,
            "result": {
                "answer": "寿险保障缺口公式如下 [1]。",
                "evidence_assessment": {
                    "sufficient": True,
                    "confidence": "high",
                    "reason": "证据包含公式。",
                    "relevant_evidence_numbers": [1],
                    "missing_info": [],
                },
                "citations": [
                    {
                        "citation_id": 1,
                        "document_id": "doc_001",
                        "file_name": "finance.txt",
                        "page_start": None,
                        "page_end": None,
                        "chunk_id": "chunk_001",
                        "score": 92.5,
                        "score_type": "normalized_hybrid_score_0_100",
                        "score_display": "92.50/100",
                        "metadata": {},
                    }
                ],
                "retrieved_count": 1,
            },
        }
    ]

    rag_payload = {
        "used": True,
        "sufficient": True,
        "retrieved_count": 1,
        "citations": executed_tools[0]["result"]["citations"],
    }

    citation_audit = RagCitationAuditor().audit(
        answer="寿险保障缺口公式如下 [1]。",
        executed_tools=executed_tools,
    )

    assert citation_audit.citation_consistent is True

    quality_audit = RagQualityAuditor().audit(
        answer="寿险保障缺口公式如下 [1]。",
        executed_tools=executed_tools,
        rag_payload=rag_payload,
    )

    assert quality_audit.quality_level == "grounded"
    assert quality_audit.citation_consistent is True

    print("local_rag_quality_audit_grounded_ok")


def test_local_rag_quality_warning() -> None:
    executed_tools: list[dict] = []

    rag_payload = {
        "used": False,
        "sufficient": None,
        "retrieved_count": None,
        "citations": [],
    }

    quality_audit = RagQualityAuditor().audit(
        answer="根据知识库，寿险保障缺口公式如下 [1]。",
        executed_tools=executed_tools,
        rag_payload=rag_payload,
    )

    assert quality_audit.quality_level == "warning"
    assert len(quality_audit.issues) >= 1

    print("local_rag_quality_audit_warning_ok")


def test_chat_has_rag_quality_audit(user_id: str) -> None:
    data = request(
        "POST",
        "/api/chat",
        json_body={
            "user_id": user_id,
            "tenant_id": "default",
            "thread_id": f"stage_2_rag_thread_{uuid.uuid4()}",
            "knowledge_base_id": "kb_finance_basic",
            "message": "寿险保障缺口的基础公式是什么？请尽量基于知识库回答。",
        },
    )

    print_json("rag_chat_response", data)

    usage = data.get("usage", {})
    rag_audit = usage.get("rag_audit")
    rag_quality_audit = usage.get("rag_quality_audit")
    tool_audit = usage.get("tool_audit")

    assert rag_audit is not None
    assert rag_quality_audit is not None
    assert tool_audit is not None

    assert "citation_consistent" in rag_audit
    assert "quality_level" in rag_quality_audit
    assert "issues" in rag_quality_audit
    assert "total_tool_calls" in tool_audit

    print("chat_rag_quality_audit_present_ok")

    print()
    print("=" * 80)
    print("RAG Quality Summary")
    print("=" * 80)
    print("rag =", json.dumps(data.get("rag", {}), ensure_ascii=False, indent=2))
    print("rag_audit =", json.dumps(rag_audit, ensure_ascii=False, indent=2))
    print("rag_quality_audit =", json.dumps(rag_quality_audit, ensure_ascii=False, indent=2))
    print("tool_audit =", json.dumps(tool_audit, ensure_ascii=False, indent=2))


def test_stage_1_still_ok(user_id: str) -> None:
    thread_id_1 = f"stage_2_memory_thread_A_{uuid.uuid4()}"
    thread_id_2 = f"stage_2_memory_thread_B_{uuid.uuid4()}"

    first_data = request(
        "POST",
        "/api/chat",
        json_body={
            "user_id": user_id,
            "tenant_id": "default",
            "thread_id": thread_id_1,
            "message": (
                "请记录一下我的家庭信息：夫妻两人分别35岁和33岁，孩子6岁，"
                "家庭年度必要支出18万元，房贷余额80万元，已有可用资产25万元，"
                "丈夫定寿30万元，妻子无寿险，风险偏好稳健。"
            ),
        },
    )

    first_usage = first_data.get("usage", {})
    first_long_memory = first_usage.get("long_memory", {})
    first_rag_quality = first_usage.get("rag_quality_audit", {})

    assert first_long_memory.get("direct_return") is True
    assert first_long_memory.get("saved_count", 0) >= 8
    assert first_rag_quality.get("quality_level") == "not_used"
    assert "已记录" in first_data.get("answer", "")

    second_data = request(
        "POST",
        "/api/chat",
        json_body={
            "user_id": user_id,
            "tenant_id": "default",
            "thread_id": thread_id_2,
            "message": "我刚才记录的家庭年度必要支出是多少？另外房贷余额是多少？",
        },
    )

    second_usage = second_data.get("usage", {})
    second_long_memory = second_usage.get("long_memory", {})
    second_rag_quality = second_usage.get("rag_quality_audit", {})

    assert second_long_memory.get("loaded") is True
    assert second_long_memory.get("saved_count") == 0
    assert "quality_level" in second_rag_quality

    answer = second_data.get("answer", "")

    assert "18" in answer or "180000" in answer
    assert "80" in answer or "800000" in answer

    print("stage_1_regression_still_ok")


def main() -> None:
    user_id = f"stage_2_user_{uuid.uuid4()}"

    print()
    print("=" * 80)
    print("Stage 2 RAG Foundation Test Started")
    print("=" * 80)

    health = request("GET", "/health")
    assert health["status"] == "ok"

    memory_health = request("GET", "/health/memory")
    assert memory_health["status"] == "ok"

    test_local_rag_quality_audit()
    test_local_rag_quality_warning()
    test_chat_has_rag_quality_audit(user_id)
    test_stage_1_still_ok(user_id)

    print()
    print("=" * 80)
    print("Stage 2 RAG Foundation Test Passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
