from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from fastapi import Request

from app.api.routes.chat_graph_v2 import (
    ProductionChatRequest,
    production_chat_graph,
)
from app.memory.long_term_memory import LongTermFact


class FakeShortMemory:
    max_messages = 12

    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    def get_messages(self, **_: Any):
        return [{"role": "user", "content": "我之前说年度支出18万元"}]

    def save_turn(self, **kwargs: Any):
        self.saved.append(kwargs)
        return 2


class FakeLongMemory:
    def list_facts(self, **_: Any):
        return [
            LongTermFact(
                id=1,
                tenant_id="default",
                user_id="u1",
                fact_type="family_finance",
                fact_key="annual_necessary_expense",
                fact_value={"amount": 180000, "currency": "CNY"},
                confidence=1.0,
                source_thread_id="old",
                created_at="now",
                updated_at="now",
                is_user_confirmed=True,
            )
        ]


class FakeService:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def run(self, **kwargs: Any):
        self.kwargs = kwargs
        return {
            "status": "completed",
            "final_answer": "年度支出是18万元。",
            "idempotency_replayed": False,
        }


def _request(service: FakeService, short: FakeShortMemory, long: FakeLongMemory):
    app = SimpleNamespace(
        state=SimpleNamespace(
            production_graph_service=service,
            short_memory=short,
            personal_long_memory=long,
            deepseek=None,
        )
    )
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat/graph-v2",
            "headers": [],
            "app": app,
        }
    )


def test_production_chat_loads_and_saves_personal_memory() -> None:
    service = FakeService()
    short = FakeShortMemory()
    long = FakeLongMemory()
    payload = ProductionChatRequest(
        request_id="req1",
        user_message="我刚才说的支出是多少？",
        user_id="u1",
        thread_id="t1",
        extract_long_memory=False,
    )
    result = asyncio.run(
        production_chat_graph(payload, _request(service, short, long))
    )
    assert service.kwargs["history_messages"][0]["content"].startswith("我之前")
    assert "annual_necessary_expense" in service.kwargs["context_summary"]
    assert short.saved
    assert result["personal_memory"]["short_memory_saved"] is True
    assert "knowledge_retrieval" in service.kwargs["allowed_tool_groups"]


def test_same_request_id_reuses_memory_snapshot() -> None:
    service = FakeService()
    short = FakeShortMemory()
    long = FakeLongMemory()
    request = _request(service, short, long)
    payload = ProductionChatRequest(
        request_id="stable-request",
        user_message="记住年度支出18万元",
        user_id="u1",
        thread_id="t1",
        extract_long_memory=False,
    )
    asyncio.run(production_chat_graph(payload, request))
    first_history = list(service.kwargs["history_messages"])
    first_context = service.kwargs["context_summary"]

    # 首轮已经写入短期记忆；重试仍应使用首轮快照。
    short.get_messages = lambda **_: [
        {"role": "user", "content": "变化后的历史"},
        {"role": "assistant", "content": "变化后的回答"},
    ]
    result = asyncio.run(production_chat_graph(payload, request))
    assert service.kwargs["history_messages"] == first_history
    assert service.kwargs["context_summary"] == first_context
    assert result["personal_memory"]["request_memory_snapshot_reused"] is True

class FakeRagResult:
    def __init__(self, *, sufficient: bool) -> None:
        self.sufficient = sufficient

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        del mode
        return {
            "query": "私有知识问题",
            "answer": (
                "根据知识库，紧急备用金应覆盖必要支出。[1]"
                if self.sufficient
                else "当前知识库证据不足，因此不能给出确定回答。"
            ),
            "retrieved_chunks": ([{"chunk_id": "chunk-1"}] if self.sufficient else []),
            "retrieved_count": 1 if self.sufficient else 0,
            "evidence_assessment": {
                "sufficient": self.sufficient,
                "confidence": "high",
                "reason": "test",
                "relevant_evidence_numbers": [1] if self.sufficient else [],
                "missing_info": [] if self.sufficient else ["missing"],
            },
            "citations": (
                [
                    {
                        "citation_id": 1,
                        "document_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "file_name": "finance.txt",
                    }
                ]
                if self.sufficient
                else []
            ),
            "usage": {"retrieval": {"retrieved_count": int(self.sufficient)}},
        }


class FakeRagService:
    def __init__(self, *, sufficient: bool) -> None:
        self.sufficient = sufficient
        self.calls = 0

    async def answer(self, **_: Any) -> FakeRagResult:
        self.calls += 1
        return FakeRagResult(sufficient=self.sufficient)


def _request_with_rag(
    service: FakeService,
    short: FakeShortMemory,
    long: FakeLongMemory,
    rag: FakeRagService,
):
    request = _request(service, short, long)
    request.app.state.rag_service = rag
    return request


def test_sufficient_rag_bypasses_graph_and_keeps_citations() -> None:
    service = FakeService()
    short = FakeShortMemory()
    long = FakeLongMemory()
    rag = FakeRagService(sufficient=True)
    request = _request_with_rag(service, short, long, rag)
    payload = ProductionChatRequest(
        request_id="rag-request-1",
        user_message="私有知识库里的紧急备用金规则是什么？",
        user_id="u1",
        thread_id="t1",
        extract_long_memory=False,
        rag_mode="auto",
    )
    result = asyncio.run(production_chat_graph(payload, request))
    assert result["execution_path"] == "kb_direct"
    assert result["finish_reason"] == "rag_direct_answer"
    assert result["rag"]["citations"][0]["document_id"] == "doc-1"
    assert service.kwargs == {}
    assert rag.calls == 1
    assert short.saved

    replay = asyncio.run(production_chat_graph(payload, request))
    assert replay["idempotency_replayed"] is True
    assert replay["run_id"] == result["run_id"]
    assert rag.calls == 1


def test_required_rag_refuses_when_evidence_is_insufficient() -> None:
    service = FakeService()
    short = FakeShortMemory()
    long = FakeLongMemory()
    rag = FakeRagService(sufficient=False)
    request = _request_with_rag(service, short, long, rag)
    payload = ProductionChatRequest(
        request_id="rag-request-2",
        user_message="必须根据私有知识库回答这个问题。",
        user_id="u1",
        thread_id="t1",
        extract_long_memory=False,
        rag_mode="required",
    )
    result = asyncio.run(production_chat_graph(payload, request))
    assert result["finish_reason"] == "rag_evidence_insufficient"
    assert result["rag"]["citations"] == []
    assert "证据不足" in result["final_answer"]
    assert service.kwargs == {}
