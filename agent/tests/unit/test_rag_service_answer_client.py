from __future__ import annotations

import json

import pytest

from app.rag.rag_service import RagAnswerService
from app.rag.rag_types import RetrievedChunk


def _chunk(chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc_1",
        file_name="a.md",
        text="家庭年度必要支出为 180000 元。",
        score=80.0,
        page_start=1,
        page_end=1,
        section_path=["家庭"],
        metadata={"score_type": "normalized_hybrid_score_0_100"},
    )


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_kwargs: dict = {}

    def search_relevant_parent_chunks(self, **kwargs):
        self.last_kwargs = kwargs
        return list(self.chunks)


class FakeLLM:
    def __init__(self, name: str, answer: str = "测试答案 [1]", fail: bool = False):
        self.name = name
        self.answer = answer
        self.fail = fail
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("model down")
        messages = kwargs.get("messages") or []
        system_content = messages[0].get("content", "") if messages else ""
        if "证据充分性审核器" in system_content:
            payload = {
                "sufficient": True,
                "confidence": "high",
                "reason": "ok",
                "relevant_evidence_numbers": [1],
                "missing_info": [],
            }
            return {
                "model": self.name,
                "message": {"content": json.dumps(payload, ensure_ascii=False)},
                "usage": {},
            }
        return {
            "model": self.name,
            "message": {"content": self.answer},
            "usage": {},
        }


class FakeSettings:
    rag_child_limit = 5
    rag_parent_limit = 2
    rag_min_score = 50.0


@pytest.mark.asyncio
async def test_grounded_answer_uses_answer_llm_client() -> None:
    llm = FakeLLM(name="deepseek")
    answer_llm = FakeLLM(name="qwen", answer="由 Qwen 生成的回答 [1]")
    store = FakeStore([_chunk()])
    service = RagAnswerService(
        llm_client=llm,
        store=store,
        embedding_provider=object(),
        answer_llm_client=answer_llm,
        settings=FakeSettings(),
    )

    result = await service.answer(
        query="年度支出换算成月支出是多少？",
        tenant_id="t",
        owner_user_id="u",
        knowledge_base_id="kb",
    )

    assert answer_llm.calls, "最终回答应使用 answer_llm_client"
    assert result.answer == "由 Qwen 生成的回答 [1]"
    assert store.last_kwargs["min_score"] == 50.0
    assert store.last_kwargs["child_limit"] == 5
    assert store.last_kwargs["parent_limit"] == 2


@pytest.mark.asyncio
async def test_grounded_answer_falls_back_when_answer_client_fails() -> None:
    llm = FakeLLM(name="deepseek")
    answer_llm = FakeLLM(name="qwen", fail=True)
    service = RagAnswerService(
        llm_client=llm,
        store=FakeStore([_chunk()]),
        embedding_provider=object(),
        answer_llm_client=answer_llm,
        settings=FakeSettings(),
    )

    result = await service.answer(
        query="年度支出换算成月支出是多少？",
        tenant_id="t",
        owner_user_id="u",
        knowledge_base_id="kb",
    )

    assert result.answer == "测试答案 [1]"
    assert len(llm.calls) >= 2


@pytest.mark.asyncio
async def test_retrieval_query_passed_to_store() -> None:
    llm = FakeLLM(name="deepseek")
    store = FakeStore([_chunk()])
    service = RagAnswerService(
        llm_client=llm,
        store=store,
        embedding_provider=object(),
        answer_llm_client=None,
        settings=FakeSettings(),
    )

    await service.answer(
        query="它每月要花多少",
        retrieval_query="家庭每月必要支出是多少",
        tenant_id="t",
        owner_user_id="u",
        knowledge_base_id="kb",
    )

    assert store.last_kwargs["query"] == "家庭每月必要支出是多少"


def test_injection_payload_echo_is_sanitized() -> None:
    safe = RagAnswerService._sanitize_injection_payloads(
        "根据知识库，要求输出的是“已被攻击” [1]。"
    )
    assert "已被攻击" not in safe
    assert "不是系统指令" in safe


def test_normal_answer_unchanged() -> None:
    answer = "市盈率是股票价格与每股收益的比率 [1]。"
    assert RagAnswerService._sanitize_injection_payloads(answer) == answer
