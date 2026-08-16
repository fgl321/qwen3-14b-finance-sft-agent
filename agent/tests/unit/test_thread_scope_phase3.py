from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.api.routes.chat_graph_v2 import (
    ProductionChatRequest,
    _persist_thread_scope,
    _resolve_document_scope,
)
from app.control_plane.floor import ExplicitConstraintParser
from app.control_plane.metrics import RED_LINE_METRICS
from app.core.config import Settings
from app.memory.short_term_memory import ShortTermMemoryService


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int] = {}

    def setex(self, key: str, seconds: int, value: str) -> None:
        self.values[key] = value
        self.expiries[key] = seconds

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self.values:
                del self.values[key]
                count += 1
            if key in self.expiries:
                del self.expiries[key]
        return count

    def pipeline(self, transaction: bool = True) -> Any:
        return self

    def rpush(self, *args: Any) -> Any:
        return self

    def ltrim(self, *args: Any) -> Any:
        return self

    def expire(self, *args: Any) -> Any:
        return self

    def execute(self) -> list[bool]:
        return [True]

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        return []

    def llen(self, key: str) -> int:
        return 0

    def ttl(self, key: str) -> int:
        return self.expiries.get(key, -2)


def _payload(message: str, **overrides: object) -> ProductionChatRequest:
    values: dict[str, object] = {
        "user_message": message,
        "thread_id": "thread-1",
        "user_id": "owner",
        "tenant_id": "personal",
        "knowledge_base_id": "kb_finance_basic",
    }
    values.update(overrides)
    return ProductionChatRequest(**values)


def test_thread_meta_roundtrip_and_clear() -> None:
    fake = FakeRedis()
    service = ShortTermMemoryService(
        redis_client=fake,
        ttl_seconds=600,
    )
    service.set_thread_meta(
        user_id="u1",
        thread_id="t1",
        tenant_id="personal",
        metadata={
            "active_resource_scope": {
                "scope_id": "uploaded_documents",
                "document_ids": ["doc_a"],
            }
        },
    )
    meta = service.get_thread_meta(
        user_id="u1",
        thread_id="t1",
        tenant_id="personal",
    )
    assert meta["active_resource_scope"]["document_ids"] == ["doc_a"]
    assert any(seconds == 600 for seconds in fake.expiries.values())
    service.clear_thread(
        user_id="u1",
        thread_id="t1",
        tenant_id="personal",
    )
    assert (
        service.get_thread_meta(
            user_id="u1",
            thread_id="t1",
            tenant_id="personal",
        )
        is None
    )


def test_no_keyword_continuation_helpers() -> None:
    import app.api.routes.chat_graph_v2 as module

    assert not hasattr(module, "_is_continuation")
    assert not hasattr(module, "_is_clear_scope")


@pytest.mark.asyncio
async def test_resolve_document_scope_mode_none() -> None:
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=Settings(),
                rag_document_lifecycle=None,
                rag_store=None,
                embedding_provider=None,
            )
        )
    )
    plan = await _resolve_document_scope(
        request=request,
        payload=_payload(
            "这次不要参考文档。",
            document_scope={"mode": "none", "document_ids": []},
        ),
        constraints=ExplicitConstraintParser().parse(
            request_id="req-1",
            user_message="这次不要参考文档。",
        ),
        request_id="req-1",
    )
    assert plan["error"] is None
    assert plan["allowed_document_ids"] == []
    assert plan["audit"]["needs_resolution"] is False


@pytest.mark.asyncio
async def test_persist_thread_scope_sets_and_clears() -> None:
    fake = FakeRedis()
    service = ShortTermMemoryService(
        redis_client=fake,
        ttl_seconds=600,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                short_memory=service,
                settings=Settings(),
            )
        )
    )
    payload = _payload("继续按照刚才那份分析。")
    await _persist_thread_scope(
        request=request,
        payload=payload,
        scope_plan={
            "allowed_document_ids": ["doc_a"],
            "audit": {"source": "title"},
        },
    )
    meta = json.loads(
        next(iter(fake.values.values()))
    )
    assert meta["active_resource_scope"]["document_ids"] == ["doc_a"]
    await _persist_thread_scope(
        request=request,
        payload=_payload(
            "这次不要用文档。",
            document_scope={"mode": "none", "document_ids": []},
        ),
        scope_plan={"allowed_document_ids": [], "audit": {}},
    )
    assert not fake.values


def test_red_line_metrics_include_citation_and_contract() -> None:
    assert "citation_scope_violation_total" in RED_LINE_METRICS
    assert "contract_execution_violation_total" in RED_LINE_METRICS
