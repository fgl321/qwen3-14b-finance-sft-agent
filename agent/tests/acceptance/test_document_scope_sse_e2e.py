from __future__ import annotations

import json
import time
from collections import OrderedDict
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.chat_graph_v2 import router as chat_graph_router
from app.core.config import Settings


class CountingDeepSeek:
    def __init__(self) -> None:
        self.call_count = 0

    async def chat(self, *args: object, **kwargs: object) -> dict:
        self.call_count += 1
        raise AssertionError(
            "DeepSeek must not be called for deterministic scope errors"
        )


class TypedRouterDeepSeek:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.call_count = 0

    async def chat(self, *args: object, **kwargs: object) -> dict:
        self.call_count += 1
        return {
            "message": {
                "content": json.dumps(
                    self.payload,
                    ensure_ascii=False,
                )
            }
        }


class FakeLifecycle:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def list_documents(self, **kwargs: object) -> list[dict]:
        return list(self.rows)


def _row(document_id: str, *, title: str = "") -> dict:
    return {
        "document_id": document_id,
        "title": title or document_id,
        "file_name": f"{title or document_id}.pdf",
        "version": "1",
        "content_hash": f"hash-{document_id}",
        "expired_date": None,
        "metadata": {"visibility": "private"},
    }


def _app(rows: list[dict]) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_graph_router)
    app.state.settings = Settings()
    app.state.deepseek = CountingDeepSeek()
    app.state.rag_document_lifecycle = FakeLifecycle(rows)
    app.state.rag_store = None
    app.state.embedding_provider = None
    app.state.short_memory = None
    app.state.personal_long_memory = None
    app.state.production_graph_service = None
    app.state.semantic_route_cache = OrderedDict()
    app.state.personal_rag_attempt_cache = OrderedDict()
    app.state.personal_request_memory_cache = OrderedDict()
    return app


def _parse_sse(text: str) -> list[dict]:
    frames: list[dict] = []
    for raw_frame in text.split("\n\n"):
        data_line = next(
            (
                line
                for line in raw_frame.splitlines()
                if line.startswith("data:")
            ),
            None,
        )
        if data_line:
            frames.append(json.loads(data_line[5:].strip()))
    return frames


def _post(client: TestClient, message: str, **overrides: object) -> str:
    payload: dict[str, object] = {
        "user_message": message,
        "thread_id": "thread-e2e",
        "user_id": "owner",
        "tenant_id": "personal",
        "knowledge_base_id": "kb_finance_basic",
        "use_short_memory": False,
        "use_long_memory": False,
        "save_memory": False,
        "extract_long_memory": False,
        "enable_rag": True,
        "rag_mode": "auto",
    }
    payload.update(overrides)
    response = client.post("/api/chat/graph-v2/stream", json=payload)
    assert response.status_code == 200
    return response.text


def test_sse_selected_unknown_fails_fast_without_deepseek() -> None:
    app = _app([])
    deepseek = app.state.deepseek
    with TestClient(app) as client:
        frames = _parse_sse(
            _post(
                client,
                "分析这个文档。",
                document_scope={
                    "mode": "selected",
                    "document_ids": ["missing_doc"],
                },
            )
        )
    error = frames[-1]["error"]
    assert error["code"] == "DOCUMENT_SCOPE_NOT_FOUND"
    assert error["details"]["action"] == "select_document"
    assert error["retryable"] is False
    assert "SCOPE_RESOLUTION_FAILED" not in error["details"]["reason_codes"]
    assert deepseek.call_count == 0


def test_sse_ambiguous_semantic_scope_returns_ambiguous() -> None:
    app = _app(
        [
            _row("v1", title="金融知识普及读本（第一版）"),
            _row("v2", title="金融知识普及读本（第二版）"),
        ]
    )
    app.state.deepseek = TypedRouterDeepSeek(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": [
                {
                    "id": "T1",
                    "description": "检索文档",
                    "capabilities": ["knowledge_retrieval"],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "resource_constraints": {
                "include_documents": [
                    {
                        "reference": "金融知识普及读本",
                        "reference_type": "title",
                    }
                ],
                "exclusive": True,
            },
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )
    app.state.production_graph_service = object()
    with TestClient(app) as client:
        frames = _parse_sse(
            _post(client, "根据我上传的文档分析。")
        )
    error = frames[-1]["error"]
    assert error["code"] == "DOCUMENT_SCOPE_AMBIGUOUS"
    assert error["details"]["action"] == "select_document"
    assert "SCOPE_RESOLUTION_FAILED" not in error["details"]["reason_codes"]
    assert app.state.deepseek.call_count == 1


def test_sse_not_found_scope_fails_fast_without_deepseek() -> None:
    app = _app([_row("doc_a", title="保险法基础")])
    app.state.deepseek = TypedRouterDeepSeek(
        {
            "orchestration_mode": "rag",
            "required_capabilities": ["knowledge_retrieval"],
            "task_requirements": [
                {
                    "id": "T1",
                    "description": "检索文档",
                    "capabilities": [
                        "knowledge_retrieval",
                        "citation_validation",
                    ],
                    "requires_citations": True,
                    "task_kind": "retrieval",
                }
            ],
            "retrieval_requirement": "required",
            "citation_requirement": "required",
            "grounding_requirement": "authoritative",
            "retrieval_scope": "uploaded_documents",
            "resource_constraints": {
                "include_documents": [
                    {
                        "reference": "金融知识普及读本（第二版）",
                        "reference_type": "title",
                    }
                ],
                "exclusive": True,
            },
            "confidence": 0.9,
            "reason_summary": "test",
        }
    )
    app.state.production_graph_service = object()
    with TestClient(app) as client:
        frames = _parse_sse(
            _post(
                client,
                "请必须检索我上传的《金融知识普及读本（第二版）》。",
            )
        )
    error = frames[-1]["error"]
    assert error["code"] == "DOCUMENT_SCOPE_NOT_FOUND"
    assert error["details"]["action"] == "select_document"
    assert "SCOPE_RESOLUTION_FAILED" not in error["details"]["reason_codes"]
    assert app.state.deepseek.call_count == 1


def test_sse_resolved_scope_passes_fail_fast() -> None:
    app = _app([_row("doc_a")])
    with TestClient(app) as client:
        frames = _parse_sse(
            _post(
                client,
                "必须检索我上传的文档。",
                document_scope={
                    "mode": "selected",
                    "document_ids": ["doc_a"],
                },
            )
        )
    error = frames[-1]["error"]
    # The scope resolution passed; the request proceeds and fails later at the
    # graph-service availability check (no production graph in this test).
    assert error["code"] == "GRAPH_SERVICE_UNAVAILABLE"
    assert "DOCUMENT_SCOPE_" not in error["code"]


def test_sse_error_frame_contains_envelope_fields() -> None:
    app = _app([])
    with TestClient(app) as client:
        frames = _parse_sse(
            _post(
                client,
                "分析这个文档。",
                document_scope={
                    "mode": "selected",
                    "document_ids": ["missing_doc"],
                },
            )
        )
    frame = frames[-1]
    error = frame["error"]
    assert frame["event"] == "error"
    assert frame["request_id"]
    assert error["error_id"]
    assert error["message"]
    assert error["details"]["user_action_required"] is True


def test_sse_scope_error_returns_without_keepalive_delay() -> None:
    app = _app([])
    with TestClient(app) as client:
        started = time.monotonic()
        frames = _parse_sse(
            _post(
                client,
                "分析这个文档。",
                document_scope={
                    "mode": "selected",
                    "document_ids": ["missing_doc"],
                },
            )
        )
        elapsed = time.monotonic() - started
    assert frames[-1]["error"]["code"] == "DOCUMENT_SCOPE_NOT_FOUND"
    assert elapsed < 5.0
