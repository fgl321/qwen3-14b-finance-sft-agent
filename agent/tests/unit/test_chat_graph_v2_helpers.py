from __future__ import annotations

from app.api.routes.chat_graph_v2 import (
    ProductionChatRequest,
    _rag_direct_execution_path,
)


def _payload(document_ids: list[str]) -> ProductionChatRequest:
    return ProductionChatRequest(
        user_message="测试",
        user_id="u1",
        thread_id="t1",
        document_ids=document_ids,
    )


def _chunk(document_id: str, retrieval_mode: str) -> dict:
    return {
        "document_id": document_id,
        "metadata": {"retrieval_mode": retrieval_mode},
    }


def test_positional_fallback_is_attachment_direct() -> None:
    rag = {
        "retrieved_chunks": [
            _chunk("doc_a", "document_scope_positional_fallback")
        ]
    }
    assert (
        _rag_direct_execution_path(rag, _payload(["doc_a"]))
        == "attachment_direct"
    )


def test_scoped_document_is_attachment_direct() -> None:
    rag = {
        "retrieved_chunks": [
            _chunk("doc_a", "hybrid_dense_sparse")
        ]
    }
    assert (
        _rag_direct_execution_path(rag, _payload(["doc_a"]))
        == "attachment_direct"
    )


def test_unscoped_is_kb_direct() -> None:
    rag = {
        "retrieved_chunks": [
            _chunk("doc_a", "hybrid_dense_sparse")
        ]
    }
    assert (
        _rag_direct_execution_path(rag, _payload([]))
        == "kb_direct"
    )


def test_scoped_but_chunks_from_other_doc_is_kb_direct() -> None:
    rag = {
        "retrieved_chunks": [
            _chunk("doc_b", "hybrid_dense_sparse")
        ]
    }
    assert (
        _rag_direct_execution_path(rag, _payload(["doc_a"]))
        == "kb_direct"
    )
