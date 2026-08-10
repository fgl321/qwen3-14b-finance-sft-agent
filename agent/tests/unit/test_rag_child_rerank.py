from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from app.rag.qdrant_store import QdrantRagStore
from app.rag.rag_types import RetrievedChunk
from app.rag.reranker import BgeReranker


class _FakeReranker:
    """按子块文本是否含邮箱给分，模拟真实重排器的相关性差异。"""

    def __init__(self) -> None:
        self.candidate_texts: list[str] = []

    def rerank(
        self,
        *,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        del query
        scored: list[RetrievedChunk] = []
        for candidate in candidates:
            self.candidate_texts.append(candidate.text)
            raw_score = 3.0 if "邮箱" in candidate.text else 0.0
            scored.append(
                BgeReranker._apply_rerank_score(
                    chunk=candidate,
                    raw_score=raw_score,
                )
            )
        return scored


class _FakeEmbedding:
    def embed_query(self, query: str) -> Any:
        del query
        return SimpleNamespace(
            dense=[0.1],
            sparse=SimpleNamespace(indices=[1], values=[0.2]),
        )


class _FakeClient:
    def __init__(self, parent_points: list[Any]) -> None:
        self.parent_points = parent_points

    def query_points(self, **kwargs: Any) -> Any:
        del kwargs
        return SimpleNamespace(points=[])

    def retrieve(self, **kwargs: Any) -> list[Any]:
        del kwargs
        return list(self.parent_points)


def _hit(child_id: str, parent_id: str, text: str, score: float) -> Any:
    return SimpleNamespace(
        id=child_id,
        score=score,
        payload={
            "chunk_id": child_id,
            "parent_id": parent_id,
            "document_id": "doc_1",
            "file_name": "resume.pdf",
            "text": text,
        },
    )


def _parent_point(parent_id: str, text: str) -> Any:
    return SimpleNamespace(
        id=parent_id,
        payload={
            "chunk_id": parent_id,
            "document_id": "doc_1",
            "file_name": "resume.pdf",
            "text": text,
            "page_start": 1,
            "page_end": 1,
            "section_path": [],
            "metadata": {"chunk_type": "parent"},
        },
    )


def _build_store(fake_client: _FakeClient) -> QdrantRagStore:
    store = QdrantRagStore.__new__(QdrantRagStore)
    store.settings = SimpleNamespace(
        rag_fusion_dense_weight=0.5,
        rag_fusion_sparse_weight=0.5,
        rag_fusion_rrf_k=60,
    )
    store.client = fake_client
    store.collection_name = "test_collection"
    store.dense_vector_name = "dense"
    store.sparse_vector_name = "sparse"
    return store


def test_parent_gets_child_level_rerank_probability() -> None:
    parent_text = (
        "实习经历很长很长" * 300
        + "\n姓名：范广禄\n邮箱 17743498317@163.com"
    )
    child_text = "姓名：范广禄 邮箱 17743498317@163.com"

    fake_client = _FakeClient(
        [_parent_point("parent_1", parent_text)]
    )
    store = _build_store(fake_client)
    reranker = _FakeReranker()

    def fake_dense(**kwargs: Any) -> list[Any]:
        del kwargs
        return [
            _hit("child_a", "parent_1", "实习经历", 0.9),
            _hit("child_b", "parent_1", child_text, 0.8),
        ]

    def fake_sparse(**kwargs: Any) -> list[Any]:
        del kwargs
        return [
            _hit("child_b", "parent_1", child_text, 0.9),
        ]

    store._search_child_dense = fake_dense  # type: ignore[method-assign]
    store._search_child_sparse = fake_sparse  # type: ignore[method-assign]

    result = store.search_relevant_parent_chunks(
        query="范广禄的邮箱是多少",
        tenant_id="t",
        owner_user_id="u",
        knowledge_base_id="kb",
        embedding_provider=_FakeEmbedding(),
        child_limit=8,
        parent_limit=4,
        reranker=reranker,
        min_score=0.0,
    )

    assert len(result) == 1
    parent = result[0]
    # 返回的仍是完整父块文本，而不是子块文本。
    assert parent.text == parent_text
    # 重排概率来自含邮箱的子块。
    assert parent.metadata["rerank_input_type"] == "child"
    assert parent.metadata["rerank_child_score_source"] is True
    assert math.isclose(
        parent.metadata["rerank_probability"],
        1.0 / (1.0 + math.exp(-3.0)),
        abs_tol=1e-5,
    )
    # 重排时打分用的确实是子块文本。
    assert child_text in reranker.candidate_texts
