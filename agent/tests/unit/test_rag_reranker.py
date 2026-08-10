from __future__ import annotations

import math
import pytest
from types import SimpleNamespace

from app.rag.chunker import ParentChildChunker
from app.rag.rag_types import RetrievedChunk
from app.rag.reranker import (
    BgeReranker,
    HttpReranker,
    NoopReranker,
)


def _chunk(
    chunk_id: str,
    text: str,
    score: float = 50.0,
    metadata: dict | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc_1",
        file_name="a.md",
        text=text,
        score=score,
        page_start=1,
        page_end=1,
        section_path=[],
        metadata=metadata or {},
    )


def test_noop_reranker_keeps_order() -> None:
    chunks = [_chunk("c1", "a"), _chunk("c2", "b")]
    result = NoopReranker().rerank(query="q", candidates=chunks)
    assert [c.chunk_id for c in result] == ["c1", "c2"]


def test_normalize_scores_min_max() -> None:
    chunks = [
        _chunk("c1", "a", metadata={"rerank_raw_score": 1.0}),
        _chunk("c2", "b", metadata={"rerank_raw_score": 3.0}),
        _chunk("c3", "c", metadata={"rerank_raw_score": 5.0}),
    ]
    normalized = BgeReranker.normalize_scores(chunks)
    assert normalized[0].score == 0.0
    assert normalized[1].score == 50.0
    assert normalized[2].score == 100.0
    assert normalized[2].score_display == "100.00/100"
    assert normalized[2].metadata["rerank_score"] == 100.0


def test_normalize_scores_single_chunk_is_100() -> None:
    chunks = [_chunk("c1", "a", metadata={"rerank_raw_score": 2.0})]
    normalized = BgeReranker.normalize_scores(chunks)
    assert normalized[0].score == 100.0


def test_apply_rerank_score_attaches_probability() -> None:
    chunk = _chunk("c1", "a")
    scored = BgeReranker._apply_rerank_score(
        chunk=chunk,
        raw_score=1.0,
    )
    assert "rerank_probability" in scored.metadata
    assert scored.metadata["rerank_probability"] == pytest.approx(
        1.0 / (1.0 + math.exp(-1.0)),
    )


def test_http_reranker_scores_single_candidate() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"scores": [2.0]}

    reranker = HttpReranker(
        base_url="http://example.invalid",
        top_k=6,
    )
    reranker._client = SimpleNamespace(
        post=lambda *args, **kwargs: FakeResponse()
    )

    result = reranker.rerank(
        query="q",
        candidates=[_chunk("c1", "a")],
    )

    assert len(result) == 1
    assert result[0].metadata["rerank_probability"] == pytest.approx(
        1.0 / (1.0 + math.exp(-2.0)),
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("# 第一章", (1, "第一章")),
        ("## 1.1 概览", (2, "1.1 概览")),
        ("一、家庭收支", (1, "家庭收支")),
        ("1. 收入", (2, "收入")),
        ("普通段落文字", None),
        ("", None),
    ],
)
def test_heading_of(line: str, expected) -> None:
    assert ParentChildChunker._heading_of(line) == expected


def test_update_heading_path_nesting() -> None:
    chunker = ParentChildChunker()
    path: list[str] = []
    path = chunker._update_heading_path(path, (1, "家庭"))
    path = chunker._update_heading_path(path, (2, "支出"))
    assert path == ["家庭", "支出"]
    path = chunker._update_heading_path(path, (2, "收入"))
    assert path == ["家庭", "收入"]
    path = chunker._update_heading_path(path, (1, "保险"))
    assert path == ["保险"]
