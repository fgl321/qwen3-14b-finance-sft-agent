from __future__ import annotations

from types import SimpleNamespace

from app.rag.embeddings import SparseEmbedding, TextEmbedding
from app.rag.qdrant_store import QdrantRagStore
from app.rag.rag_types import RagChunk


class RecordingEmbeddingProvider:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[TextEmbedding]:
        self.batches.append(list(texts))
        return [
            TextEmbedding(
                dense=[float(index + 1), 0.0, 0.0],
                sparse=SparseEmbedding(indices=[index + 1], values=[1.0]),
            )
            for index, _ in enumerate(texts)
        ]


class RecordingClient:
    def __init__(self) -> None:
        self.points = []

    def upsert(self, *, collection_name, points, wait):
        del collection_name, wait
        self.points.extend(points)


def _chunk(chunk_id: str, text: str, chunk_type: str) -> RagChunk:
    return RagChunk(
        chunk_id=chunk_id,
        parent_id=None if chunk_type == "parent" else "parent-1",
        document_id="doc-1",
        tenant_id="personal",
        owner_user_id="owner",
        knowledge_base_id="kb_finance_basic",
        visibility="private",
        file_name="report.pdf",
        text=text,
        metadata={
            "chunk_type": chunk_type,
            "title": "报告",
            "aliases": ["report.pdf", "报告"],
        },
    )


def test_upsert_embeds_only_unique_children_and_batches_progress() -> None:
    store = QdrantRagStore.__new__(QdrantRagStore)
    store.settings = SimpleNamespace(rag_dense_vector_size=3)
    store.client = RecordingClient()
    store.collection_name = "knowledge"
    store.dense_vector_name = "dense"
    store.sparse_vector_name = "sparse"
    provider = RecordingEmbeddingProvider()
    progress: list[tuple[str, int, int]] = []

    result = store.upsert_chunks(
        chunks=[
            _chunk("parent-1", "parent payload", "parent"),
            _chunk("child-1", "same footer", "child"),
            _chunk("child-2", "same footer", "child"),
            _chunk("child-3", "unique body", "child"),
        ],
        embedding_provider=provider,
        embedding_batch_size=1,
        batch_size=2,
        progress_callback=lambda phase, current, total: progress.append(
            (phase, current, total)
        ),
    )

    assert provider.batches == [["same footer"], ["unique body"]]
    assert result["embedded_unique_child_count"] == 2
    assert result["reused_child_embedding_count"] == 1
    assert result["skipped_parent_embedding_count"] == 1
    parent = store.client.points[0]
    assert parent.vector == {"dense": [0.0, 0.0, 0.0]}
    assert parent.payload["document_title"] == "报告"
    assert parent.payload["aliases"] == ["report.pdf", "报告"]
    assert progress[-1] == ("indexing", 4, 4)
