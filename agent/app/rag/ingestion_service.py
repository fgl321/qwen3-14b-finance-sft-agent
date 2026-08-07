from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.chunker import ChunkingConfig, ParentChildChunker
from app.rag.document_parser import DocumentParser
from app.rag.embedding_factory import build_embedding_provider
from app.rag.embeddings import EmbeddingProvider
from app.rag.qdrant_store import QdrantRagStore


logger = get_logger(__name__)


class RagIngestionService:
    """
    RAG 文档入库服务。

    重要行为：
    - document_id 由 tenant_id + knowledge_base_id + file_sha256 稳定生成。
    - 同一个文档重复上传时，先删除旧分块，再写入新分块。
    - embedding_provider 支持外部注入，避免每次上传都重新加载 BGE-M3。
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        store: QdrantRagStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.parser = DocumentParser()

        self.chunker = ParentChildChunker(
            config=ChunkingConfig(
                parent_token_limit=900,
                parent_overlap_tokens=120,
                child_token_limit=280,
                child_overlap_tokens=60,
                min_chunk_tokens=20,
            )
        )

        self.embedding_provider = embedding_provider or build_embedding_provider(
            settings=self.settings,
        )

        self.store = store or QdrantRagStore(
            settings=self.settings,
        )

    def ingest_file(
        self,
        *,
        file_path: str | Path,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        visibility: str = "private",
        original_file_name: str | None = None,
    ) -> dict[str, Any]:
        parsed = self.parser.parse(
            file_path=file_path,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            visibility=visibility,
            display_file_name=original_file_name,
        )

        logger.info(
            "rag_document_parsed",
            document_id=parsed.meta.document_id,
            file_name=parsed.meta.file_name,
            file_sha256=parsed.meta.file_sha256,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            page_count=len(parsed.pages),
        )

        chunks = self.chunker.chunk(parsed)

        counter = Counter(
            chunk.metadata.get("chunk_type")
            for chunk in chunks
        )

        logger.info(
            "rag_document_chunked",
            document_id=parsed.meta.document_id,
            total_chunks=len(chunks),
            parent_count=counter.get("parent", 0),
            child_count=counter.get("child", 0),
        )

        delete_existing_result = self.store.delete_document(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            document_id=parsed.meta.document_id,
        )

        logger.info(
            "rag_existing_document_deleted_before_ingest",
            document_id=parsed.meta.document_id,
            deleted_count_estimate=delete_existing_result.get(
                "deleted_count_estimate"
            ),
        )

        qdrant_result = self.store.upsert_chunks(
            chunks=chunks,
            embedding_provider=self.embedding_provider,
            batch_size=64,
        )

        qdrant_result["delete_existing"] = delete_existing_result
        qdrant_result["embedding_provider"] = self.settings.embedding_provider

        point_count = self.store.count_points()

        logger.info(
            "rag_document_ingested",
            document_id=parsed.meta.document_id,
            file_name=parsed.meta.file_name,
            embedding_provider=self.settings.embedding_provider,
            upserted_count=qdrant_result.get("upserted_count"),
            point_count_after_ingest=point_count,
        )

        return {
            "ok": True,
            "document": parsed.meta.model_dump(),
            "chunks": {
                "total_chunks": len(chunks),
                "parent_count": counter.get("parent", 0),
                "child_count": counter.get("child", 0),
            },
            "qdrant": qdrant_result,
            "point_count_after_ingest": point_count,
        }
