from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
import time
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.chunker import ChunkingConfig, ParentChildChunker
from app.rag.document_parser import DocumentParser
from app.rag.embedding_factory import build_embedding_provider
from app.rag.embeddings import EmbeddingProvider
from app.rag.file_utils import calculate_sha256
from app.rag.qdrant_store import QdrantRagStore
from app.rag.source_classifier import SourceClassifier
from app.rag.rag_types import (
    DocumentMeta,
    ParsedDocument,
    ParsedPage,
)


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
        source_classifier: SourceClassifier | None = None,
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
        self.source_classifier = (
            source_classifier
            or SourceClassifier(settings=self.settings)
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
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        stage_started = started
        timings: dict[str, float] = {}

        def report(phase: str, percent: float, message: str) -> None:
            if progress_callback is not None:
                progress_callback(phase, max(0.0, min(99.0, percent)), message)

        report("parsing", 2, "正在解析文档页面")
        legacy_parsed = self.parser.parse(
            file_path=file_path,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            visibility=visibility,
            display_file_name=original_file_name,
        )
        timings["parse_seconds"] = round(time.perf_counter() - stage_started, 3)
        report("classifying", 14, "正在识别知识来源与可信度")
        stage_started = time.perf_counter()

        # DocumentParser.parse() 返回旧版 str 兼容对象，没有 .meta。
        # 这里把它转换成 rag_types.ParsedDocument，供父子分块器使用。
        file_sha256 = calculate_sha256(file_path)
        document_id = str(
            uuid5(
                NAMESPACE_URL,
                "|".join(
                    [
                        str(tenant_id),
                        str(owner_user_id),
                        str(knowledge_base_id),
                        file_sha256,
                    ]
                ),
            )
        )
        display_file_name = (
            original_file_name
            or getattr(legacy_parsed, "file_name", None)
            or Path(file_path).name
        )
        parsed_meta = (
            getattr(legacy_parsed, "metadata", None) or {}
        )
        extracted_title = str(
            parsed_meta.get("title") or ""
        ).strip()
        title = extracted_title or display_file_name
        aliases = [
            str(item).strip()
            for item in (parsed_meta.get("aliases") or [])
            if str(item).strip()
        ]
        for value in (
            display_file_name,
            Path(display_file_name).stem,
            title,
        ):
            if value and value not in aliases:
                aliases.append(value)
        source_metadata = self.source_classifier.classify(
            text=str(legacy_parsed)[:16000],
            file_name=display_file_name,
        )
        timings["classify_seconds"] = round(time.perf_counter() - stage_started, 3)
        meta = DocumentMeta(
            document_id=document_id,
            file_name=display_file_name,
            title=title,
            aliases=aliases,
            file_sha256=file_sha256,
            tenant_id=str(tenant_id),
            owner_user_id=str(owner_user_id),
            knowledge_base_id=str(knowledge_base_id),
            source_type=getattr(
                legacy_parsed,
                "source_type",
                Path(file_path).suffix.lower().lstrip(".") or "text",
            ),
            visibility=visibility,
            version=1,
            content_type=source_metadata.content_type,
            scope=source_metadata.scope,
            trust_level=source_metadata.trust_level,
            generated_content=source_metadata.generated_content,
            allow_rag_direct=source_metadata.allow_rag_direct,
        )
        parsed = ParsedDocument(
            meta=meta,
            pages=[
                ParsedPage(
                    page_number=page.page_number,
                    text=page.text,
                )
                for page in (legacy_parsed.pages or [])
            ],
        )

        total_pages = int(
            parsed_meta.get("total_pages") or 0
        )
        extracted_pages = int(
            parsed_meta.get("extracted_pages") or len(parsed.pages or [])
        )
        skipped_image_pages = max(
            total_pages - extracted_pages,
            0,
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

        report("chunking", 18, "正在执行父子分块")
        stage_started = time.perf_counter()
        chunks = self.chunker.chunk(parsed)
        timings["chunk_seconds"] = round(time.perf_counter() - stage_started, 3)

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

        def vector_progress(phase: str, current: int, total: int) -> None:
            ratio = current / max(total, 1)
            if phase == "embedding":
                report(
                    "embedding",
                    20 + ratio * 65,
                    f"GPU 向量化 {current}/{total} 个唯一子块",
                )
            else:
                report(
                    "indexing",
                    85 + ratio * 13,
                    f"写入向量库 {current}/{total} 个分块",
                )

        report("embedding", 20, "正在批量生成向量")
        stage_started = time.perf_counter()
        qdrant_result = self.store.upsert_chunks(
            chunks=chunks,
            embedding_provider=self.embedding_provider,
            batch_size=int(
                getattr(self.settings, "rag_qdrant_upsert_batch_size", 128)
            ),
            embedding_batch_size=int(
                getattr(self.settings, "rag_ingest_embedding_batch_size", 128)
            ),
            progress_callback=vector_progress,
        )
        timings["embed_and_index_seconds"] = round(
            time.perf_counter() - stage_started,
            3,
        )

        qdrant_result["delete_existing"] = delete_existing_result
        qdrant_result["embedding_provider"] = self.settings.embedding_provider

        point_count = self.store.count_points()
        timings["total_seconds"] = round(time.perf_counter() - started, 3)
        report("finalizing", 99, "正在完成文档索引")

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
            "page_stats": {
                "total_pages": total_pages,
                "extracted_pages": extracted_pages,
                "skipped_image_pages": skipped_image_pages,
            },
            "qdrant": qdrant_result,
            "timings": timings,
            "point_count_after_ingest": point_count,
        }
