from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

# 必须放在 qdrant_client 导入之前。
# 目的：确保本地 Qdrant 请求不走系统代理。
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

for proxy_key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]:
    os.environ.pop(proxy_key, None)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    SparseVector,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingProvider, TextEmbedding
from app.rag.rag_types import RagChunk, RetrievedChunk


logger = get_logger(__name__)


class QdrantRagStore:
    """
    RAG 向量库访问层。

    当前能力：
    1. parent / child chunk 入库。
    2. dense 语义召回。
    3. sparse 关键词召回。
    4. dense + sparse 排名融合。
    5. 将融合后的原始分数放入 metadata。
    6. 将对外展示 score 归一化为 0~100。
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            timeout=int(self.settings.qdrant_timeout),
            prefer_grpc=False,
        )

        self.collection_name = self.settings.qdrant_collection
        self.dense_vector_name = self.settings.rag_dense_vector_name
        self.sparse_vector_name = self.settings.rag_sparse_vector_name

    def upsert_chunks(
        self,
        *,
        chunks: list[RagChunk],
        embedding_provider: EmbeddingProvider,
        batch_size: int = 64,
    ) -> dict[str, Any]:
        if not chunks:
            return {
                "ok": True,
                "upserted_count": 0,
                "batch_count": 0,
                "collection_name": self.collection_name,
            }

        total_upserted = 0
        batch_count = 0
        ingested_at = datetime.now(timezone.utc).isoformat()

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start: start + batch_size]
            texts = [chunk.text for chunk in batch_chunks]
            embeddings = embedding_provider.embed_documents(texts)

            if len(embeddings) != len(batch_chunks):
                raise ValueError(
                    "embedding 数量和 chunk 数量不一致："
                    f"embeddings={len(embeddings)}, chunks={len(batch_chunks)}"
                )

            points: list[PointStruct] = []

            for chunk, embedding in zip(batch_chunks, embeddings):
                payload = self._chunk_to_payload(
                    chunk=chunk,
                    ingested_at=ingested_at,
                )

                points.append(
                    PointStruct(
                        id=chunk.chunk_id,
                        vector={
                            self.dense_vector_name: embedding.dense,
                            self.sparse_vector_name: SparseVector(
                                indices=embedding.sparse.indices,
                                values=embedding.sparse.values,
                            ),
                        },
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            total_upserted += len(points)
            batch_count += 1

        return {
            "ok": True,
            "upserted_count": total_upserted,
            "batch_count": batch_count,
            "collection_name": self.collection_name,
            "ingested_at": ingested_at,
        }

    def search_relevant_parent_chunks(
        self,
        *,
        query: str,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        embedding_provider: EmbeddingProvider,
        child_limit: int = 8,
        parent_limit: int = 4,
        score_threshold: float | None = None,
        reranker: Any | None = None,
        min_score: float | None = None,
        rerank_candidate_limit: int = 12,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = embedding_provider.embed_query(query)

        child_filter = self._build_child_search_filter(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            document_ids=document_ids,
        )

        dense_hits = self._search_child_dense(
            query_embedding=query_embedding,
            child_filter=child_filter,
            limit=child_limit,
            score_threshold=score_threshold,
        )

        sparse_hits = self._search_child_sparse(
            query_embedding=query_embedding,
            child_filter=child_filter,
            limit=child_limit,
        )

        fused_hits = self._fuse_child_hits(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            dense_weight=self.settings.rag_fusion_dense_weight,
            sparse_weight=self.settings.rag_fusion_sparse_weight,
            rrf_k=self.settings.rag_fusion_rrf_k,
        )

        logger.info(
            "rag_hybrid_child_search_finished",
            query=query,
            dense_hit_count=len(dense_hits),
            sparse_hit_count=len(sparse_hits),
            fused_hit_count=len(fused_hits),
        )

        if not fused_hits:
            if document_ids:
                fallback = self._fallback_parent_chunks(
                    tenant_id=tenant_id,
                    owner_user_id=owner_user_id,
                    knowledge_base_id=knowledge_base_id,
                    document_ids=document_ids,
                    parent_limit=parent_limit,
                )
                if fallback:
                    logger.info(
                        "rag_document_scope_positional_fallback",
                        query=query,
                        document_ids=list(document_ids),
                        fallback_count=len(fallback),
                    )
                    return fallback
            return []

        parent_best_scores: dict[str, float] = {}
        parent_best_debug: dict[str, dict[str, Any]] = {}
        parent_child_hits: dict[str, list[dict[str, Any]]] = {}

        for fused_hit in fused_hits:
            payload = fused_hit["payload"]
            parent_id = payload.get("parent_id")

            if not parent_id:
                continue

            relevance_score = float(fused_hit["relevance_score"])

            current_best = parent_best_scores.get(parent_id)

            if current_best is None or relevance_score > current_best:
                parent_best_scores[parent_id] = relevance_score
                parent_best_debug[parent_id] = self._build_retrieval_debug(
                    fused_hit=fused_hit,
                )

            parent_child_hits.setdefault(parent_id, []).append(
                {
                    "child_id": payload.get("chunk_id"),
                    "score": relevance_score,
                    "score_display": fused_hit.get("score_display"),
                    "score_type": "normalized_hybrid_score_0_100",
                    "retrieval_mode": "hybrid_dense_sparse",
                    "fused_score_raw": fused_hit.get("fused_score_raw"),
                    "dense_score": fused_hit.get("dense_score"),
                    "sparse_score": fused_hit.get("sparse_score"),
                    "dense_rank": fused_hit.get("dense_rank"),
                    "sparse_rank": fused_hit.get("sparse_rank"),
                    "text_preview": (payload.get("text") or "")[:160],
                }
            )

        parent_ids = list(parent_best_scores.keys())

        if not parent_ids:
            return []

        parent_points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=parent_ids,
            with_payload=True,
            with_vectors=False,
        )

        retrieved: list[RetrievedChunk] = []

        for point in parent_points:
            payload = point.payload or {}
            parent_id = payload.get("chunk_id")

            if not parent_id:
                continue

            score = parent_best_scores.get(parent_id, 0.0)
            best_debug = parent_best_debug.get(parent_id, {})

            retrieved.append(
                RetrievedChunk(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    document_id=payload.get("document_id", ""),
                    file_name=payload.get("file_name", ""),
                    text=payload.get("text", ""),
                    score=score,
                    page_start=payload.get("page_start"),
                    page_end=payload.get("page_end"),
                    section_path=payload.get("section_path") or [],
                    metadata={
                        **(payload.get("metadata") or {}),
                        "retrieval_mode": "hybrid_dense_sparse",
                        "score_type": "normalized_hybrid_score_0_100",
                        "score_display": f"{score:.2f}/100",
                        "retrieval_debug": best_debug,
                        "matched_child_hits": parent_child_hits.get(
                            parent_id,
                            [],
                        ),
                    },
                )
            )

        retrieved.sort(key=lambda item: item.score, reverse=True)

        if reranker is not None:
            from app.rag.reranker import (
                BgeReranker,
                NoopReranker,
            )

            if not isinstance(reranker, NoopReranker):
                candidates = retrieved[: max(rerank_candidate_limit, 1)]
                reranked = reranker.rerank(
                    query=query,
                    candidates=candidates,
                )
                if reranked:
                    normalized = BgeReranker.normalize_scores(reranked)
                    retrieved = normalized
                    logger.info(
                        "rag_rerank_applied",
                        query=query,
                        before=len(candidates),
                        after=len(normalized),
                    )

        if min_score is not None and min_score > 0:
            before_filter = len(retrieved)
            retrieved = [
                item
                for item in retrieved
                if float(item.score) >= float(min_score)
            ]
            logger.info(
                "rag_min_score_filter_applied",
                min_score=min_score,
                before=before_filter,
                after=len(retrieved),
            )

        return retrieved[: max(parent_limit, 1)]

    def _search_child_dense(
        self,
        *,
        query_embedding: TextEmbedding,
        child_filter: Filter,
        limit: int,
        score_threshold: float | None,
    ) -> list[Any]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.dense,
            using=self.dense_vector_name,
            query_filter=child_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )

        hits = list(response.points)

        logger.info(
            "rag_dense_child_search_finished",
            hit_count=len(hits),
        )

        return hits

    def _search_child_sparse(
        self,
        *,
        query_embedding: TextEmbedding,
        child_filter: Filter,
        limit: int,
    ) -> list[Any]:
        if not query_embedding.sparse.indices:
            logger.info(
                "rag_sparse_child_search_skipped",
                reason="empty_sparse_indices",
            )
            return []

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=SparseVector(
                    indices=query_embedding.sparse.indices,
                    values=query_embedding.sparse.values,
                ),
                using=self.sparse_vector_name,
                query_filter=child_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            hits = list(response.points)

            logger.info(
                "rag_sparse_child_search_finished",
                hit_count=len(hits),
            )

            return hits

        except Exception as exc:
            logger.warning(
                "rag_sparse_child_search_failed_fallback_to_dense_only",
                error=str(exc),
            )
            return []

    @staticmethod
    def _fuse_child_hits(
        *,
        dense_hits: list[Any],
        sparse_hits: list[Any],
        dense_weight: float,
        sparse_weight: float,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        """
        排名融合。

        注意：
        - dense_score 和 sparse_score 原始尺度不同，不能直接加。
        - fused_score_raw 是排名融合后的原始小数。
        - relevance_score 是归一化后的 0~100 分，给前端展示。
        """

        fused: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            point_id = str(hit.id)

            item = fused.setdefault(
                point_id,
                {
                    "id": point_id,
                    "payload": hit.payload or {},
                    "fused_score_raw": 0.0,
                    "dense_score": None,
                    "sparse_score": None,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "relevance_score": 0.0,
                    "score_display": None,
                },
            )

            item["fused_score_raw"] += dense_weight * (1.0 / (rrf_k + rank))
            item["dense_score"] = float(hit.score)
            item["dense_rank"] = rank

        for rank, hit in enumerate(sparse_hits, start=1):
            point_id = str(hit.id)

            item = fused.setdefault(
                point_id,
                {
                    "id": point_id,
                    "payload": hit.payload or {},
                    "fused_score_raw": 0.0,
                    "dense_score": None,
                    "sparse_score": None,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "relevance_score": 0.0,
                    "score_display": None,
                },
            )

            item["fused_score_raw"] += sparse_weight * (1.0 / (rrf_k + rank))
            item["sparse_score"] = float(hit.score)
            item["sparse_rank"] = rank

            if not item.get("payload"):
                item["payload"] = hit.payload or {}

        result = list(fused.values())

        result.sort(
            key=lambda item: float(item["fused_score_raw"]),
            reverse=True,
        )

        if not result:
            return result

        best_raw_score = float(result[0]["fused_score_raw"])

        for item in result:
            raw_score = float(item["fused_score_raw"])

            if best_raw_score > 0:
                relevance_score = round(raw_score / best_raw_score * 100, 4)
            else:
                relevance_score = 0.0

            item["relevance_score"] = relevance_score
            item["score_display"] = f"{relevance_score:.2f}/100"

        return result

    def _build_retrieval_debug(
        self,
        *,
        fused_hit: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "retrieval_mode": "hybrid_dense_sparse",
            "score_type": "normalized_hybrid_score_0_100",
            "score": fused_hit.get("relevance_score"),
            "score_display": fused_hit.get("score_display"),
            "fused_score_raw": fused_hit.get("fused_score_raw"),
            "dense_score": fused_hit.get("dense_score"),
            "sparse_score": fused_hit.get("sparse_score"),
            "dense_rank": fused_hit.get("dense_rank"),
            "sparse_rank": fused_hit.get("sparse_rank"),
            "fusion_method": "weighted_rrf",
            "dense_weight": self.settings.rag_fusion_dense_weight,
            "sparse_weight": self.settings.rag_fusion_sparse_weight,
            "rrf_k": self.settings.rag_fusion_rrf_k,
        }

    def list_documents(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        base_filter = self._build_base_filter(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
        )

        points = self._scroll_all_points(
            scroll_filter=base_filter,
            max_points=max(limit * 100, 1000),
            with_payload=True,
        )

        grouped: dict[str, dict[str, Any]] = {}

        for point in points:
            payload = point.payload or {}
            document_id = payload.get("document_id")

            if not document_id:
                continue

            chunk_type = payload.get("chunk_type")

            current = grouped.get(document_id)

            if current is None:
                current = {
                    "document_id": document_id,
                    "file_name": payload.get("file_name"),
                    "file_sha256": payload.get("file_sha256"),
                    "tenant_id": payload.get("tenant_id"),
                    "owner_user_id": payload.get("owner_user_id"),
                    "knowledge_base_id": payload.get("knowledge_base_id"),
                    "visibility": payload.get("visibility"),
                    "source_type": payload.get("source_type"),
                    "document_version": payload.get("document_version"),
                    "ingested_at": payload.get("ingested_at"),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "parent_count": 0,
                    "child_count": 0,
                    "total_chunks": 0,
                }

                grouped[document_id] = current

            if chunk_type == "parent":
                current["file_name"] = payload.get("file_name") or current.get("file_name")
                current["file_sha256"] = payload.get("file_sha256") or current.get("file_sha256")
                current["tenant_id"] = payload.get("tenant_id") or current.get("tenant_id")
                current["owner_user_id"] = payload.get("owner_user_id") or current.get("owner_user_id")
                current["knowledge_base_id"] = payload.get("knowledge_base_id") or current.get("knowledge_base_id")
                current["visibility"] = payload.get("visibility") or current.get("visibility")
                current["source_type"] = payload.get("source_type") or current.get("source_type")
                current["document_version"] = payload.get("document_version") or current.get("document_version")
                current["ingested_at"] = payload.get("ingested_at") or current.get("ingested_at")

            if chunk_type == "parent":
                current["parent_count"] += 1
            elif chunk_type == "child":
                current["child_count"] += 1

            current["total_chunks"] = current["parent_count"] + current["child_count"]

            page_start = payload.get("page_start")
            page_end = payload.get("page_end")

            if page_start is not None:
                old_page_start = current.get("page_start")
                current["page_start"] = page_start if old_page_start is None else min(old_page_start, page_start)

            if page_end is not None:
                old_page_end = current.get("page_end")
                current["page_end"] = page_end if old_page_end is None else max(old_page_end, page_end)

        documents = list(grouped.values())

        documents.sort(
            key=lambda item: str(item.get("ingested_at") or ""),
            reverse=True,
        )

        return documents[:limit]

    def delete_document(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        document_filter = self._build_document_filter(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )

        chunks_before_delete = self.count_points(
            count_filter=document_filter,
        )

        if chunks_before_delete == 0:
            return {
                "ok": True,
                "document_id": document_id,
                "deleted_count_estimate": 0,
                "point_count_after_delete": self.count_points(),
                "message": "没有找到需要删除的文档分块。",
            }

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=document_filter,
            ),
            wait=True,
        )

        return {
            "ok": True,
            "document_id": document_id,
            "deleted_count_estimate": chunks_before_delete,
            "point_count_after_delete": self.count_points(),
            "message": "文档分块已删除。",
        }

    def count_document_chunks(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        document_id: str,
        chunk_type: str | None = None,
    ) -> int:
        document_filter = self._build_document_filter(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            chunk_type=chunk_type,
        )

        return self.count_points(
            count_filter=document_filter,
        )

    def count_points(
        self,
        count_filter: Filter | None = None,
    ) -> int:
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=count_filter,
                exact=True,
            )
            return int(result.count)

        except TypeError:
            points = self._scroll_all_points(
                scroll_filter=count_filter,
                max_points=100_000,
                with_payload=False,
            )
            return len(points)

    def scroll_points(
        self,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        result: list[dict[str, Any]] = []

        for point in points:
            result.append(
                {
                    "id": str(point.id),
                    "payload": point.payload or {},
                }
            )

        return result

    def _scroll_all_points(
        self,
        *,
        scroll_filter: Filter | None,
        max_points: int,
        with_payload: bool,
    ) -> list[Any]:
        points: list[Any] = []
        offset: Any | None = None

        while len(points) < max_points:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=min(256, max_points - len(points)),
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )

            points.extend(batch)

            if next_offset is None:
                break

            offset = next_offset

        return points

    @staticmethod
    def _build_base_filter(
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
    ) -> Filter:
        return Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id),
                ),
                FieldCondition(
                    key="owner_user_id",
                    match=MatchValue(value=owner_user_id),
                ),
                FieldCondition(
                    key="knowledge_base_id",
                    match=MatchValue(value=knowledge_base_id),
                ),
            ]
        )

    @staticmethod
    def _build_child_search_filter(
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        document_ids: list[str] | None = None,
    ) -> Filter:
        must_conditions = [
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id),
            ),
            FieldCondition(
                key="owner_user_id",
                match=MatchValue(value=owner_user_id),
            ),
            FieldCondition(
                key="knowledge_base_id",
                match=MatchValue(value=knowledge_base_id),
            ),
            FieldCondition(
                key="chunk_type",
                match=MatchValue(value="child"),
            ),
        ]
        if document_ids:
            must_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchAny(
                        any=[str(document_id) for document_id in document_ids]
                    ),
                )
            )
        return Filter(
            must=must_conditions,
        )

    def _fallback_parent_chunks(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        document_ids: list[str],
        parent_limit: int,
    ) -> list[RetrievedChunk]:
        """按文档范围直接取父块（按位置），用于“这个文档讲了什么”类问题。"""
        base = self._build_base_filter(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
        )
        parent_filter = Filter(
            must=[
                *base.must,
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value="parent"),
                ),
                FieldCondition(
                    key="document_id",
                    match=MatchAny(
                        any=[str(document_id) for document_id in document_ids]
                    ),
                ),
            ]
        )
        points = self._scroll_all_points(
            scroll_filter=parent_filter,
            max_points=max(parent_limit * 4, 16),
            with_payload=True,
        )
        retrieved: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    document_id=payload.get("document_id", ""),
                    file_name=payload.get("file_name", ""),
                    text=payload.get("text", ""),
                    score=0.0,
                    page_start=payload.get("page_start"),
                    page_end=payload.get("page_end"),
                    section_path=payload.get("section_path") or [],
                    metadata={
                        **(payload.get("metadata") or {}),
                        "retrieval_mode": "document_scope_positional_fallback",
                        "score_type": "document_scope_fallback",
                        "score_display": "文档范围原文",
                    },
                )
            )
        return retrieved[: max(parent_limit, 1)]

    @staticmethod
    def _build_document_filter(
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        document_id: str,
        chunk_type: str | None = None,
    ) -> Filter:
        must_conditions = [
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id),
            ),
            FieldCondition(
                key="owner_user_id",
                match=MatchValue(value=owner_user_id),
            ),
            FieldCondition(
                key="knowledge_base_id",
                match=MatchValue(value=knowledge_base_id),
            ),
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id),
            ),
        ]

        if chunk_type is not None:
            must_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=chunk_type),
                )
            )

        return Filter(
            must=must_conditions,
        )

    @staticmethod
    def _chunk_to_payload(
        *,
        chunk: RagChunk,
        ingested_at: str,
    ) -> dict[str, Any]:
        chunk_type = chunk.metadata.get("chunk_type")

        return {
            "chunk_id": chunk.chunk_id,
            "parent_id": chunk.parent_id,
            "document_id": chunk.document_id,
            "tenant_id": chunk.tenant_id,
            "owner_user_id": chunk.owner_user_id,
            "knowledge_base_id": chunk.knowledge_base_id,
            "visibility": chunk.visibility,
            "file_name": chunk.file_name,
            "file_sha256": chunk.metadata.get("file_sha256"),
            "source_type": chunk.metadata.get("source_type"),
            "document_version": chunk.metadata.get("document_version"),
            "chunk_type": chunk_type,
            "parent_index": chunk.metadata.get("parent_index"),
            "child_index": chunk.metadata.get("child_index"),
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_path": chunk.section_path,
            "text": chunk.text,
            "token_count_estimate": chunk.token_count_estimate,
            "metadata": chunk.metadata,
            "ingested_at": ingested_at,
        }
