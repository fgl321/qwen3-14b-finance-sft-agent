from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from app.personal_data.privacy import sanitize_personal_value
from app.rag.document_parser import normalize_document_text, parse_document


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    parent_chunk_id: str | None
    chunk_type: str
    text: str
    position: int


class RagDocumentLifecycleService:
    """RAG 文档元数据、版本、禁用、删除、重建与 Qdrant 入库服务。"""

    def __init__(
        self,
        *,
        postgres_dsn: str | None = None,
        settings: Any | None = None,
        rag_store: Any | None = None,
        embedding_provider: Any | None = None,
        upload_dir: str | Path = "data/uploads",
    ) -> None:
        if settings is None and postgres_dsn is None:
            try:
                from app.core.config import get_settings

                settings = get_settings()
            except Exception:
                settings = None
        self.settings = settings
        self.postgres_dsn = postgres_dsn or getattr(
            settings,
            "postgres_dsn",
            "postgresql://agent:agent@127.0.0.1:5432/agent",
        )
        self.rag_store = rag_store
        self.embedding_provider = embedding_provider
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.qdrant_url = getattr(settings, "qdrant_url", "http://127.0.0.1:6333")
        self.collection_name = getattr(
            settings, "qdrant_collection", "finance_knowledge"
        )
        self.dense_vector_name = getattr(
            settings, "rag_dense_vector_name", "dense"
        )
        self.sparse_vector_name = getattr(
            settings, "rag_sparse_vector_name", "sparse"
        )

        # 优先复用项目已有 QdrantRagStore 的真实配置，避免生命周期
        # 服务和检索服务写入不同集合或使用不同向量名称。
        if rag_store is not None:
            self.collection_name = (
                getattr(rag_store, "collection_name", None)
                or getattr(rag_store, "_collection_name", None)
                or self.collection_name
            )
            self.dense_vector_name = (
                getattr(rag_store, "dense_vector_name", None)
                or getattr(rag_store, "_dense_vector_name", None)
                or self.dense_vector_name
            )
            self.sparse_vector_name = (
                getattr(rag_store, "sparse_vector_name", None)
                or getattr(rag_store, "_sparse_vector_name", None)
                or self.sparse_vector_name
            )

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "缺少 psycopg，请执行 python -m pip install 'psycopg[binary]'。"
            ) from exc
        return psycopg.connect(self.postgres_dsn, row_factory=dict_row)

    @staticmethod
    def _json_value(value: Any) -> Any:
        try:
            from psycopg.types.json import Jsonb

            return Jsonb(value)
        except Exception:  # pragma: no cover
            return json.dumps(value, ensure_ascii=False)

    def init_schema(self) -> None:
        sql = """
        CREATE TABLE IF NOT EXISTS rag_documents (
            id BIGSERIAL PRIMARY KEY,
            document_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL DEFAULT 'default',
            owner_user_id TEXT NOT NULL,
            knowledge_base_id TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'personal_upload',
            version TEXT NOT NULL DEFAULT '1',
            effective_date DATE,
            expired_date DATE,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'processing',
            file_name TEXT,
            stored_path TEXT,
            content_text TEXT,
            parent_count INTEGER NOT NULL DEFAULT 0,
            child_count INTEGER NOT NULL DEFAULT 0,
            point_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT rag_documents_scope_id_unique
                UNIQUE (tenant_id, owner_user_id, knowledge_base_id, document_id)
        );
        CREATE INDEX IF NOT EXISTS idx_rag_documents_scope
            ON rag_documents
            (tenant_id, owner_user_id, knowledge_base_id, status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_rag_documents_content_hash
            ON rag_documents
            (tenant_id, owner_user_id, knowledge_base_id, content_hash);
        CREATE INDEX IF NOT EXISTS idx_rag_documents_title_source
            ON rag_documents
            (tenant_id, owner_user_id, knowledge_base_id, title, source);
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(
            normalize_document_text(text).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _stable_document_id(
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        source: str,
        title: str,
        content_hash: str,
    ) -> str:
        raw = "|".join(
            [
                tenant_id,
                owner_user_id,
                knowledge_base_id,
                source,
                title,
                content_hash,
            ]
        )
        return str(uuid5(NAMESPACE_URL, raw))

    @staticmethod
    def _split_windows(text: str, size: int, overlap: int) -> list[str]:
        clean = normalize_document_text(text)
        if len(clean) <= size:
            return [clean]
        chunks: list[str] = []
        start = 0
        while start < len(clean):
            end = min(start + size, len(clean))
            if end < len(clean):
                candidates = [
                    clean.rfind("\n", start + size // 2, end),
                    clean.rfind("。", start + size // 2, end),
                    clean.rfind("；", start + size // 2, end),
                ]
                boundary = max(candidates)
                if boundary > start:
                    end = boundary + 1
            part = clean[start:end].strip()
            if part:
                chunks.append(part)
            if end >= len(clean):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def chunk_text(
        self,
        *,
        document_id: str,
        text: str,
        parent_size: int = 1_800,
        parent_overlap: int = 200,
        child_size: int = 500,
        child_overlap: int = 80,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        parents = self._split_windows(text, parent_size, parent_overlap)
        global_position = 0
        for parent_index, parent_text in enumerate(parents):
            parent_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{document_id}:parent:{parent_index}",
                )
            )
            chunks.append(
                TextChunk(
                    chunk_id=parent_id,
                    parent_chunk_id=None,
                    chunk_type="parent",
                    text=parent_text,
                    position=global_position,
                )
            )
            global_position += 1
            for child_index, child_text in enumerate(
                self._split_windows(parent_text, child_size, child_overlap)
            ):
                chunks.append(
                    TextChunk(
                        chunk_id=str(
                            uuid5(
                                NAMESPACE_URL,
                                (
                                    f"{document_id}:parent:{parent_index}:"
                                    f"child:{child_index}"
                                ),
                            )
                        ),
                        parent_chunk_id=parent_id,
                        chunk_type="child",
                        text=child_text,
                        position=global_position,
                    )
                )
                global_position += 1
        return chunks

    def _get_qdrant_client(self) -> Any:
        if self.rag_store is not None:
            for attr in ("client", "qdrant_client", "_client"):
                client = getattr(self.rag_store, attr, None)
                if client is not None:
                    return client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "缺少 qdrant-client：python -m pip install qdrant-client"
            ) from exc
        return QdrantClient(url=self.qdrant_url, timeout=60)

    @staticmethod
    def _embedding_parts(embedding: Any) -> tuple[list[float], list[int], list[float]]:
        if isinstance(embedding, dict):
            dense = embedding.get("dense") or embedding.get("dense_vector") or []
            sparse = embedding.get("sparse") or {}
            indices = sparse.get("indices") or embedding.get("sparse_indices") or []
            values = sparse.get("values") or embedding.get("sparse_values") or []
        else:
            dense = getattr(embedding, "dense", None) or getattr(
                embedding, "dense_vector", []
            )
            sparse = getattr(embedding, "sparse", None)
            indices = getattr(sparse, "indices", []) if sparse is not None else []
            values = getattr(sparse, "values", []) if sparse is not None else []
        return list(dense), list(indices), list(values)

    def _embed(self, text: str) -> tuple[list[float], list[int], list[float]]:
        if self.embedding_provider is None:
            raise RuntimeError("embedding_provider 尚未初始化。")
        if hasattr(self.embedding_provider, "embed_query"):
            result = self.embedding_provider.embed_query(text)
        elif hasattr(self.embedding_provider, "embed_documents"):
            result = self.embedding_provider.embed_documents([text])[0]
        else:
            raise RuntimeError("Embedding 服务缺少 embed_query/embed_documents。")
        return self._embedding_parts(result)

    def _ensure_collection(self, *, client: Any, dense_size: int) -> None:
        try:
            client.get_collection(self.collection_name)
            return
        except Exception:
            pass
        from qdrant_client.http import models

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.dense_vector_name: models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.sparse_vector_name: models.SparseVectorParams()
            },
        )

    def _delete_points(self, *, document_id: str) -> int:
        client = self._get_qdrant_client()
        from qdrant_client.http import models

        filter_ = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )
        try:
            count_result = client.count(
                collection_name=self.collection_name,
                count_filter=filter_,
                exact=True,
            )
            count = int(getattr(count_result, "count", 0))
        except Exception:
            count = 0
        try:
            client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_),
                wait=True,
            )
        except Exception as exc:
            # 集合尚不存在时视为没有可删除向量。
            if "not found" not in str(exc).lower():
                raise
        return count

    def _set_payload_status(self, *, document_id: str, status: str) -> None:
        client = self._get_qdrant_client()
        from qdrant_client.http import models

        client.set_payload(
            collection_name=self.collection_name,
            payload={"status": status},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            wait=True,
        )

    def _upsert_chunks(
        self,
        *,
        document: dict[str, Any],
        chunks: list[TextChunk],
    ) -> int:
        client = self._get_qdrant_client()
        from qdrant_client.http import models

        points: list[Any] = []
        dense_size = 0
        for chunk in chunks:
            dense, indices, values = self._embed(chunk.text)
            if not dense:
                raise RuntimeError("Embedding 返回了空稠密向量。")
            dense_size = dense_size or len(dense)
            # chunk_id 本身就是稳定 UUID。point ID、父子引用和最终引用
            # 使用同一标识，兼容现有 parent/child 回填检索器。
            point_id = chunk.chunk_id
            citation_id = hashlib.sha1(
                chunk.chunk_id.encode("utf-8")
            ).hexdigest()[:16]
            now_iso = datetime.now(timezone.utc).isoformat()
            file_name = document.get("file_name") or f"{document['title']}.txt"
            source_type = (
                Path(str(file_name)).suffix.lower().lstrip(".") or "text"
            )
            parent_reference = (
                chunk.parent_chunk_id
                if chunk.chunk_type == "child"
                else None
            )
            payload = {
                "tenant_id": document["tenant_id"],
                "owner_user_id": document["owner_user_id"],
                "knowledge_base_id": document["knowledge_base_id"],
                "document_id": document["document_id"],
                "document_title": document["title"],
                "title": document["title"],
                "source": document["source"],
                "version": document["version"],
                "document_version": document["version"],
                "status": "active",
                "visibility": "private",
                "source_type": source_type,
                "file_name": file_name,
                "file_sha256": document.get("content_hash"),
                "content_hash": document.get("content_hash"),
                "ingested_at": now_iso,
                "effective_date": document.get("effective_date"),
                "expired_date": document.get("expired_date"),
                "page_start": 1,
                "page_end": 1,
                "chunk_id": chunk.chunk_id,
                "parent_chunk_id": parent_reference,
                "parent_id": parent_reference,
                "chunk_type": chunk.chunk_type,
                "chunk_level": chunk.chunk_type,
                "is_parent": chunk.chunk_type == "parent",
                "content": chunk.text,
                "text": chunk.text,
                "text_content": chunk.text,
                "parent_text": (
                    chunk.text if chunk.chunk_type == "parent" else None
                ),
                "child_text": (
                    chunk.text if chunk.chunk_type == "child" else None
                ),
                "position": chunk.position,
                "citation_id": citation_id,
                "metadata": {
                    "document_title": document["title"],
                    "source": document["source"],
                    "version": document["version"],
                    "status": "active",
                },
            }
            vectors: dict[str, Any] = {self.dense_vector_name: dense}
            if indices and values:
                vectors[self.sparse_vector_name] = models.SparseVector(
                    indices=indices,
                    values=values,
                )
            points.append(
                models.PointStruct(id=point_id, vector=vectors, payload=payload)
            )

        self._ensure_collection(client=client, dense_size=dense_size)
        batch_size = 64
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )
        return len(points)

    @staticmethod
    def _row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        return dict(row) if row else None

    def get_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        include_content: bool = False,
    ) -> dict[str, Any] | None:
        content_field = "content_text," if include_content else "NULL AS content_text,"
        sql = f"""
        SELECT id, document_id, tenant_id, owner_user_id, knowledge_base_id,
               title, source, version, effective_date::text AS effective_date,
               expired_date::text AS expired_date, content_hash, status,
               file_name, stored_path, {content_field}
               parent_count, child_count, point_count, error_message, metadata,
               created_at::text AS created_at, updated_at::text AS updated_at
        FROM rag_documents
        WHERE tenant_id=%(tenant_id)s AND owner_user_id=%(owner_user_id)s
          AND knowledge_base_id=%(knowledge_base_id)s
          AND document_id=%(document_id)s
        LIMIT 1;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    {
                        "tenant_id": tenant_id,
                        "owner_user_id": owner_user_id,
                        "knowledge_base_id": knowledge_base_id,
                        "document_id": document_id,
                    },
                )
                return self._row(cur.fetchone())

    def list_documents(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        status_clause = " AND status=%(status)s" if status else ""
        sql = f"""
        SELECT id, document_id, tenant_id, owner_user_id, knowledge_base_id,
               title, source, version, effective_date::text AS effective_date,
               expired_date::text AS expired_date, content_hash, status,
               file_name, stored_path, NULL AS content_text,
               parent_count, child_count, point_count, error_message, metadata,
               created_at::text AS created_at, updated_at::text AS updated_at
        FROM rag_documents
        WHERE tenant_id=%(tenant_id)s AND owner_user_id=%(owner_user_id)s
          AND knowledge_base_id=%(knowledge_base_id)s
          {status_clause}
        ORDER BY updated_at DESC, id DESC
        LIMIT %(limit)s;
        """
        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "owner_user_id": owner_user_id,
            "knowledge_base_id": knowledge_base_id,
            "limit": min(max(int(limit), 1), 500),
        }
        if status:
            params["status"] = status
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]

    def _find_duplicate(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        content_hash: str,
    ) -> dict[str, Any] | None:
        sql = """
        SELECT id, document_id, tenant_id, owner_user_id, knowledge_base_id,
               title, source, version, effective_date::text AS effective_date,
               expired_date::text AS expired_date, content_hash, status,
               file_name, stored_path, NULL AS content_text,
               parent_count, child_count, point_count, error_message, metadata,
               created_at::text AS created_at, updated_at::text AS updated_at
        FROM rag_documents
        WHERE tenant_id=%(tenant_id)s AND owner_user_id=%(owner_user_id)s
          AND knowledge_base_id=%(knowledge_base_id)s
          AND content_hash=%(content_hash)s AND status='active'
        ORDER BY id DESC LIMIT 1;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    {
                        "tenant_id": tenant_id,
                        "owner_user_id": owner_user_id,
                        "knowledge_base_id": knowledge_base_id,
                        "content_hash": content_hash,
                    },
                )
                return self._row(cur.fetchone())

    def _supersede_previous_versions(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        title: str,
        source: str,
        exclude_document_id: str,
    ) -> list[str]:
        params = {
            "tenant_id": tenant_id,
            "owner_user_id": owner_user_id,
            "knowledge_base_id": knowledge_base_id,
            "title": title,
            "source": source,
            "exclude_document_id": exclude_document_id,
        }
        # 先读取候选并删除旧向量；全部成功后才修改 PostgreSQL 状态。
        # 这样 Qdrant 删除失败时，旧版本仍保持 active，不会造成知识空窗。
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document_id FROM rag_documents
                    WHERE tenant_id=%(tenant_id)s
                      AND owner_user_id=%(owner_user_id)s
                      AND knowledge_base_id=%(knowledge_base_id)s
                      AND title=%(title)s AND source=%(source)s
                      AND document_id<>%(exclude_document_id)s
                      AND status='active'
                    ORDER BY id;
                    """,
                    params,
                )
                ids = [str(row["document_id"]) for row in cur.fetchall()]

        for old_id in ids:
            self._delete_points(document_id=old_id)

        if ids:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE rag_documents
                        SET status='superseded', point_count=0, updated_at=NOW()
                        WHERE tenant_id=%(tenant_id)s
                          AND owner_user_id=%(owner_user_id)s
                          AND knowledge_base_id=%(knowledge_base_id)s
                          AND title=%(title)s AND source=%(source)s
                          AND document_id<>%(exclude_document_id)s
                          AND status='active';
                        """,
                        params,
                    )
                conn.commit()
        return ids

    def ingest_text(
        self,
        *,
        text: str,
        title: str,
        tenant_id: str = "default",
        owner_user_id: str,
        knowledge_base_id: str = "kb_finance_basic",
        source: str = "personal_upload",
        version: str = "1",
        effective_date: str | date | None = None,
        expired_date: str | date | None = None,
        file_name: str | None = None,
        stored_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
        replace_same_title: bool = True,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        clean_text = normalize_document_text(text)
        if len(clean_text) < 10:
            raise ValueError("文档内容过短，至少需要 10 个字符。")
        clean_title = str(title).strip()
        if not clean_title:
            raise ValueError("title 不能为空。")
        clean_owner = str(owner_user_id).strip()
        if not clean_owner:
            raise ValueError("owner_user_id 不能为空。")
        clean_tenant = str(tenant_id).strip() or "default"
        clean_kb = str(knowledge_base_id).strip() or "kb_finance_basic"
        clean_source = str(source).strip() or "personal_upload"
        hash_ = self.content_hash(clean_text)

        duplicate = None if force_rebuild else self._find_duplicate(
            tenant_id=clean_tenant,
            owner_user_id=clean_owner,
            knowledge_base_id=clean_kb,
            content_hash=hash_,
        )
        if duplicate:
            return {**duplicate, "duplicate": True, "superseded_document_ids": []}

        document_id = self._stable_document_id(
            tenant_id=clean_tenant,
            owner_user_id=clean_owner,
            knowledge_base_id=clean_kb,
            source=clean_source,
            title=clean_title,
            content_hash=hash_,
        )
        clean_metadata = sanitize_personal_value(dict(metadata or {}))
        alias_values = [
            str(item).strip()
            for item in (aliases or [])
            if str(item).strip()
        ]
        if alias_values:
            clean_metadata["aliases"] = list(
                dict.fromkeys(alias_values)
            )
        chunks = self.chunk_text(document_id=document_id, text=clean_text)
        parent_count = sum(c.chunk_type == "parent" for c in chunks)
        child_count = sum(c.chunk_type == "child" for c in chunks)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_documents (
                        document_id, tenant_id, owner_user_id,
                        knowledge_base_id, title, source, version,
                        effective_date, expired_date, content_hash, status,
                        file_name, stored_path, content_text,
                        parent_count, child_count, point_count,
                        error_message, metadata
                    ) VALUES (
                        %(document_id)s, %(tenant_id)s, %(owner_user_id)s,
                        %(knowledge_base_id)s, %(title)s, %(source)s,
                        %(version)s, %(effective_date)s, %(expired_date)s,
                        %(content_hash)s, 'processing', %(file_name)s,
                        %(stored_path)s, %(content_text)s,
                        %(parent_count)s, %(child_count)s, 0, NULL,
                        %(metadata)s
                    )
                    ON CONFLICT (
                        tenant_id, owner_user_id, knowledge_base_id, document_id
                    ) DO UPDATE SET
                        title=EXCLUDED.title,
                        source=EXCLUDED.source,
                        version=EXCLUDED.version,
                        effective_date=EXCLUDED.effective_date,
                        expired_date=EXCLUDED.expired_date,
                        content_text=EXCLUDED.content_text,
                        status='processing',
                        file_name=EXCLUDED.file_name,
                        stored_path=EXCLUDED.stored_path,
                        parent_count=EXCLUDED.parent_count,
                        child_count=EXCLUDED.child_count,
                        point_count=0,
                        error_message=NULL,
                        metadata=EXCLUDED.metadata,
                        updated_at=NOW();
                    """,
                    {
                        "document_id": document_id,
                        "tenant_id": clean_tenant,
                        "owner_user_id": clean_owner,
                        "knowledge_base_id": clean_kb,
                        "title": clean_title,
                        "source": clean_source,
                        "version": str(version or "1"),
                        "effective_date": effective_date or None,
                        "expired_date": expired_date or None,
                        "content_hash": hash_,
                        "file_name": file_name,
                        "stored_path": stored_path,
                        "content_text": clean_text,
                        "parent_count": parent_count,
                        "child_count": child_count,
                        "metadata": self._json_value(clean_metadata),
                    },
                )
            conn.commit()

        document = {
            "document_id": document_id,
            "tenant_id": clean_tenant,
            "owner_user_id": clean_owner,
            "knowledge_base_id": clean_kb,
            "title": clean_title,
            "source": clean_source,
            "version": str(version or "1"),
            "file_name": file_name,
            "content_hash": hash_,
            "effective_date": (
                effective_date.isoformat()
                if isinstance(effective_date, date)
                else effective_date
            ),
            "expired_date": (
                expired_date.isoformat()
                if isinstance(expired_date, date)
                else expired_date
            ),
        }
        superseded: list[str] = []
        try:
            self._delete_points(document_id=document_id)
            point_count = self._upsert_chunks(document=document, chunks=chunks)
            if replace_same_title:
                superseded = self._supersede_previous_versions(
                    tenant_id=clean_tenant,
                    owner_user_id=clean_owner,
                    knowledge_base_id=clean_kb,
                    title=clean_title,
                    source=clean_source,
                    exclude_document_id=document_id,
                )
            status = "active"
            error_message = None
        except Exception as exc:
            # 新版本入库未完整完成时，清理可能已经写入的部分向量。
            # 旧版本在 _supersede_previous_versions 成功前不会被改状态。
            try:
                self._delete_points(document_id=document_id)
            except Exception:
                pass
            point_count = 0
            status = "failed"
            error_message = f"{type(exc).__name__}: {str(exc)[:500]}"

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE rag_documents
                    SET status=%(status)s, point_count=%(point_count)s,
                        error_message=%(error_message)s, updated_at=NOW()
                    WHERE tenant_id=%(tenant_id)s
                      AND owner_user_id=%(owner_user_id)s
                      AND knowledge_base_id=%(knowledge_base_id)s
                      AND document_id=%(document_id)s;
                    """,
                    {
                        **document,
                        "status": status,
                        "point_count": point_count,
                        "error_message": error_message,
                    },
                )
            conn.commit()

        result = self.get_document(
            document_id=document_id,
            tenant_id=clean_tenant,
            owner_user_id=clean_owner,
            knowledge_base_id=clean_kb,
        ) or document
        result.update(
            {
                "duplicate": False,
                "superseded_document_ids": superseded,
            }
        )
        if status == "failed":
            raise RuntimeError(error_message or "RAG 文档入库失败。")
        return result

    def ingest_file(
        self,
        *,
        path: str | Path,
        title: str | None,
        tenant_id: str = "default",
        owner_user_id: str,
        knowledge_base_id: str = "kb_finance_basic",
        source: str = "personal_upload",
        version: str = "1",
        effective_date: str | None = None,
        expired_date: str | None = None,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
        replace_same_title: bool = True,
    ) -> dict[str, Any]:
        file_path = Path(path)
        text = parse_document(file_path)
        return self.ingest_text(
            text=text,
            title=title or file_path.stem,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            source=source,
            version=version,
            effective_date=effective_date,
            expired_date=expired_date,
            file_name=file_path.name,
            stored_path=str(file_path),
            metadata=metadata,
            aliases=aliases,
            replace_same_title=replace_same_title,
        )

    def set_document_enabled(
        self,
        *,
        document_id: str,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        enabled: bool,
    ) -> dict[str, Any]:
        document = self.get_document(
            document_id=document_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            include_content=True,
        )
        if not document:
            raise KeyError("文档不存在。")
        if enabled:
            if not document.get("content_text"):
                raise ValueError("文档没有保留可重建文本，无法重新启用。")
            return self.ingest_text(
                text=document["content_text"],
                title=document["title"],
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                knowledge_base_id=knowledge_base_id,
                source=document["source"],
                version=document["version"],
                effective_date=document.get("effective_date"),
                expired_date=document.get("expired_date"),
                file_name=document.get("file_name"),
                stored_path=document.get("stored_path"),
                metadata=document.get("metadata") or {},
                replace_same_title=False,
                force_rebuild=True,
            )
        self._delete_points(document_id=document_id)
        self._update_status(
            document_id=document_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            status="disabled",
            point_count=0,
        )
        return self.get_document(
            document_id=document_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
        ) or {}

    def register_ingested_document(
        self,
        *,
        document_id: str,
        title: str,
        file_name: str,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        content_hash: str,
        parent_count: int = 0,
        child_count: int = 0,
        point_count: int = 0,
        stored_path: str | None = None,
        version: str = "1",
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Register/refresh the PostgreSQL authority row after Qdrant-only
        ingestion (the async upload path writes Qdrant first)."""
        clean_metadata = dict(metadata or {})
        alias_values = [
            str(item).strip()
            for item in (aliases or [])
            if str(item).strip()
        ]
        if alias_values:
            clean_metadata["aliases"] = list(
                dict.fromkeys(alias_values)
            )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_documents (
                        document_id, tenant_id, owner_user_id,
                        knowledge_base_id, title, source, version,
                        effective_date, expired_date, content_hash, status,
                        file_name, stored_path, content_text,
                        parent_count, child_count, point_count,
                        error_message, metadata
                    ) VALUES (
                        %(document_id)s, %(tenant_id)s, %(owner_user_id)s,
                        %(knowledge_base_id)s, %(title)s, 'personal_upload',
                        %(version)s, NULL, NULL, %(content_hash)s, 'active',
                        %(file_name)s, %(stored_path)s, NULL,
                        %(parent_count)s, %(child_count)s, %(point_count)s,
                        NULL, %(metadata)s::jsonb
                    )
                    ON CONFLICT (
                        tenant_id, owner_user_id, knowledge_base_id, document_id
                    ) DO UPDATE SET
                        status='active',
                        title=EXCLUDED.title,
                        file_name=EXCLUDED.file_name,
                        stored_path=EXCLUDED.stored_path,
                        content_hash=EXCLUDED.content_hash,
                        parent_count=EXCLUDED.parent_count,
                        child_count=EXCLUDED.child_count,
                        point_count=EXCLUDED.point_count,
                        metadata=EXCLUDED.metadata,
                        updated_at=NOW()
                    """,
                    {
                        "document_id": str(document_id),
                        "tenant_id": str(tenant_id),
                        "owner_user_id": str(owner_user_id),
                        "knowledge_base_id": str(knowledge_base_id),
                        "title": str(title),
                        "version": str(version or "1"),
                        "content_hash": str(content_hash or ""),
                        "file_name": str(file_name),
                        "stored_path": stored_path,
                        "parent_count": int(parent_count or 0),
                        "child_count": int(child_count or 0),
                        "point_count": int(point_count or 0),
                        "metadata": self._json_value(
                            clean_metadata
                        ),
                    },
                )
            conn.commit()

    def _update_status(
        self,
        *,
        document_id: str,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        status: str,
        point_count: int | None = None,
    ) -> None:
        point_clause = ", point_count=%(point_count)s" if point_count is not None else ""
        sql = f"""
        UPDATE rag_documents SET status=%(status)s {point_clause}, updated_at=NOW()
        WHERE tenant_id=%(tenant_id)s AND owner_user_id=%(owner_user_id)s
          AND knowledge_base_id=%(knowledge_base_id)s
          AND document_id=%(document_id)s;
        """
        params = {
            "document_id": document_id,
            "tenant_id": tenant_id,
            "owner_user_id": owner_user_id,
            "knowledge_base_id": knowledge_base_id,
            "status": status,
            "point_count": point_count,
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()

    def delete_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
        hard_delete: bool = False,
    ) -> dict[str, Any]:
        document = self.get_document(
            document_id=document_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
        )
        if not document:
            return {"deleted": False, "document_id": document_id, "point_count": 0}
        point_count = self._delete_points(document_id=document_id)
        with self._connect() as conn:
            with conn.cursor() as cur:
                if hard_delete:
                    cur.execute(
                        """
                        DELETE FROM rag_documents
                        WHERE tenant_id=%(tenant_id)s
                          AND owner_user_id=%(owner_user_id)s
                          AND knowledge_base_id=%(knowledge_base_id)s
                          AND document_id=%(document_id)s;
                        """,
                        {
                            "tenant_id": tenant_id,
                            "owner_user_id": owner_user_id,
                            "knowledge_base_id": knowledge_base_id,
                            "document_id": document_id,
                        },
                    )
                else:
                    cur.execute(
                        """
                        UPDATE rag_documents
                        SET status='deleted', point_count=0, updated_at=NOW()
                        WHERE tenant_id=%(tenant_id)s
                          AND owner_user_id=%(owner_user_id)s
                          AND knowledge_base_id=%(knowledge_base_id)s
                          AND document_id=%(document_id)s;
                        """,
                        {
                            "tenant_id": tenant_id,
                            "owner_user_id": owner_user_id,
                            "knowledge_base_id": knowledge_base_id,
                            "document_id": document_id,
                        },
                    )
            conn.commit()
        return {
            "deleted": True,
            "hard_delete": hard_delete,
            "document_id": document_id,
            "point_count": point_count,
        }

    def sync_index_status(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        """Reconcile Postgres lifecycle status with Qdrant index reality.

        ``active`` must mean metadata ready AND index ready.  A document whose
        vectors disappeared from Qdrant is downgraded to ``index_degraded`` so
        it can never be advertised as a normally searchable document.
        """

        store = self.rag_store
        if store is None:
            raise RuntimeError("rag_store 尚未初始化，无法对账索引状态")
        checked = 0
        changed: list[dict[str, Any]] = []
        for status in ("active", "index_degraded"):
            rows = self.list_documents(
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                knowledge_base_id=knowledge_base_id,
                status=status,
                limit=500,
            )
            for row in rows:
                document_id = str(row.get("document_id") or "")
                if not document_id:
                    continue
                checked += 1
                try:
                    count = int(
                        store.count_document_chunks(
                            tenant_id=tenant_id,
                            owner_user_id=owner_user_id,
                            knowledge_base_id=knowledge_base_id,
                            document_id=document_id,
                        )
                        or 0
                    )
                except Exception:
                    count = 0
                stored_count = int(row.get("point_count") or 0)
                if status == "active" and count == 0:
                    self._update_status(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        owner_user_id=owner_user_id,
                        knowledge_base_id=knowledge_base_id,
                        status="index_degraded",
                        point_count=0,
                    )
                    changed.append(
                        {
                            "document_id": document_id,
                            "from": "active",
                            "to": "index_degraded",
                        }
                    )
                elif status == "index_degraded" and count > 0:
                    self._update_status(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        owner_user_id=owner_user_id,
                        knowledge_base_id=knowledge_base_id,
                        status="active",
                        point_count=count,
                    )
                    changed.append(
                        {
                            "document_id": document_id,
                            "from": "index_degraded",
                            "to": "active",
                        }
                    )
                elif status == "active" and count > 0 and stored_count != count:
                    self._update_status(
                        document_id=document_id,
                        tenant_id=tenant_id,
                        owner_user_id=owner_user_id,
                        knowledge_base_id=knowledge_base_id,
                        status="active",
                        point_count=count,
                    )
                    changed.append(
                        {
                            "document_id": document_id,
                            "from": f"active@{stored_count}",
                            "to": f"active@{count}",
                        }
                    )
        return {"checked": checked, "changed": changed}

    def rebuild_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        document = self.get_document(
            document_id=document_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            include_content=True,
        )
        if not document:
            raise KeyError("文档不存在。")
        if not document.get("content_text"):
            raise ValueError("文档没有可用于重建的文本。")
        return self.ingest_text(
            text=document["content_text"],
            title=document["title"],
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            source=document["source"],
            version=document["version"],
            effective_date=document.get("effective_date"),
            expired_date=document.get("expired_date"),
            file_name=document.get("file_name"),
            stored_path=document.get("stored_path"),
            metadata=document.get("metadata") or {},
            replace_same_title=False,
            force_rebuild=True,
        )

    def answer_with_evidence(
        self,
        *,
        query: str,
        tenant_id: str,
        owner_user_id: str,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        if self.rag_store is None:
            raise RuntimeError("rag_store 尚未初始化。")
        # 优先复用现有 RagAnswerService；本类只负责生命周期。
        raise RuntimeError(
            "请从 API 路由使用 app.state.rag_service.answer() 执行带引用问答。"
        )
