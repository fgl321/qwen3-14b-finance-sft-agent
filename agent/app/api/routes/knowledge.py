from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status

from app.api.schemas.knowledge import (
    DocumentDeleteResponse,
    DocumentIngestResponse,
    KnowledgeDocumentListResponse,
)
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.request_boundary import (
    personal_request_identity,
    validate_public_identifier,
    validate_uploaded_document,
)
from app.rag.embeddings import EmbeddingProvider
from app.rag.file_utils import SUPPORTED_DOCUMENT_EXTENSIONS
from app.rag.document_lifecycle import RagDocumentLifecycleService
from app.rag.ingestion_service import RagIngestionService
from app.rag.ingestion_jobs import IngestionJobStore
from app.rag.qdrant_store import QdrantRagStore


logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/knowledge",
    tags=["knowledge"],
)


UPLOAD_DIR = Path("data/uploads")
@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    tenant_id: str = Form(default="tenant_001"),
    owner_user_id: str = Form(default="u001"),
    knowledge_base_id: str = Form(default="kb_finance_basic"),
    visibility: str = Form(default="private"),
) -> DocumentIngestResponse:
    settings: Settings = request.app.state.settings
    embedding_provider: EmbeddingProvider = request.app.state.embedding_provider
    store: QdrantRagStore = request.app.state.rag_store
    request_id = getattr(request.state, "request_id", "unknown")
    identity = personal_request_identity(settings)
    if settings.single_user_mode:
        tenant_id = identity.tenant_id
        owner_user_id = identity.user_id
    knowledge_base_id = validate_public_identifier(
        knowledge_base_id,
        field_name="knowledge_base_id",
    )

    original_name = Path(file.filename or "uploaded_document").name
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "不支持的文件类型。",
                "extension": extension,
                "supported": sorted(SUPPORTED_DOCUMENT_EXTENSIONS),
            },
        )

    if visibility not in {"private", "public"}:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "visibility 只能是 private 或 public。",
                "visibility": visibility,
            },
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"

    try:
        size = await _save_upload_file(
            upload_file=file,
            saved_path=saved_path,
            max_bytes=settings.max_upload_bytes,
        )
        validate_uploaded_document(
            saved_path,
            extension=extension,
            content_type=file.content_type,
        )

        logger.info(
            "knowledge_document_uploaded",
            request_id=request_id,
            original_file_name=original_name,
            saved_path=str(saved_path),
            size=size,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            visibility=visibility,
        )

        ingestion_service = RagIngestionService(
            settings=settings,
            embedding_provider=embedding_provider,
            store=store,
        )

        # 同步 CPU/GPU 密集的解析与向量化放入线程池，
        # 避免阻塞 FastAPI 事件循环导致整个 API 在索引期间不可用。
        result = await asyncio.to_thread(
            ingestion_service.ingest_file,
            file_path=saved_path,
            original_file_name=original_name,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            visibility=visibility,
        )

        _register_ingested_document(
            request=request,
            result=result,
            saved_path=saved_path,
            original_name=original_name,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            visibility=visibility,
        )

        return DocumentIngestResponse.model_validate(result)

    except HTTPException:
        saved_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        logger.exception(
            "knowledge_document_ingest_failed",
            request_id=request_id,
            file_name=original_name,
            error_type=type(exc).__name__,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "文档入库失败。",
                "code": "DOCUMENT_INGESTION_FAILED",
            },
        ) from exc


@router.get(
    "/documents",
    response_model=KnowledgeDocumentListResponse,
)
async def list_documents(
    request: Request,
    tenant_id: str = Query(default="tenant_001"),
    owner_user_id: str = Query(default="u001"),
    knowledge_base_id: str = Query(default="kb_finance_basic"),
    limit: int = Query(default=50, ge=1, le=200),
) -> KnowledgeDocumentListResponse:
    store: QdrantRagStore = request.app.state.rag_store
    settings: Settings = request.app.state.settings
    identity = personal_request_identity(settings)
    if settings.single_user_mode:
        tenant_id = identity.tenant_id
        owner_user_id = identity.user_id
    knowledge_base_id = validate_public_identifier(
        knowledge_base_id,
        field_name="knowledge_base_id",
    )

    lifecycle = getattr(request.app.state, "rag_document_lifecycle", None)
    if lifecycle is None:
        lifecycle = RagDocumentLifecycleService(
            settings=settings,
            rag_store=store,
        )
        lifecycle.init_schema()
        request.app.state.rag_document_lifecycle = lifecycle

    rows = await asyncio.to_thread(
        lifecycle.list_documents,
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        knowledge_base_id=knowledge_base_id,
        limit=limit,
    )
    rows = [
        row
        for row in rows
        if str(row.get("status") or "") not in {"deleted"}
    ]
    source = "postgres"
    if not rows:
        # Legacy fallback: documents ingested before the Postgres registry
        # existed are still visible, but their index status is derived from
        # Qdrant directly.
        qdrant_docs = await asyncio.to_thread(
            store.list_documents,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            limit=limit,
        )
        rows = [
            {
                "document_id": str(doc.get("document_id") or ""),
                "file_name": doc.get("file_name"),
                "title": (
                    doc.get("title")
                    or doc.get("document_title")
                    or doc.get("file_name")
                ),
                "status": "active",
                "version": str(doc.get("document_version") or "1"),
                "content_hash": (
                    doc.get("file_sha256")
                    or doc.get("content_hash")
                    or ""
                ),
                "tenant_id": doc.get("tenant_id"),
                "owner_user_id": doc.get("owner_user_id"),
                "knowledge_base_id": doc.get("knowledge_base_id"),
                "parent_count": int(doc.get("parent_count") or 0),
                "child_count": int(doc.get("child_count") or 0),
                "point_count": int(doc.get("total_chunks") or 0),
                "updated_at": doc.get("ingested_at"),
                "metadata": {
                    "visibility": doc.get("visibility") or "private",
                    "aliases": (
                        doc.get("aliases")
                        or (doc.get("metadata") or {}).get("aliases")
                        or []
                    ),
                },
            }
            for doc in qdrant_docs
            if doc.get("document_id")
        ]
        source = "qdrant_legacy_fallback"

    documents: list[dict[str, Any]] = []
    for row in rows:
        document_id = str(row.get("document_id") or "")
        status = str(row.get("status") or "")
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        aliases = metadata.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [
            str(alias).strip()
            for alias in aliases
            if str(alias).strip()
        ]

        qdrant_chunk_count: int | None = None
        if document_id and status in {"active", "index_degraded"}:
            try:
                qdrant_chunk_count = await asyncio.to_thread(
                    store.count_document_chunks,
                    tenant_id=tenant_id,
                    owner_user_id=owner_user_id,
                    knowledge_base_id=knowledge_base_id,
                    document_id=document_id,
                )
            except Exception:
                qdrant_chunk_count = None

        point_count = (
            qdrant_chunk_count
            if qdrant_chunk_count is not None
            else int(row.get("point_count") or 0)
        )
        if status == "active":
            index_status = "ready" if point_count > 0 else "degraded"
        elif status == "index_degraded":
            index_status = "degraded"
        else:
            index_status = status or None

        documents.append(
            {
                "document_id": document_id,
                "file_name": row.get("file_name"),
                "title": row.get("title"),
                "aliases": aliases,
                "file_sha256": row.get("content_hash"),
                "tenant_id": row.get("tenant_id"),
                "owner_user_id": row.get("owner_user_id"),
                "knowledge_base_id": row.get("knowledge_base_id"),
                "visibility": metadata.get("visibility"),
                "source_type": row.get("source_type"),
                "status": status or None,
                "index_status": index_status,
                "document_version": int(row.get("version") or 1),
                "ingested_at": row.get("updated_at"),
                "parent_count": int(row.get("parent_count") or 0),
                "child_count": int(row.get("child_count") or 0),
                "total_chunks": point_count,
                "error_message": row.get("error_message"),
            }
        )

    documents.sort(
        key=lambda item: str(item.get("ingested_at") or ""),
        reverse=True,
    )

    return KnowledgeDocumentListResponse(
        ok=True,
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        knowledge_base_id=knowledge_base_id,
        total=len(documents),
        documents=documents,
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
)
async def delete_document(
    document_id: str,
    request: Request,
    tenant_id: str = Query(default="tenant_001"),
    owner_user_id: str = Query(default="u001"),
    knowledge_base_id: str = Query(default="kb_finance_basic"),
) -> DocumentDeleteResponse:
    store: QdrantRagStore = request.app.state.rag_store
    settings: Settings = request.app.state.settings
    identity = personal_request_identity(settings)
    if settings.single_user_mode:
        tenant_id = identity.tenant_id
        owner_user_id = identity.user_id
    knowledge_base_id = validate_public_identifier(
        knowledge_base_id,
        field_name="knowledge_base_id",
    )
    request_id = getattr(request.state, "request_id", "unknown")

    result = store.delete_document(
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
    )

    try:
        lifecycle = getattr(request.app.state, "rag_document_lifecycle", None)
        if lifecycle is None:
            lifecycle = RagDocumentLifecycleService(
                settings=settings,
                rag_store=store,
            )
            lifecycle.init_schema()
            request.app.state.rag_document_lifecycle = lifecycle
        lifecycle.delete_document(
            document_id=document_id,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
        )
    except Exception as exc:
        logger.warning(
            "knowledge_postgres_delete_failed",
            document_id=document_id,
            error_type=type(exc).__name__,
        )

    logger.info(
        "knowledge_document_deleted",
        request_id=request_id,
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        deleted_count_estimate=result.get("deleted_count_estimate"),
        point_count_after_delete=result.get("point_count_after_delete"),
    )

    return DocumentDeleteResponse.model_validate(result)


async def _save_upload_file(
    *,
    upload_file: UploadFile,
    saved_path: Path,
    max_bytes: int,
) -> int:
    total_size = 0

    with saved_path.open("wb") as output_file:
        while True:
            chunk = await upload_file.read(1024 * 1024)

            if not chunk:
                break

            total_size += len(chunk)

            if total_size > max_bytes:
                output_file.close()
                saved_path.unlink(missing_ok=True)

                raise HTTPException(
                    status_code=413,
                    detail={
                        "message": "上传文件过大。",
                        "max_upload_mb": max_bytes // 1024 // 1024,
                    },
                )

            output_file.write(chunk)

    await upload_file.close()

    return total_size


async def _execute_ingestion_job(
    *,
    job_id: str,
    request: Request,
    saved_path: Path,
    original_name: str,
    tenant_id: str,
    owner_user_id: str,
    knowledge_base_id: str,
    visibility: str,
) -> None:
    jobs: IngestionJobStore = request.app.state.ingestion_jobs
    jobs.set_processing(job_id)
    try:
        service = RagIngestionService(
            settings=request.app.state.settings,
            embedding_provider=request.app.state.embedding_provider,
            store=request.app.state.rag_store,
        )
        result = await asyncio.to_thread(
            service.ingest_file,
            file_path=saved_path,
            original_file_name=original_name,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            visibility=visibility,
            progress_callback=lambda phase, percent, message: jobs.set_progress(
                job_id,
                phase=phase,
                percent=percent,
                message=message,
            ),
        )
        _register_ingested_document(
            request=request,
            result=result,
            saved_path=saved_path,
            original_name=original_name,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            visibility=visibility,
            job_id=job_id,
        )
        jobs.set_completed(job_id, result)
    except Exception as exc:
        logger.exception(
            "knowledge_async_ingestion_failed",
            job_id=job_id,
            error_type=type(exc).__name__,
        )
        jobs.set_failed(
            job_id,
            code="DOCUMENT_INGESTION_FAILED",
            message="文档解析或索引失败，请检查文件后重试。",
        )


def _register_ingested_document(
    *,
    request: Request,
    result: dict[str, Any],
    saved_path: Path,
    original_name: str,
    tenant_id: str,
    owner_user_id: str,
    knowledge_base_id: str,
    visibility: str,
    job_id: str | None = None,
) -> None:
    """Refresh the PostgreSQL authority row after Qdrant ingestion."""

    try:
        document_meta = result.get("document") or {}
        chunks = result.get("chunks") or {}
        parent_count = int(chunks.get("parent_count") or 0)
        child_count = int(chunks.get("child_count") or 0)
        lifecycle = RagDocumentLifecycleService(
            settings=request.app.state.settings,
            rag_store=request.app.state.rag_store,
            embedding_provider=request.app.state.embedding_provider,
        )
        lifecycle.register_ingested_document(
            document_id=str(document_meta.get("document_id") or ""),
            title=str(
                document_meta.get("title")
                or document_meta.get("file_name")
                or original_name
            ),
            file_name=str(
                document_meta.get("file_name")
                or original_name
            ),
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            content_hash=str(document_meta.get("file_sha256") or ""),
            parent_count=parent_count,
            child_count=child_count,
            point_count=parent_count + child_count,
            stored_path=str(saved_path),
            version=str(document_meta.get("version") or "1"),
            aliases=list(document_meta.get("aliases") or []),
            metadata={"visibility": visibility},
        )
    except Exception as exc:
        # Qdrant ingestion already succeeded; a Postgres authority-row
        # refresh failure must not fail the upload, but it is logged.
        logger.warning(
            "knowledge_postgres_register_failed",
            job_id=job_id,
            error_type=type(exc).__name__,
        )


@router.post("/documents/async", status_code=status.HTTP_202_ACCEPTED)
async def upload_document_async(
    request: Request,
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(default="kb_finance_basic"),
    visibility: str = Form(default="private"),
) -> dict:
    settings: Settings = request.app.state.settings
    identity = personal_request_identity(settings)
    knowledge_base_id = validate_public_identifier(
        knowledge_base_id,
        field_name="knowledge_base_id",
    )
    if visibility != "private":
        raise HTTPException(status_code=400, detail={"message": "个人模式只允许 private 文档。"})
    original_name = Path(file.filename or "uploaded_document").name
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={"message": "不支持的文件类型。", "supported": sorted(SUPPORTED_DOCUMENT_EXTENSIONS)},
        )
    content_type = file.content_type
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    try:
        await _save_upload_file(
            upload_file=file,
            saved_path=saved_path,
            max_bytes=settings.max_upload_bytes,
        )
        validate_uploaded_document(
            saved_path,
            extension=extension,
            content_type=content_type,
        )
        jobs: IngestionJobStore = request.app.state.ingestion_jobs
        job = jobs.create(
            file_name=original_name,
            stored_path=str(saved_path),
            tenant_id=identity.tenant_id,
            owner_user_id=identity.user_id,
            knowledge_base_id=knowledge_base_id,
        )
        task = asyncio.create_task(
            _execute_ingestion_job(
                job_id=job["job_id"],
                request=request,
                saved_path=saved_path,
                original_name=original_name,
                tenant_id=identity.tenant_id,
                owner_user_id=identity.user_id,
                knowledge_base_id=knowledge_base_id,
                visibility=visibility,
            )
        )
        tasks: set = request.app.state.ingestion_tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return job
    except HTTPException:
        saved_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        logger.exception(
            "knowledge_async_upload_failed",
            request_id=getattr(request.state, "request_id", "unknown"),
            file_name=original_name,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "文档上传任务创建失败。",
                "code": "DOCUMENT_JOB_CREATION_FAILED",
            },
        ) from exc


@router.get("/jobs/{job_id}")
async def get_ingestion_job(job_id: str, request: Request) -> dict:
    jobs: IngestionJobStore = request.app.state.ingestion_jobs
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"message": "文档任务不存在。"})
    identity = personal_request_identity(request.app.state.settings)
    if job["tenant_id"] != identity.tenant_id or job["owner_user_id"] != identity.user_id:
        raise HTTPException(status_code=404, detail={"message": "文档任务不存在。"})
    return job
