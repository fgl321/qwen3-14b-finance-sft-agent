from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from app.api.schemas.knowledge import (
    DocumentDeleteResponse,
    DocumentIngestResponse,
    KnowledgeDocumentListResponse,
)
from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingProvider
from app.rag.file_utils import SUPPORTED_DOCUMENT_EXTENSIONS
from app.rag.ingestion_service import RagIngestionService
from app.rag.qdrant_store import QdrantRagStore


logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/knowledge",
    tags=["knowledge"],
)


UPLOAD_DIR = Path("data/uploads")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


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

    saved_path = UPLOAD_DIR / f"{request_id}_{original_name}"

    try:
        size = await _save_upload_file(
            upload_file=file,
            saved_path=saved_path,
            max_bytes=MAX_UPLOAD_BYTES,
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

        result = ingestion_service.ingest_file(
            file_path=saved_path,
            original_file_name=original_name,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            knowledge_base_id=knowledge_base_id,
            visibility=visibility,
        )

        return DocumentIngestResponse.model_validate(result)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "knowledge_document_ingest_failed",
            request_id=request_id,
            file_name=original_name,
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "文档入库失败。",
                "error": str(exc),
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

    documents = store.list_documents(
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        knowledge_base_id=knowledge_base_id,
        limit=limit,
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
    request_id = getattr(request.state, "request_id", "unknown")

    result = store.delete_document(
        tenant_id=tenant_id,
        owner_user_id=owner_user_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
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
