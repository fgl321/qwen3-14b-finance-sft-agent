from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentMetaPayload(BaseModel):
    document_id: str
    file_name: str
    file_sha256: str
    tenant_id: str
    owner_user_id: str
    knowledge_base_id: str
    source_type: str
    visibility: str
    version: int


class ChunkStatsPayload(BaseModel):
    total_chunks: int
    parent_count: int
    child_count: int


class DocumentIngestResponse(BaseModel):
    ok: bool
    document: DocumentMetaPayload
    chunks: ChunkStatsPayload
    qdrant: dict
    point_count_after_ingest: int


class KnowledgeDocumentPayload(BaseModel):
    document_id: str
    file_name: str | None = None
    file_sha256: str | None = None
    tenant_id: str | None = None
    owner_user_id: str | None = None
    knowledge_base_id: str | None = None
    visibility: str | None = None
    source_type: str | None = None
    document_version: int | None = None
    ingested_at: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    parent_count: int = 0
    child_count: int = 0
    total_chunks: int = 0


class KnowledgeDocumentListResponse(BaseModel):
    ok: bool = True
    tenant_id: str
    owner_user_id: str
    knowledge_base_id: str
    total: int
    documents: list[KnowledgeDocumentPayload] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    ok: bool
    document_id: str
    deleted_count_estimate: int
    point_count_after_delete: int
    message: str
