from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    document_id: str
    file_name: str
    file_sha256: str
    tenant_id: str
    owner_user_id: str
    knowledge_base_id: str
    source_type: str
    visibility: str
    version: int = 1
    # 知识源治理元数据（见 app/rag/source_classifier.py）
    content_type: str = "unclassified"
    scope: str = "thread"
    trust_level: str = "unverified"
    generated_content: bool = False
    allow_rag_direct: bool = True


class ParsedPage(BaseModel):
    page_number: int
    text: str


class ParsedDocument(BaseModel):
    meta: DocumentMeta
    pages: list[ParsedPage]


class RagChunk(BaseModel):
    chunk_id: str
    parent_id: str | None = None
    document_id: str
    tenant_id: str
    owner_user_id: str
    knowledge_base_id: str
    visibility: str
    file_name: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    token_count_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    text: str

    # 对外展示分数，0~100。
    score: float

    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)

    # 检索调试信息。
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagEvidenceAssessment(BaseModel):
    sufficient: bool
    confidence: str = "low"
    reason: str = ""
    relevant_evidence_numbers: list[int] = Field(default_factory=list)

    # 保持旧字段名，避免 rag_service.py 里已有代码报错。
    missing_info: list[str] = Field(default_factory=list)


# 兼容别的文件可能使用 EvidenceAssessment 这个名字。
EvidenceAssessment = RagEvidenceAssessment


class RagCitation(BaseModel):
    citation_id: int
    document_id: str
    file_name: str
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: str

    # 对外展示分数，0~100。
    score: float

    score_type: str = "normalized_hybrid_score_0_100"
    score_display: str | None = None

    # 检索调试信息。
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagAnswerResult(BaseModel):
    query: str | None = None
    answer: str
    evidence_assessment: RagEvidenceAssessment
    citations: list[RagCitation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
