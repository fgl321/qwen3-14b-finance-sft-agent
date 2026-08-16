from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DocumentMeta(BaseModel):
    document_id: str
    file_name: str
    title: str = ""
    aliases: list[str] = Field(default_factory=list)
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
    score_display: str | None = None

    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)

    # 检索调试信息。
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagEvidenceAssessment(BaseModel):
    sufficient: bool
    support_level: Literal[
        "direct_support",
        "partial_support",
        "background_support",
        "irrelevant",
    ] = "irrelevant"
    confidence: str = "low"
    reason: str = ""
    relevant_evidence_numbers: list[int] = Field(default_factory=list)
    direct_evidence_numbers: list[int] = Field(default_factory=list)
    partial_evidence_numbers: list[int] = Field(default_factory=list)
    background_evidence_numbers: list[int] = Field(default_factory=list)
    evidence_claims: list["EvidenceClaim"] = Field(default_factory=list)
    claim_relations: list["EvidenceClaimRelation"] = Field(default_factory=list)
    evidence_conflicts: list["EvidenceConflict"] = Field(default_factory=list)

    # 保持旧字段名，避免 rag_service.py 里已有代码报错。
    missing_info: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_support_level(self) -> "RagEvidenceAssessment":
        if self.sufficient and self.support_level == "irrelevant":
            self.support_level = "direct_support"
        if self.sufficient and not self.direct_evidence_numbers:
            self.direct_evidence_numbers = list(self.relevant_evidence_numbers)
        if not self.relevant_evidence_numbers:
            self.relevant_evidence_numbers = sorted(
                {
                    *self.direct_evidence_numbers,
                    *self.partial_evidence_numbers,
                    *self.background_evidence_numbers,
                }
            )
        return self


class LogicalEvidenceRequirement(BaseModel):
    """A logical evidence requirement is the semantic contract of the RAG
    pipeline.  It is created once from the router's task requirements and must
    never disappear due to query dedup/merge/retry optimizations.
    """

    id: str
    task_id: str
    description: str
    required: bool = True


class SourceAuthorityContract(BaseModel):
    """Correctness contract: which information sources may be used."""

    current_user_facts: Literal["allowed", "forbidden"] = "allowed"
    selected_documents: Literal["allowed", "forbidden"] = "allowed"
    deterministic_derivation: Literal["allowed", "forbidden"] = "allowed"
    memory: Literal["allowed", "forbidden"] = "allowed"
    general_model_knowledge: Literal["allowed", "forbidden"] = "allowed"
    domain_heuristics: Literal["allowed", "forbidden"] = "allowed"
    web: Literal["allowed", "forbidden"] = "forbidden"


class PhysicalRetrievalQuery(BaseModel):
    """One physical query may serve many logical evidence requirements.

    Query optimization can reduce physical query count, but it must preserve
    provenance: ``source_requirement_ids`` is the full set of logical
    requirements this query was executed for.
    """

    id: str
    query: str
    source_requirement_ids: list[str] = Field(default_factory=list)
    merged_from_query_ids: list[str] = Field(default_factory=list)


class RequirementObservation(BaseModel):
    """Per-logical-requirement assessment result (the acceptance record)."""

    requirement_id: str
    task_id: str
    status: Literal[
        "direct_support",
        "partial_support",
        "background_support",
        "insufficient_evidence",
        "irrelevant",
        "conflict",
        "technical_unavailable",
        "not_observed",
    ]
    source_query_ids: list[str] = Field(default_factory=list)
    citation_ids: list[int] = Field(default_factory=list)
    conflict_ids: list[int] = Field(default_factory=list)
    assessor_status: str | None = None
    reason: str | None = None


EVIDENCE_STATUS_USER_TEXT: dict[str, str] = {
    "direct_support": "根据文档……",
    "partial_support": "文档部分支持……",
    "background_support": "文档提供背景，但不足以直接确认……",
    "insufficient_evidence": "当前上传文档无法确认……",
    "not_observed": "本次检索未覆盖该要求……",
    "technical_unavailable": "检索服务技术异常……",
    "conflict": "检索证据存在冲突……",
    "assessment_protocol_failed": "本轮证据验证未成功完成……",
}


class EvidenceClaim(BaseModel):
    """Atomic document fact extracted by the semantic evidence judge."""

    claim_id: str
    subject: str
    attribute: str
    value: str
    canonical_value: str | None = None
    qualifier: str | None = None
    unit: str | None = None
    evidence_number: int = Field(ge=1)
    source_type: Literal[
        "regulation_text", "official_explanation", "table", "book_text", "example"
    ] = "book_text"
    support_level: Literal["direct", "partial", "background"] = "direct"
    value_semantics: Literal["scalar", "set_member", "range", "boolean"] = "scalar"
    scope: str | None = None


class EvidenceClaimRelation(BaseModel):
    claim_a_id: str
    claim_b_id: str
    relation: Literal[
        "equivalent", "compatible", "refinement", "contradiction", "incomparable"
    ]
    explanation: str = ""
    conflict_type: Literal[
        "scalar_value_conflict",
        "rule_conflict",
        "scope_conflict",
        "temporal_conflict",
        "definition_conflict",
    ] | None = None


class EvidenceConflict(BaseModel):
    conflict_id: str
    requirement_id: str = "retrieval_requirement"
    subject: str
    attribute: str
    unit: str | None = None
    values: list[str] = Field(min_length=2)
    evidence_numbers: list[int] = Field(min_length=2)
    claim_ids: list[str] = Field(min_length=2)
    relation: Literal["contradiction"] = "contradiction"
    conflict_type: Literal[
        "scalar_value_conflict",
        "rule_conflict",
        "scope_conflict",
        "temporal_conflict",
        "definition_conflict",
    ] = "scalar_value_conflict"
    severity: Literal["blocking", "non_blocking"] = "blocking"
    retryable: bool = False
    explanation: str = ""
    unresolved: bool = True


class CitationScore(BaseModel):
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score_raw: float | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    evidence_confidence: float | None = None
    display_score: float | None = None
    display_score_source: Literal["retrieval", "reranker", "evidence"] = "retrieval"


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
    scores: CitationScore = Field(default_factory=CitationScore)

    # 检索调试信息。
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagStageStatus(BaseModel):
    retrieval_status: Literal["not_run", "completed", "failed"] = "not_run"
    rerank_status: Literal["not_run", "completed", "degraded", "failed"] = "not_run"
    evidence_assessment_status: Literal[
        "not_run", "completed", "repaired", "protocol_failed", "service_failed"
    ] = "not_run"
    conflict_detection_status: Literal[
        "not_run", "completed", "degraded"
    ] = "not_run"
    retrieved_count: int = Field(default=0, ge=0)
    reranked_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    protocol_error_stage: str | None = None


class RagAnswerResult(BaseModel):
    query: str | None = None
    answer: str
    evidence_assessment: RagEvidenceAssessment
    citations: list[RagCitation] = Field(default_factory=list)
    # Candidate passages retained for observability after assessor degradation.
    # They are not verified citations and must never satisfy a citation contract.
    provisional_citations: list[RagCitation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    stage_status: RagStageStatus = Field(default_factory=RagStageStatus)
    usage: dict[str, Any] = Field(default_factory=dict)
