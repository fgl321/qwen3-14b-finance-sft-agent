from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    user_id: str = "u001"
    thread_id: str | None = None
    tenant_id: str = "tenant_001"
    knowledge_base_id: str = "kb_finance_basic"


class RagCitationPayload(BaseModel):
    citation_id: int
    document_id: str
    file_name: str
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: str

    # 展示分数：0~100。
    score: float

    # 分数类型和展示字符串。
    score_type: str = "normalized_hybrid_score_0_100"
    score_display: str | None = None

    # 检索调试信息。
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagToolPayload(BaseModel):
    used: bool = False
    sufficient: bool | None = None
    retrieved_count: int | None = None
    citations: list[RagCitationPayload] = Field(default_factory=list)


class ExecutedToolPayload(BaseModel):
    tool_name: str | None = None
    ok: bool | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    usage: Any = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    finish_reason: str
    message_count: int
    executed_tools: list[ExecutedToolPayload] = Field(default_factory=list)
    rag: RagToolPayload = Field(default_factory=RagToolPayload)
    safety_check: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
