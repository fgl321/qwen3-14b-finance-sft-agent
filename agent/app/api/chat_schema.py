from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="用户输入的问题",
    )
    user_id: str = Field(
        default="u001",
        description="用户 ID，用于权限过滤和日志追踪",
    )
    thread_id: str | None = Field(
        default=None,
        description="会话 ID",
    )
    tenant_id: str = Field(
        default="tenant_001",
        description="租户 ID",
    )
    knowledge_base_id: str = Field(
        default="kb_finance_basic",
        description="知识库 ID",
    )


class RagCitationPayload(BaseModel):
    citation_id: int
    document_id: str
    file_name: str
    page_start: int | None = None
    page_end: int | None = None
    chunk_id: str
    score: float


class RagToolPayload(BaseModel):
    used: bool = False
    sufficient: bool | None = None
    retrieved_count: int | None = None
    citations: list[RagCitationPayload] = Field(default_factory=list)


class ExecutedToolPayload(BaseModel):
    tool_name: str | None = None
    ok: bool | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    finish_reason: str
    message_count: int
    executed_tools: list[ExecutedToolPayload]
    rag: RagToolPayload
    safety_check: dict[str, Any]
    usage: dict[str, Any]
