from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


PERSONAL_DATA_VERSION = "stage_4_4_lite"


class ShortMemoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(default_factory=lambda: uuid4().hex)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8_000)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class LongTermFactRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    tenant_id: str
    user_id: str
    fact_type: str
    fact_key: str
    fact_value: dict[str, Any]
    confidence: float
    source_thread_id: str | None = None
    source_message_id: str | None = None
    status: Literal["active", "superseded", "deleted"] = "active"
    version: int = 1
    is_user_confirmed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class RagDocumentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    document_id: str
    tenant_id: str
    owner_user_id: str
    knowledge_base_id: str
    title: str
    source: str
    version: str = "1"
    effective_date: str | None = None
    expired_date: str | None = None
    content_hash: str
    status: Literal[
        "processing", "active", "failed", "disabled", "deleted", "superseded"
    ]
    file_name: str | None = None
    stored_path: str | None = None
    content_text: str | None = None
    parent_count: int = 0
    child_count: int = 0
    point_count: int = 0
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class PersonalQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = PERSONAL_DATA_VERSION
    short_memory_ok: bool
    long_memory_ok: bool
    rag_metadata_ok: bool
    qdrant_ok: bool | None = None
    checks: dict[str, Any] = Field(default_factory=dict)
