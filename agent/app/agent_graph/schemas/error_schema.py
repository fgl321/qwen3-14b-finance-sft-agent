from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


AgentErrorCategory = Literal[
    "validation",
    "conflict",
    "permission",
    "timeout",
    "dependency",
    "protocol",
    "budget",
    "unavailable",
    "tool",
    "internal",
]

AgentErrorStage = Literal[
    "api",
    "service",
    "graph",
    "agent_loop",
    "planner",
    "reviewer",
    "tool",
    "final_response",
    "synthesis",
    "output_guard",
    "idempotency",
]


def _new_error_id() -> str:
    return f"err_{uuid4().hex}"


class AgentErrorEnvelope(BaseModel):
    """
    跨 Service、Graph、Planner、Tool 和 HTTP 的统一错误结构。

    message 只保存可安全返回给客户端的描述；原始异常堆栈只进入日志，
    不进入该模型或 LangGraph Checkpoint。
    """

    model_config = ConfigDict(extra="forbid")

    error_id: str = Field(
        default_factory=_new_error_id,
        min_length=1,
        max_length=80,
    )

    code: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Z][A-Z0-9_]*$",
    )

    category: AgentErrorCategory
    stage: AgentErrorStage

    message: str = Field(
        min_length=1,
        max_length=1000,
    )

    retryable: bool = False

    http_status: int = Field(
        default=500,
        ge=400,
        le=599,
    )

    request_id: str | None = Field(
        default=None,
        max_length=200,
    )

    run_id: str | None = Field(
        default=None,
        max_length=200,
    )

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )
