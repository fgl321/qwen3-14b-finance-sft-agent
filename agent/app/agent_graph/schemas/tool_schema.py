from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ToolErrorCode = Literal[
    "TOOL_NOT_FOUND",
    "ARGUMENT_SCHEMA_ERROR",
    "DOMAIN_INPUT_ERROR",
    "TOOL_TIMEOUT",
    "DEPENDENCY_UNAVAILABLE",
    "RATE_LIMITED",
    "PERMISSION_DENIED",
    "SOURCE_AUTHORITY_DENIED",
    "TOOL_INTERNAL_ERROR",
    "AGENT_BUDGET_EXCEEDED",
]

ToolExecutionStatus = Literal[
    "started",
    "succeeded",
    "failed",
    "timed_out",
    "rejected",
]


class ToolErrorInfo(BaseModel):
    """
    统一工具错误。

    model_repairable:
        大模型修改工具名或参数后，可能恢复。

    infrastructure_retryable:
        Python 运行时可在不改变计划的情况下重试。
    """

    model_config = ConfigDict(extra="forbid")

    code: ToolErrorCode

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    model_repairable: bool = False

    infrastructure_retryable: bool = False

    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """
    Tool Executor 的统一返回结构。

    工具异常不能直接冲出 LangGraph，
    必须转换为该结构后返回 Planner。
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(
        min_length=1,
        max_length=128,
    )

    tool_name: str = Field(
        min_length=1,
        max_length=128,
    )

    success: bool

    output: Any = None

    error: ToolErrorInfo | None = None

    duration_ms: int = Field(
        default=0,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_result(self) -> "ToolResult":
        if self.success and self.error is not None:
            raise ValueError(
                "成功的 ToolResult 不能包含 error。"
            )

        if not self.success and self.error is None:
            raise ValueError(
                "失败的 ToolResult 必须包含 error。"
            )

        return self


class ToolTraceEntry(BaseModel):
    """
    用于日志、接口调试字段和审计记录。

    arguments_summary 与 output_summary 应当是脱敏摘要，
    不应直接保存密码、API Key 或完整敏感财务信息。
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str

    status: ToolExecutionStatus

    arguments_summary: dict[str, Any] = Field(default_factory=dict)

    output_summary: dict[str, Any] = Field(default_factory=dict)

    error_code: ToolErrorCode | None = None

    duration_ms: int = Field(
        default=0,
        ge=0,
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
