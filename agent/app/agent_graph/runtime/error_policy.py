from __future__ import annotations

import hashlib
import re
from typing import Any

from app.agent_graph.schemas.tool_schema import (
    ToolErrorCode,
    ToolErrorInfo,
)


MODEL_REPAIRABLE_CODES: frozenset[ToolErrorCode] = frozenset(
    {
        "TOOL_NOT_FOUND",
        "ARGUMENT_SCHEMA_ERROR",
        "DOMAIN_INPUT_ERROR",
    }
)

INFRASTRUCTURE_RETRYABLE_CODES: frozenset[ToolErrorCode] = frozenset(
    {
        "TOOL_TIMEOUT",
        "DEPENDENCY_UNAVAILABLE",
        "RATE_LIMITED",
    }
)


def normalize_error_message(
    message: str,
    *,
    max_length: int = 500,
) -> str:
    """
    规范化错误文本，避免堆栈、换行或超长错误污染模型上下文。
    """

    normalized = re.sub(r"\s+", " ", message).strip()

    if not normalized:
        normalized = "未提供错误信息。"

    return normalized[:max_length]


def build_tool_error(
    *,
    code: ToolErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
) -> ToolErrorInfo:
    """
    根据固定错误码决定错误应由模型修复，
    还是由基础设施层内部重试。
    """

    return ToolErrorInfo(
        code=code,
        message=normalize_error_message(message),
        model_repairable=code in MODEL_REPAIRABLE_CODES,
        infrastructure_retryable=(
            code in INFRASTRUCTURE_RETRYABLE_CODES
        ),
        details=details or {},
    )


def build_error_signature(
    *,
    tool_name: str,
    error: ToolErrorInfo,
) -> str:
    """
    生成稳定错误签名，用于识别重复错误并终止死循环。

    不直接保存完整错误文本，避免日志中重复暴露敏感内容。
    """

    raw_value = "|".join(
        [
            tool_name.strip().lower(),
            error.code,
            normalize_error_message(
                error.message,
                max_length=300,
            ).lower(),
        ]
    )

    digest = hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:24]

    return f"{tool_name}:{error.code}:{digest}"
