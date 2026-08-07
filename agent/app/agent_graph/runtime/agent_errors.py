from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, Protocol

from fastapi import HTTPException

from app.agent_graph.runtime.request_idempotency import (
    RequestIdempotencyConflict,
)
from app.agent_graph.schemas.error_schema import (
    AgentErrorCategory,
    AgentErrorEnvelope,
    AgentErrorStage,
)
class ToolErrorLike(Protocol):
    """
    工具错误的结构化协议。

    不依赖 tool_schema.py 中某个具体类名，避免不同阶段的
    ToolError / ToolFailureDetail 命名变化破坏统一错误模块。
    """

    code: str
    message: str
    model_repairable: bool
    infrastructure_retryable: bool
    details: Mapping[str, Any]


_SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "session",
    "private_key",
}


def _is_sensitive_key(value: str) -> bool:
    normalized = value.strip().lower()
    return any(
        marker in normalized
        for marker in _SENSITIVE_KEYS
    )


def sanitize_error_details(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    """
    将错误附加信息裁剪为可审计、可序列化且不含凭证的结构。
    """

    if depth > 4:
        return "[max_depth]"

    if value is None or isinstance(
        value,
        (bool, int, float),
    ):
        return value

    if isinstance(value, str):
        if len(value) > 300:
            return f"{value[:300]}...[truncated]"
        return value

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for index, (key, item) in enumerate(
            value.items()
        ):
            if index >= 30:
                result["__truncated__"] = True
                break

            clean_key = str(key)

            if _is_sensitive_key(clean_key):
                result[clean_key] = "[redacted]"
                continue

            result[clean_key] = sanitize_error_details(
                item,
                depth=depth + 1,
            )

        return result

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        result_list = [
            sanitize_error_details(
                item,
                depth=depth + 1,
            )
            for item in value[:20]
        ]

        if len(value) > 20:
            result_list.append("[truncated]")

        return result_list

    return str(value)[:300]


def build_agent_error(
    *,
    code: str,
    category: AgentErrorCategory,
    stage: AgentErrorStage,
    message: str,
    retryable: bool,
    http_status: int,
    request_id: str | None = None,
    run_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> AgentErrorEnvelope:
    return AgentErrorEnvelope(
        code=code,
        category=category,
        stage=stage,
        message=message,
        retryable=retryable,
        http_status=http_status,
        request_id=request_id,
        run_id=run_id,
        details=sanitize_error_details(
            dict(details or {})
        ),
    )


class AgentExecutionError(RuntimeError):
    """
    携带统一错误结构的运行时异常。

    继承 RuntimeError 是为了兼容旧调用方和已有回归测试。
    """

    def __init__(
        self,
        error: AgentErrorEnvelope,
    ) -> None:
        self.error = error
        super().__init__(error.message)


def exception_to_agent_error(
    exc: BaseException,
    *,
    stage: AgentErrorStage,
    request_id: str | None = None,
    run_id: str | None = None,
) -> AgentErrorEnvelope:
    if isinstance(exc, AgentExecutionError):
        return exc.error

    if isinstance(exc, RequestIdempotencyConflict):
        return build_agent_error(
            code="REQUEST_ID_CONFLICT",
            category="conflict",
            stage="idempotency",
            message=(
                "同一个 request_id 已经用于不同的请求内容，"
                "请为新请求使用新的 request_id。"
            ),
            retryable=False,
            http_status=409,
            request_id=request_id,
            run_id=run_id,
        )

    if isinstance(exc, PermissionError):
        return build_agent_error(
            code="REQUEST_PERMISSION_DENIED",
            category="permission",
            stage=stage,
            message="当前请求没有完成该操作所需的权限。",
            retryable=False,
            http_status=403,
            request_id=request_id,
            run_id=run_id,
        )

    if isinstance(exc, TimeoutError):
        return build_agent_error(
            code="AGENT_EXECUTION_TIMEOUT",
            category="timeout",
            stage=stage,
            message="Agent 执行超时，请稍后重试。",
            retryable=True,
            http_status=504,
            request_id=request_id,
            run_id=run_id,
        )

    if isinstance(exc, (ConnectionError, OSError)):
        return build_agent_error(
            code="DEPENDENCY_UNAVAILABLE",
            category="dependency",
            stage=stage,
            message="Agent 依赖的服务暂时不可用，请稍后重试。",
            retryable=True,
            http_status=503,
            request_id=request_id,
            run_id=run_id,
            details={
                "exception_type": type(exc).__name__,
            },
        )

    exception_name = type(exc).__name__

    if exception_name in {
        "PlannerProtocolError",
        "PlannerDecisionConsistencyError",
    }:
        return build_agent_error(
            code="PLANNER_PROTOCOL_ERROR",
            category="protocol",
            stage="planner",
            message="Planner 返回的结构化协议不合法。",
            retryable=True,
            http_status=502,
            request_id=request_id,
            run_id=run_id,
            details={
                "exception_type": exception_name,
            },
        )

    if isinstance(exc, ValueError):
        clean_message = str(exc).strip()

        return build_agent_error(
            code="REQUEST_VALIDATION_ERROR",
            category="validation",
            stage=stage,
            message=(
                clean_message
                or "请求参数不合法。"
            ),
            retryable=False,
            http_status=422,
            request_id=request_id,
            run_id=run_id,
        )

    return build_agent_error(
        code="AGENT_INTERNAL_ERROR",
        category="internal",
        stage=stage,
        message="生产 Agent 执行失败。",
        retryable=False,
        http_status=500,
        request_id=request_id,
        run_id=run_id,
        details={
            "exception_type": exception_name,
        },
    )


def tool_error_to_agent_error(
    tool_error: ToolErrorLike,
    *,
    tool_name: str,
    request_id: str | None = None,
    run_id: str | None = None,
) -> AgentErrorEnvelope:
    mapping: dict[
        str,
        tuple[
            AgentErrorCategory,
            int,
            bool,
        ],
    ] = {
        "ARGUMENT_SCHEMA_ERROR": (
            "validation",
            422,
            False,
        ),
        "DOMAIN_INPUT_ERROR": (
            "validation",
            422,
            False,
        ),
        "PERMISSION_DENIED": (
            "permission",
            403,
            False,
        ),
        "TOOL_TIMEOUT": (
            "timeout",
            504,
            True,
        ),
        "DEPENDENCY_UNAVAILABLE": (
            "dependency",
            503,
            True,
        ),
        "TOOL_NOT_FOUND": (
            "protocol",
            500,
            False,
        ),
        "AGENT_BUDGET_EXCEEDED": (
            "budget",
            429,
            False,
        ),
        "TOOL_INTERNAL_ERROR": (
            "internal",
            500,
            False,
        ),
    }

    category, http_status, default_retryable = (
        mapping.get(
            tool_error.code,
            ("tool", 500, False),
        )
    )

    return build_agent_error(
        code=tool_error.code,
        category=category,
        stage="tool",
        message=tool_error.message,
        retryable=(
            tool_error.infrastructure_retryable
            or default_retryable
        ),
        http_status=http_status,
        request_id=request_id,
        run_id=run_id,
        details={
            "tool_name": tool_name,
            "model_repairable": (
                tool_error.model_repairable
            ),
            "tool_details": tool_error.details,
        },
    )


def raise_agent_http_exception(
    error: AgentErrorEnvelope,
) -> NoReturn:
    raise HTTPException(
        status_code=error.http_status,
        detail=error.model_dump(mode="json"),
        headers={
            "X-Agent-Error-Code": error.code,
            "X-Agent-Error-ID": error.error_id,
        },
    )


def log_event(
    logger: Any,
    level: str,
    event: str,
    **fields: Any,
) -> None:
    """
    同时兼容 structlog BoundLogger 和标准 logging.Logger。
    """

    method = getattr(logger, level)

    try:
        method(event, **fields)
        return
    except TypeError:
        pass

    if level == "exception":
        logger.error(
            "%s | %s",
            event,
            fields,
            exc_info=True,
        )
        return

    method(
        "%s | %s",
        event,
        fields,
    )
