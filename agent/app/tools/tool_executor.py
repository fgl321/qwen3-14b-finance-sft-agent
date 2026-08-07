from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError

from app.agent_graph.runtime.agent_limits import (
    AgentLimits,
    DEFAULT_AGENT_LIMITS,
)
from app.agent_graph.runtime.error_policy import build_tool_error
from app.agent_graph.schemas.planner_schema import ToolCallRequest
from app.agent_graph.schemas.tool_schema import (
    ToolResult,
    ToolTraceEntry,
)
from app.tools.runtime_registry import ToolRegistry
from app.tools.tool_specs import ToolSpec


_CREDENTIAL_KEYS = {
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

_FINANCIAL_KEYS = {
    "income",
    "expense",
    "amount",
    "balance",
    "asset",
    "assets",
    "debt",
    "mortgage",
    "insurance",
    "fund",
    "principal",
    "salary",
    "cash",
}


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """
    一次 Agent 运行中的工具执行上下文。

    None 表示不按该维度额外限制；
    工具仍然必须存在于显式 Registry 中。
    """

    request_id: str
    run_id: str

    tenant_id: str = "default"
    user_id: str = "anonymous"
    role: str = "user"

    allowed_tool_names: frozenset[str] | None = None
    allowed_tool_groups: frozenset[str] | None = None

    allow_side_effects: bool = False

    remaining_tool_calls: int = 12


@dataclass(frozen=True, slots=True)
class ToolExecutionOutcome:
    result: ToolResult
    trace: ToolTraceEntry


def _canonical_number(value: Decimal) -> str:
    """
    将数值转换为稳定文本，保证 180000、180000.0、
    Decimal("180000.00") 生成同一个复用签名。
    """

    if not value.is_finite():
        return str(value)

    if value == 0:
        return "0"

    normalized = value.normalize()
    text = format(normalized, "f")

    if "." in text:
        text = text.rstrip("0").rstrip(".")

    return text or "0"


def _canonical_signature_value(value: Any) -> Any:
    """
    将已通过工具输入模型校验的参数转换为稳定 JSON 结构。

    这里不记录日志，只用于本次 Agent 运行中的哈希比较。
    """

    if isinstance(value, BaseModel):
        return _canonical_signature_value(
            value.model_dump(mode="python")
        )

    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_signature_value(asdict(value))

    if isinstance(value, Enum):
        return _canonical_signature_value(value.value)

    if isinstance(value, bool):
        return value

    if isinstance(value, Decimal):
        return {
            "__number__": _canonical_number(value),
        }

    if isinstance(value, int):
        return {
            "__number__": _canonical_number(
                Decimal(value)
            ),
        }

    if isinstance(value, float):
        return {
            "__number__": _canonical_number(
                Decimal(str(value))
            ),
        }

    if isinstance(value, (datetime, date)):
        return {
            "__datetime__": value.isoformat(),
        }

    if isinstance(value, Mapping):
        return {
            str(key): _canonical_signature_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _canonical_signature_value(item)
            for item in value
        ]

    if value is None or isinstance(value, str):
        return value

    return str(value)


def normalize_tool_output(value: Any) -> Any:
    """
    将任意工具输出转换为可写入 JSON 和 LangGraph State 的结构。
    """

    if value is None:
        return None

    if isinstance(value, BaseModel):
        return normalize_tool_output(
            value.model_dump(mode="json")
        )

    if is_dataclass(value) and not isinstance(value, type):
        return normalize_tool_output(asdict(value))

    if isinstance(value, Decimal):
        return format(value, "f")

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Enum):
        return normalize_tool_output(value.value)

    if isinstance(value, Mapping):
        return {
            str(key): normalize_tool_output(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            normalize_tool_output(item)
            for item in value
        ]

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _contains_keyword(
    key: str,
    keyword_set: set[str],
) -> bool:
    normalized = key.lower()

    return any(
        keyword in normalized
        for keyword in keyword_set
    )


def summarize_for_trace(
    value: Any,
    *,
    field_name: str | None = None,
    depth: int = 0,
) -> Any:
    """
    创建脱敏审计摘要。

    - 凭证字段完全隐藏；
    - 金融金额字段不进入普通日志；
    - 长字符串截断；
    - 过深、过长结构裁剪。
    """

    if depth > 5:
        return "[max_depth]"

    if field_name:
        if _contains_keyword(field_name, _CREDENTIAL_KEYS):
            return "[redacted]"

        if _contains_keyword(field_name, _FINANCIAL_KEYS):
            return "[financial_value]"

    normalized = normalize_tool_output(value)

    if isinstance(normalized, str):
        if len(normalized) > 200:
            return f"{normalized[:200]}...[truncated]"

        return normalized

    if isinstance(normalized, Mapping):
        result: dict[str, Any] = {}

        for index, (key, item) in enumerate(
            normalized.items()
        ):
            if index >= 30:
                result["__truncated__"] = True
                break

            result[str(key)] = summarize_for_trace(
                item,
                field_name=str(key),
                depth=depth + 1,
            )

        return result

    if isinstance(normalized, list):
        result_list = [
            summarize_for_trace(
                item,
                depth=depth + 1,
            )
            for item in normalized[:20]
        ]

        if len(normalized) > 20:
            result_list.append("[truncated]")

        return result_list

    return normalized


def _validation_error_details(
    error: ValidationError,
) -> dict[str, Any]:
    """
    不把 Pydantic errors() 中的原始 input 直接返回给模型，
    避免把敏感参数复制进错误上下文。
    """

    safe_errors: list[dict[str, Any]] = []

    for item in error.errors():
        safe_errors.append(
            {
                "location": [
                    str(part)
                    for part in item.get("loc", ())
                ],
                "type": str(item.get("type", "")),
                "message": str(item.get("msg", "")),
            }
        )

    return {
        "validation_errors": safe_errors,
    }


class ProductionToolExecutor:
    """
    生产级工具执行器。

    职责：
    1. 只执行 Registry 中显式注册的工具；
    2. 用工具自己的输入模型解析参数；
    3. 执行权限、风险和预算检查；
    4. 控制超时和基础设施重试；
    5. 捕获异常并返回统一 ToolResult；
    6. 生成脱敏 ToolTraceEntry。

    它不从用户自然语言中提取参数，
    也不判断 DeepSeek 对用户语义的理解是否正确。
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        limits: AgentLimits = DEFAULT_AGENT_LIMITS,
    ) -> None:
        self.registry = registry
        self.limits = limits

    def build_reuse_signature(
        self,
        tool_call: ToolCallRequest,
        *,
        context: ToolExecutionContext,
    ) -> str | None:
        """
        为可安全复用的工具调用生成稳定签名。

        返回 None 表示该调用不能进入成功结果复用：
        - 工具不存在；
        - 当前上下文无权限；
        - 参数不符合输入模型；
        - 工具有副作用；
        - 工具不是幂等工具。

        该方法只做 Registry、权限和参数协议校验，
        不调用工具处理函数。
        """

        spec = self.registry.get(tool_call.tool_name)

        if spec is None:
            return None

        if spec.side_effect or not spec.idempotent:
            return None

        if self._check_permission(
            spec=spec,
            context=context,
        ) is not None:
            return None

        try:
            validated_input = spec.input_model.model_validate(
                tool_call.arguments
            )
        except ValidationError:
            return None

        canonical_payload = {
            "tool_name": spec.name,
            "arguments": _canonical_signature_value(
                validated_input.model_dump(mode="python")
            ),
        }

        serialized = json.dumps(
            canonical_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        digest = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

        return f"{spec.name}:{digest}"

    async def execute_one(
        self,
        tool_call: ToolCallRequest,
        *,
        context: ToolExecutionContext,
    ) -> ToolExecutionOutcome:
        started_at = time.perf_counter()

        if context.remaining_tool_calls <= 0:
            return self._failed_outcome(
                tool_call=tool_call,
                started_at=started_at,
                code="AGENT_BUDGET_EXCEEDED",
                message="本次 Agent 运行的工具调用预算已耗尽。",
            )

        spec = self.registry.get(tool_call.tool_name)

        if spec is None:
            return self._failed_outcome(
                tool_call=tool_call,
                started_at=started_at,
                code="TOOL_NOT_FOUND",
                message=(
                    f"工具 {tool_call.tool_name} 未注册。"
                    f"可用工具：{', '.join(self.registry.names())}"
                ),
                details={
                    "available_tools": list(
                        self.registry.names()
                    ),
                },
            )

        permission_error = self._check_permission(
            spec=spec,
            context=context,
        )

        if permission_error is not None:
            return self._failed_outcome(
                tool_call=tool_call,
                started_at=started_at,
                code="PERMISSION_DENIED",
                message=permission_error,
            )

        try:
            validated_input = spec.input_model.model_validate(
                tool_call.arguments
            )
        except ValidationError as exc:
            return self._failed_outcome(
                tool_call=tool_call,
                started_at=started_at,
                code="ARGUMENT_SCHEMA_ERROR",
                message=(
                    f"工具 {spec.name} 的参数不符合输入结构。"
                ),
                details=_validation_error_details(exc),
            )

        arguments = validated_input.model_dump(mode="python")

        try:
            raw_output = await self._invoke_with_retry(
                spec=spec,
                arguments=arguments,
            )

            output = normalize_tool_output(raw_output)

            duration_ms = self._duration_ms(started_at)

            result = ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )

            trace = ToolTraceEntry(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                status="succeeded",
                arguments_summary=summarize_for_trace(
                    arguments
                ),
                output_summary=summarize_for_trace(
                    output
                ),
                duration_ms=duration_ms,
            )

            return ToolExecutionOutcome(
                result=result,
                trace=trace,
            )

        except TimeoutError:
            return self._failed_outcome(
                tool_call=tool_call,
                started_at=started_at,
                code="TOOL_TIMEOUT",
                message=(
                    f"工具 {spec.name} 在 "
                    f"{spec.timeout_seconds:.2f} 秒内未完成。"
                ),
            )

        except PermissionError as exc:
            return self._failed_outcome(
                tool_call=tool_call,
                started_at=started_at,
                code="PERMISSION_DENIED",
                message=str(exc) or "工具执行权限不足。",
            )

        except ValueError as exc:
            return self._failed_outcome(
                tool_call=tool_call,
                started_at=started_at,
                code="DOMAIN_INPUT_ERROR",
                message=str(exc) or "工具领域参数不合法。",
            )

        except (ConnectionError, OSError) as exc:
            return self._failed_outcome(
                tool_call=tool_call,
                started_at=started_at,
                code="DEPENDENCY_UNAVAILABLE",
                message=(
                    f"工具 {spec.name} 依赖的外部服务暂时不可用："
                    f"{type(exc).__name__}"
                ),
            )

        except Exception as exc:
            return self._failed_outcome(
                tool_call=tool_call,
                started_at=started_at,
                code="TOOL_INTERNAL_ERROR",
                message=(
                    f"工具 {spec.name} 执行时发生未处理异常："
                    f"{type(exc).__name__}"
                ),
            )

    async def execute_many(
        self,
        tool_calls: list[ToolCallRequest],
        *,
        context: ToolExecutionContext,
    ) -> list[ToolExecutionOutcome]:
        """
        并行执行同一 Planner 回合中的工具调用。

        返回顺序与 tool_calls 输入顺序一致。
        """

        if not tool_calls:
            return []

        if len(tool_calls) > context.remaining_tool_calls:
            return [
                self._failed_outcome(
                    tool_call=tool_call,
                    started_at=time.perf_counter(),
                    code="AGENT_BUDGET_EXCEEDED",
                    message=(
                        "本轮工具调用数量超过剩余工具调用预算，"
                        "本批次未执行。"
                    ),
                )
                for tool_call in tool_calls
            ]

        max_parallel = min(
            self.limits.max_parallel_tool_calls,
            len(tool_calls),
        )

        semaphore = asyncio.Semaphore(max_parallel)

        async def run_one(
            tool_call: ToolCallRequest,
        ) -> ToolExecutionOutcome:
            async with semaphore:
                return await self.execute_one(
                    tool_call,
                    context=context,
                )

        tasks = [
            asyncio.create_task(run_one(tool_call))
            for tool_call in tool_calls
        ]

        return list(await asyncio.gather(*tasks))

    def _check_permission(
        self,
        *,
        spec: ToolSpec,
        context: ToolExecutionContext,
    ) -> str | None:
        if context.role not in spec.allowed_roles:
            return (
                f"角色 {context.role} 无权调用工具 {spec.name}。"
            )

        if (
            context.allowed_tool_names
            and spec.name not in context.allowed_tool_names
        ):
            return (
                f"当前路由没有授权调用工具 {spec.name}。"
            )

        if (
            context.allowed_tool_groups
            and spec.tool_group
            not in context.allowed_tool_groups
        ):
            return (
                f"当前路由没有授权调用工具组 "
                f"{spec.tool_group}。"
            )

        if spec.side_effect and not context.allow_side_effects:
            return (
                f"工具 {spec.name} 具有副作用，"
                "当前运行未获得副作用执行授权。"
            )

        return None

    async def _invoke_with_retry(
        self,
        *,
        spec: ToolSpec,
        arguments: dict[str, Any],
    ) -> Any:
        retry_index = 0

        while True:
            try:
                async with asyncio.timeout(
                    spec.timeout_seconds
                ):
                    return await self._invoke_handler(
                        spec=spec,
                        arguments=arguments,
                    )

            except (TimeoutError, ConnectionError, OSError):
                can_retry = (
                    spec.idempotent
                    and retry_index
                    < spec.max_infrastructure_retries
                )

                if not can_retry:
                    raise

                base_delay = 0.25 * (2 ** retry_index)
                jitter = random.uniform(0, 0.1)

                await asyncio.sleep(base_delay + jitter)

                retry_index += 1

    async def _invoke_handler(
        self,
        *,
        spec: ToolSpec,
        arguments: dict[str, Any],
    ) -> Any:
        handler = spec.handler

        if inspect.iscoroutinefunction(handler):
            return await handler(**arguments)

        # 同步金融计算放入工作线程，避免阻塞 FastAPI 事件循环。
        result = await asyncio.to_thread(
            handler,
            **arguments,
        )

        # 兼容被装饰器包装后返回 awaitable 的函数。
        if inspect.isawaitable(result):
            return await result

        return result

    def _failed_outcome(
        self,
        *,
        tool_call: ToolCallRequest,
        started_at: float,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> ToolExecutionOutcome:
        error = build_tool_error(
            code=code,  # type: ignore[arg-type]
            message=message,
            details=details,
        )

        duration_ms = self._duration_ms(started_at)

        result = ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            success=False,
            error=error,
            duration_ms=duration_ms,
        )

        trace = ToolTraceEntry(
            tool_call_id=tool_call.tool_call_id,
            tool_name=tool_call.tool_name,
            status=(
                "timed_out"
                if code == "TOOL_TIMEOUT"
                else "rejected"
                if code in {
                    "TOOL_NOT_FOUND",
                    "PERMISSION_DENIED",
                    "AGENT_BUDGET_EXCEEDED",
                }
                else "failed"
            ),
            arguments_summary=summarize_for_trace(
                tool_call.arguments
            ),
            output_summary={},
            error_code=error.code,
            duration_ms=duration_ms,
        )

        return ToolExecutionOutcome(
            result=result,
            trace=trace,
        )

    @staticmethod
    def _duration_ms(started_at: float) -> int:
        return max(
            0,
            int((time.perf_counter() - started_at) * 1000),
        )
