from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, TypeVar

from pydantic import BaseModel

from app.core.json_utils import extract_json_object, parse_arguments


T = TypeVar("T", bound=BaseModel)
StructuredStatus = Literal[
    "success",
    "repaired",
    "protocol_failed",
    "service_failed",
]


@dataclass(slots=True)
class StructuredLLMResult(Generic[T]):
    status: StructuredStatus
    parsed: T | None = None
    attempts: int = 1
    raw_response: str | None = None
    validation_errors: list[dict[str, str]] = field(default_factory=list)
    model: str | None = None
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class StructuredLLMGateway:
    """One protocol boundary for schema-constrained LLM stages.

    A protocol repair retries the same stage only. It never consumes an Agent
    execution round and never re-runs unrelated tools.
    """

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    async def invoke_json(
        self,
        *,
        schema: type[T],
        messages: list[dict[str, Any]],
        stage: str,
        max_completion_tokens: int,
        max_protocol_repairs: int = 1,
        normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> StructuredLLMResult[T]:
        return await self._invoke(
            schema=schema,
            messages=messages,
            stage=stage,
            max_completion_tokens=max_completion_tokens,
            max_protocol_repairs=max_protocol_repairs,
            normalize=normalize,
            tools=None,
            expected_tool_name=None,
        )

    async def invoke_tool(
        self,
        *,
        schema: type[T],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        expected_tool_name: str,
        stage: str,
        max_completion_tokens: int,
        max_protocol_repairs: int = 1,
        normalize: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> StructuredLLMResult[T]:
        return await self._invoke(
            schema=schema,
            messages=messages,
            stage=stage,
            max_completion_tokens=max_completion_tokens,
            max_protocol_repairs=max_protocol_repairs,
            normalize=normalize,
            tools=tools,
            expected_tool_name=expected_tool_name,
        )

    async def _invoke(
        self,
        *,
        schema: type[T],
        messages: list[dict[str, Any]],
        stage: str,
        max_completion_tokens: int,
        max_protocol_repairs: int,
        normalize: Callable[[dict[str, Any]], dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
        expected_tool_name: str | None,
    ) -> StructuredLLMResult[T]:
        working_messages = list(messages)
        errors: list[dict[str, str]] = []
        last_raw: str | None = None
        last_response: dict[str, Any] = {}

        for attempt in range(1, max_protocol_repairs + 2):
            try:
                kwargs: dict[str, Any] = {
                    "messages": working_messages,
                    "thinking_enabled": False,
                    "max_completion_tokens": max_completion_tokens,
                }
                if tools is None:
                    kwargs["response_format"] = {"type": "json_object"}
                else:
                    kwargs["tools"] = tools
                response = await self.llm_client.chat(**kwargs)
                last_response = response
            except Exception as exc:
                return StructuredLLMResult(
                    status="service_failed",
                    attempts=attempt,
                    raw_response=last_raw,
                    validation_errors=[
                        *errors,
                        {"stage": stage, "error": type(exc).__name__},
                    ],
                )

            assistant = response.get("message") or {}
            try:
                if expected_tool_name is None:
                    last_raw = str(assistant.get("content") or "")
                    payload = extract_json_object(last_raw)
                else:
                    tool_calls = assistant.get("tool_calls") or []
                    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                        raise ValueError("必须且只能调用一次指定结构化工具")
                    function = tool_calls[0].get("function") or {}
                    if str(function.get("name") or "") != expected_tool_name:
                        raise ValueError("调用了错误的结构化工具")
                    raw_arguments = function.get("arguments")
                    last_raw = (
                        raw_arguments
                        if isinstance(raw_arguments, str)
                        else json.dumps(raw_arguments, ensure_ascii=False)
                    )
                    try:
                        payload = parse_arguments(raw_arguments)
                    except (TypeError, ValueError):
                        payload = extract_json_object(str(raw_arguments))
                if normalize is not None:
                    payload = normalize(payload)
                parsed = schema.model_validate(payload)
            except Exception as exc:
                errors.append(
                    {
                        "stage": stage,
                        "attempt": str(attempt),
                        "error": f"{type(exc).__name__}: {exc}"[:1000],
                    }
                )
                if attempt <= max_protocol_repairs:
                    if last_raw:
                        working_messages.append(
                            {"role": "assistant", "content": last_raw[:12_000]}
                        )
                    working_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "上一响应未通过结构化协议校验。只修复格式和缺失字段，"
                                "不要改变事实判断；重新输出完整、闭合、严格符合 Schema 的 JSON。"
                                f"\n阶段：{stage}\n错误：{errors[-1]['error']}"
                            ),
                        }
                    )
                    continue
                return StructuredLLMResult(
                    status="protocol_failed",
                    attempts=attempt,
                    raw_response=last_raw,
                    validation_errors=errors,
                    model=response.get("model"),
                    finish_reason=str(response.get("finish_reason") or ""),
                    usage=response.get("usage") or {},
                )

            return StructuredLLMResult(
                status="repaired" if attempt > 1 else "success",
                parsed=parsed,
                attempts=attempt,
                raw_response=last_raw,
                validation_errors=errors,
                model=response.get("model"),
                finish_reason=str(response.get("finish_reason") or ""),
                usage=response.get("usage") or {},
            )

        return StructuredLLMResult(
            status="protocol_failed",
            attempts=max_protocol_repairs + 1,
            raw_response=last_raw,
            validation_errors=errors,
            model=last_response.get("model"),
        )
