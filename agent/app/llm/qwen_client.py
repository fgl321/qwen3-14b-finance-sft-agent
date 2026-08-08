from __future__ import annotations

import json
from typing import Any, Sequence

import httpx
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.json_utils import extract_json_object


class QwenClient:
    """OpenAI-compatible client used only for final answer generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.qwen_connect_timeout,
                read=settings.qwen_read_timeout,
                write=settings.qwen_write_timeout,
                pool=settings.qwen_pool_timeout,
            ),
            trust_env=settings.http_trust_env,
        )
        self._client = AsyncOpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            http_client=self._http_client,
            max_retries=settings.qwen_max_retries,
        )

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        thinking_enabled: bool = False,
        temperature: float = 0.2,
        max_completion_tokens: int = 2048,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request_messages = [dict(item) for item in messages]
        tool = tools[0] if tools else None
        if tool:
            function = tool.get("function") or {}
            request_messages.append({
                "role": "system",
                "content": (
                    "Return only one JSON object containing the arguments for function "
                    f"{function.get('name')}. Do not use markdown. Required schema: "
                    + json.dumps(function.get("parameters") or {}, ensure_ascii=False)
                ),
            })
        response = await self._client.chat.completions.create(
            model=self.settings.qwen_model,
            messages=request_messages,
            temperature=temperature,
            max_tokens=max_completion_tokens,
            stream=False,
            extra_body={"enable_thinking": bool(thinking_enabled)},
        )
        if not response.choices:
            raise RuntimeError("Qwen service returned no choices")
        choice = response.choices[0]
        message = choice.message.model_dump(exclude_none=True)
        if tool:
            arguments = extract_json_object(str(message.get("content") or ""))
            function = tool["function"]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "qwen_final_answer",
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }],
            }
        return {
            "id": response.id,
            "model": response.model,
            "message": message,
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage else {},
        }

    async def close(self) -> None:
        await self._client.close()
        await self._http_client.aclose()
