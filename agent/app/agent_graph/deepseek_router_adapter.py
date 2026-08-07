from __future__ import annotations

from typing import Any

from app.agent_graph.llm_question_router import (
    HybridQuestionRouter,
    LLMQuestionRouter,
)
from app.llm.deepseek_client import DeepSeekClient


class DeepSeekRouterCompletion:
    def __init__(
        self,
        llm_client: DeepSeekClient,
        *,
        max_completion_tokens: int = 512,
    ) -> None:
        if max_completion_tokens <= 0:
            raise ValueError(
                "max_completion_tokens 必须大于 0。"
            )

        self._llm_client = llm_client
        self._max_completion_tokens = (
            max_completion_tokens
        )

    async def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        result = await self._llm_client.chat(
            messages=messages,
            thinking_enabled=False,
            max_completion_tokens=(
                self._max_completion_tokens
            ),
            response_format={
                "type": "json_object",
            },
        )

        if not isinstance(result, dict):
            raise RuntimeError(
                "DeepSeek 路由返回值不是字典。"
            )

        message = result.get("message")

        if not isinstance(message, dict):
            raise RuntimeError(
                "DeepSeek 路由结果缺少 message。"
            )

        content: Any = message.get("content")

        if not isinstance(content, str):
            raise RuntimeError(
                "DeepSeek 路由结果中的 content 不是字符串。"
            )

        content = content.strip()

        if not content:
            raise RuntimeError(
                "DeepSeek 路由返回了空内容。"
            )

        return content


def build_hybrid_question_router(
    *,
    llm_client: DeepSeekClient,
    timeout_seconds: float = 20.0,
    max_completion_tokens: int = 512,
) -> HybridQuestionRouter:
    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds 必须大于 0。"
        )

    completion = DeepSeekRouterCompletion(
        llm_client=llm_client,
        max_completion_tokens=max_completion_tokens,
    )

    llm_router = LLMQuestionRouter(
        completion_callable=completion,
        timeout_seconds=timeout_seconds,
    )

    return HybridQuestionRouter(
        llm_router=llm_router,
    )


__all__ = [
    "DeepSeekRouterCompletion",
    "build_hybrid_question_router",
]
