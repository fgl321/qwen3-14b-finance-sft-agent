from __future__ import annotations

from typing import Any, Sequence

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.core.config import Settings


RetryableError = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class DeepSeekClient:
    """
    DeepSeek Flash API 客户端。

    生产级关键点：
    1. 使用异步客户端。
    2. 使用连接池。
    3. 设置连接超时、读取超时、写入超时。
    4. 禁止自动读取系统代理。
    5. 只对网络错误、超时、限流、服务端错误做重试。
    6. thinking 作为 DeepSeek 扩展参数，必须放在 extra_body 里。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        timeout = httpx.Timeout(
            connect=settings.deepseek_connect_timeout,
            read=settings.deepseek_read_timeout,
            write=30.0,
            pool=10.0,
        )

        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )

        self._http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            trust_env=settings.http_trust_env,
            http2=True,
        )

        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            http_client=self._http_client,
            # 关闭 OpenAI SDK 自带重试，统一交给 tenacity 管理。
            max_retries=0,
        )

    @retry(
        retry=retry_if_exception_type(RetryableError),
        wait=wait_random_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
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
        """
        调用 DeepSeek Chat Completions。

        注意：
        - 不要把 thinking 直接放在最外层。
        - OpenAI SDK 不认识最外层 thinking 参数。
        - DeepSeek 扩展参数必须通过 extra_body 传入。
        """

        request_body: dict[str, Any] = {
            "model": self.settings.deepseek_model,
            "messages": list(messages),
            "stream": False,

            # 为了兼容更多 OpenAI SDK 版本，这里用 max_tokens。
            # 你外部仍然可以叫 max_completion_tokens。
            "max_tokens": max_completion_tokens,

            # 关键修复点：thinking 必须放在 extra_body。
            "extra_body": {
                "thinking": {
                    "type": "enabled" if thinking_enabled else "disabled"
                }
            },
        }

        if thinking_enabled:
            request_body["reasoning_effort"] = "high"
        else:
            request_body["temperature"] = temperature

        if tools is not None:
            request_body["tools"] = tools
            request_body["tool_choice"] = "auto"

        if response_format is not None:
            request_body["response_format"] = response_format

        response = await self._client.chat.completions.create(**request_body)

        if not response.choices:
            raise RuntimeError("DeepSeek API 没有返回 choices。")

        choice = response.choices[0]
        message = choice.message

        return {
            "id": response.id,
            "model": response.model,
            "message": message.model_dump(exclude_none=True),
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage else {},
        }

    async def close(self) -> None:
        await self._client.close()
        await self._http_client.aclose()
