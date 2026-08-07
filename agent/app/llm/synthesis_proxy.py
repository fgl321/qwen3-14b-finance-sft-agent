from __future__ import annotations

import contextvars
from typing import Any


_synthesis_provider_var: contextvars.ContextVar[str] = (
    contextvars.ContextVar(
        "synthesis_provider",
        default="qwen",
    )
)


def set_synthesis_provider(
    provider: str,
) -> contextvars.Token:
    """在当前请求上下文中设置最终回答使用的模型。"""
    return _synthesis_provider_var.set(provider)


def reset_synthesis_provider(
    token: contextvars.Token,
) -> None:
    _synthesis_provider_var.reset(token)


def current_synthesis_provider() -> str:
    return _synthesis_provider_var.get()


class SynthesisClientProxy:
    """
    按请求上下文分发最终回答模型：qwen=蒸馏模型，deepseek=DeepSeek API。

    允许前端在同一个 Agent 服务里自由切换，无需重启。
    """

    def __init__(
        self,
        *,
        qwen_client: Any,
        deepseek_client: Any,
        default_provider: str = "qwen",
    ) -> None:
        self.qwen_client = qwen_client
        self.deepseek_client = deepseek_client
        self.default_provider = (
            "deepseek"
            if str(default_provider).strip().lower() == "deepseek"
            else "qwen"
        )

    def _pick_client(self) -> Any:
        provider = current_synthesis_provider()
        if provider == "deepseek":
            return self.deepseek_client
        return self.qwen_client

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        client = self._pick_client()
        return await client.chat(**kwargs)

    async def close(self) -> None:
        # 两个底层客户端由应用生命周期统一关闭。
        return None
