from __future__ import annotations

from typing import Any

from app.core.logging import get_logger


logger = get_logger(__name__)


class QueryRewriter:
    """
    多轮查询改写器。

    作用：结合短期记忆历史，把包含指代（“它”“这个”“上面”等）的用户问题
    改写成独立、可检索的中文查询，提升 RAG 召回。

    原则：
    - 改写只用于检索，不用于最终回答；
    - 改写失败时静默回退到原始问题；
    - 不把对话历史本身写入任何持久化数据。
    """

    def __init__(
        self,
        *,
        llm_client: Any,
        enabled: bool = True,
        max_tokens: int = 256,
    ) -> None:
        self.llm_client = llm_client
        self.enabled = enabled
        self.max_tokens = max(32, int(max_tokens))

    async def rewrite(
        self,
        *,
        query: str,
        history_messages: list[dict[str, Any]],
    ) -> str:
        original = (query or "").strip()
        if not original:
            return ""

        if not self.enabled or self.llm_client is None:
            return original

        recent_history = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or "")[:1000],
            }
            for item in (history_messages or [])[-6:]
            if str(item.get("role") or "") in {"user", "assistant"}
            and str(item.get("content") or "").strip()
        ]

        if not recent_history:
            return original

        messages = [
            {
                "role": "system",
                "content": (
                    "你是中文金融 RAG 查询改写器。"
                    "根据对话历史和当前问题，把当前问题改写成一个"
                    "独立、完整、可直接检索的中文查询。"
                    "要求："
                    "1. 只输出改写后的查询本身，不要解释、不要引号、不要 Markdown；"
                    "2. 保留用户问题的原意和关键数字；"
                    "3. 如果当前问题已经完整明确，直接原样输出；"
                    "4. 不要补充对话中没有的信息。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"【对话历史】\n{self._format_history(recent_history)}\n\n"
                    f"【当前问题】\n{original}\n\n"
                    "请输出改写后的检索查询："
                ),
            },
        ]

        try:
            result = await self.llm_client.chat(
                messages=messages,
                thinking_enabled=False,
                max_completion_tokens=self.max_tokens,
            )
            rewritten = str(result["message"].get("content") or "").strip()
            rewritten = rewritten.strip("\"'`")
            if rewritten:
                logger.info(
                    "rag_query_rewritten",
                    original_length=len(original),
                    rewritten_length=len(rewritten),
                )
                return rewritten
        except Exception as exc:
            logger.warning(
                "rag_query_rewrite_failed_fallback_to_original",
                error_type=type(exc).__name__,
            )

        return original

    @staticmethod
    def _format_history(
        history_messages: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []
        for item in history_messages:
            role = "用户" if item.get("role") == "user" else "助手"
            lines.append(f"{role}: {item.get('content', '')}")
        return "\n".join(lines)
