from functools import lru_cache
from typing import Any
from uuid import uuid4

from app.agent_graph.graph import build_finance_agent_graph


class FinanceAgentGraphService:
    """
    LangGraph 对外服务入口。

    这一层的作用：
    1. 屏蔽 LangGraph 底层 ainvoke 细节。
    2. 统一整理输入参数。
    3. 让 API 层以后只依赖 service，而不是直接依赖 graph。
    4. 复用已编译好的 graph，避免每次请求都重新 build。
    """

    def __init__(self) -> None:
        self._graph = build_finance_agent_graph()

    async def run(
        self,
        *,
        user_message: str,
        user_id: str,
        thread_id: str,
        request_id: str | None = None,
        tenant_id: str = "tenant_001",
        knowledge_base_id: str = "kb_finance_basic",
        history_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        执行一次 LangGraph Agent 调用。
        """
        clean_user_message = user_message.strip()
        clean_user_id = user_id.strip()
        clean_thread_id = thread_id.strip()

        if not clean_user_message:
            raise ValueError("user_message 不能为空")

        if not clean_user_id:
            raise ValueError("user_id 不能为空")

        if not clean_thread_id:
            raise ValueError("thread_id 不能为空")

        final_request_id = request_id or f"graph-{uuid4()}"

        result = await self._graph.ainvoke(
            {
                "request_id": final_request_id,
                "user_id": clean_user_id,
                "thread_id": clean_thread_id,
                "tenant_id": tenant_id,
                "knowledge_base_id": knowledge_base_id,
                "user_message": clean_user_message,
                "history_messages": history_messages or [],
            }
        )

        return dict(result)


@lru_cache(maxsize=1)
def get_finance_agent_graph_service() -> FinanceAgentGraphService:
    """
    获取全局单例 Graph Service。

    注意：
    这里缓存的是服务对象，服务对象里缓存了已编译的 LangGraph 图。
    这比每次请求重新 build graph 更适合后续接 FastAPI。
    """
    return FinanceAgentGraphService()
