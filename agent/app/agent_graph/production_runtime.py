from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)

from app.agent_graph.production_dependencies import (
    build_production_graph_dependencies,
)
from app.agent_graph.production_graph import (
    build_production_finance_graph,
)
from app.agent_graph.production_service import (
    ProductionFinanceGraphService,
)
from app.agent_graph.runtime.agent_limits import (
    AgentLimits,
    DEFAULT_AGENT_LIMITS,
)
from app.core.logging import get_logger


logger = get_logger(__name__)


@dataclass(slots=True)
class ProductionGraphRuntime:
    """
    FastAPI 生命周期内持有的生产图运行时。
    """

    checkpointer: AsyncPostgresSaver
    graph: Any
    service: ProductionFinanceGraphService


@asynccontextmanager
async def open_production_graph_runtime(
    *,
    llm_client: Any,
    synthesis_llm_client: Any | None = None,
    postgres_dsn: str,
    limits: AgentLimits = DEFAULT_AGENT_LIMITS,
    setup_checkpointer: bool = False,
) -> AsyncIterator[ProductionGraphRuntime]:
    """
    打开生产 LangGraph 运行时。

    AsyncPostgresSaver 的连接必须覆盖整个应用生命周期，
    不能在创建完 graph 后立即退出上下文，否则后续请求使用
    graph 时数据库连接已经关闭。
    """

    clean_dsn = postgres_dsn.strip()

    if not clean_dsn:
        raise ValueError(
            "postgres_dsn 不能为空。"
        )

    logger.info(
        "production_graph_runtime_starting",
        setup_checkpointer=setup_checkpointer,
    )

    async with (
        AsyncPostgresSaver.from_conn_string(
            clean_dsn
        )
    ) as checkpointer:
        if setup_checkpointer:
            logger.info(
                "langgraph_checkpointer_setup_started"
            )

            await checkpointer.setup()

            logger.info(
                "langgraph_checkpointer_setup_finished"
            )

        dependencies = (
            build_production_graph_dependencies(
                llm_client=llm_client,
                synthesis_llm_client=synthesis_llm_client,
                limits=limits,
            )
        )

        graph = build_production_finance_graph(
            dependencies=dependencies,
            checkpointer=checkpointer,
        )

        service = ProductionFinanceGraphService(
            graph=graph
        )

        runtime = ProductionGraphRuntime(
            checkpointer=checkpointer,
            graph=graph,
            service=service,
        )

        logger.info(
            "production_graph_runtime_ready"
        )

        try:
            yield runtime
        finally:
            logger.info(
                "production_graph_runtime_stopped"
            )
