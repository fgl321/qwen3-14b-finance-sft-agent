from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent_graph.production_dependencies import (
    ProductionGraphDependencies,
)
from app.agent_graph.production_nodes import (
    build_agent_loop_node,
    build_final_response_node,
    failure_response_node,
    prepare_production_run_node,
    route_after_agent_loop,
)
from app.agent_graph.production_state import (
    ProductionFinanceGraphState,
)


def build_production_finance_graph(
    *,
    dependencies: ProductionGraphDependencies,
    checkpointer: Any,
):
    """
    Stage 4.2 最终生产主图。

    图结构：

    START
      ↓
    prepare_run
      ↓
    agent_loop
      ├─ 正常 → final_response
      └─ 异常 → failure_response
      ↓
    END

    生产要求：
    1. checkpointer 必须由应用生命周期注入；
    2. 本文件不创建数据库连接；
    3. 本文件不使用 InMemorySaver；
    4. LLM 客户端和工具执行器不写入 Graph State；
    5. Graph State 只保存可序列化数据。
    """

    if checkpointer is None:
        raise ValueError(
            "生产主图必须注入持久化 checkpointer。"
        )

    graph_builder = StateGraph(
        ProductionFinanceGraphState
    )

    graph_builder.add_node(
        "prepare_run",
        prepare_production_run_node,
    )

    graph_builder.add_node(
        "agent_loop",
        build_agent_loop_node(
            dependencies
        ),
    )

    graph_builder.add_node(
        "final_response",
        build_final_response_node(
            dependencies
        ),
    )

    graph_builder.add_node(
        "failure_response",
        failure_response_node,
    )

    graph_builder.add_edge(
        START,
        "prepare_run",
    )

    graph_builder.add_edge(
        "prepare_run",
        "agent_loop",
    )

    graph_builder.add_conditional_edges(
        "agent_loop",
        route_after_agent_loop,
        {
            "final_response": "final_response",
            "failure": "failure_response",
        },
    )

    graph_builder.add_edge(
        "final_response",
        END,
    )

    graph_builder.add_edge(
        "failure_response",
        END,
    )

    return graph_builder.compile(
        checkpointer=checkpointer
    )
