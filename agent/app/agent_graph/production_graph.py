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
from app.agent_graph.explicit_workflow import (
    build_agent_result_node,
    build_answer_synthesis_node,
    build_intent_router_node,
    build_observation_validator_node,
    build_capability_validator_node,
    build_plan_review_node,
    build_output_guard_node,
    build_planner_node,
    build_tool_executor_node,
    record_trace_node,
    route_after_observation,
    route_after_capability_validation,
    route_after_output_guard,
    route_after_planner,
    route_after_review,
    route_after_synthesis,
)


def build_production_finance_graph(
    *,
    dependencies: ProductionGraphDependencies,
    checkpointer: Any,
):
    """
    金融 Agent v2 生产主图。

    图结构：

    正式运行时使用显式、可审计的节点链路：请求准备、意图路由、
    规划、计划审核、工具执行、观察校验、能力/结果校验、结果组装、
    答案生成、输出防护和轨迹落盘。Plan Repair 留在当前目标执行轮内；
    只有 Execute -> Observe -> Result Validate 才增加 execution_round。

    只注入旧版聚合 agent_loop 的单元测试依赖会进入兼容分支；
    应用生命周期构造的真实依赖始终走显式工作流。

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

    # Compatibility adapters used by focused unit tests can still inject only
    # the legacy aggregate loop.  The real application runtime always builds
    # the explicit workflow below.
    if not dependencies.explicit_workflow_ready:
        graph_builder.add_node(
            "agent_loop",
            build_agent_loop_node(dependencies),
        )
        graph_builder.add_node(
            "final_response",
            build_final_response_node(dependencies),
        )
        graph_builder.add_node("failure_response", failure_response_node)
        graph_builder.add_edge(START, "prepare_run")
        graph_builder.add_edge("prepare_run", "agent_loop")
        graph_builder.add_conditional_edges(
            "agent_loop",
            route_after_agent_loop,
            {"final_response": "final_response", "failure": "failure_response"},
        )
        graph_builder.add_edge("final_response", END)
        graph_builder.add_edge("failure_response", END)
        return graph_builder.compile(checkpointer=checkpointer)

    graph_builder.add_node(
        "intent_router",
        build_intent_router_node(dependencies),
    )
    graph_builder.add_node(
        "planner",
        build_planner_node(dependencies),
    )
    graph_builder.add_node(
        "plan_review",
        build_plan_review_node(dependencies),
    )
    graph_builder.add_node(
        "tool_executor",
        build_tool_executor_node(dependencies),
    )
    graph_builder.add_node(
        "observation_validator",
        build_observation_validator_node(dependencies),
    )
    graph_builder.add_node(
        "result_validator",
        build_capability_validator_node(dependencies),
    )
    graph_builder.add_node(
        "agent_result_assembler",
        build_agent_result_node(dependencies),
    )

    graph_builder.add_node(
        "answer_synthesis",
        build_answer_synthesis_node(dependencies),
    )
    graph_builder.add_node(
        "output_guard",
        build_output_guard_node(dependencies),
    )

    graph_builder.add_node(
        "failure_response",
        failure_response_node,
    )
    graph_builder.add_node("trace_finalizer", record_trace_node)

    graph_builder.add_edge(
        START,
        "prepare_run",
    )

    graph_builder.add_edge(
        "prepare_run",
        "intent_router",
    )
    graph_builder.add_edge("intent_router", "planner")
    graph_builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "review": "plan_review",
            "assemble": "agent_result_assembler",
            "failure": "failure_response",
        },
    )
    graph_builder.add_conditional_edges(
        "plan_review",
        route_after_review,
        {
            "execute": "tool_executor",
            "replan": "planner",
            "assemble": "agent_result_assembler",
            "failure": "failure_response",
        },
    )
    graph_builder.add_edge("tool_executor", "observation_validator")
    graph_builder.add_conditional_edges(
        "observation_validator",
        route_after_observation,
        {
            "validate": "result_validator",
        },
    )
    graph_builder.add_conditional_edges(
        "result_validator",
        route_after_capability_validation,
        {
            "replan": "planner",
            "assemble": "agent_result_assembler",
        },
    )
    graph_builder.add_edge("agent_result_assembler", "answer_synthesis")
    graph_builder.add_conditional_edges(
        "answer_synthesis",
        route_after_synthesis,
        {"guard": "output_guard", "done": "trace_finalizer"},
    )
    graph_builder.add_conditional_edges(
        "output_guard",
        route_after_output_guard,
        {
            "rewrite": "answer_synthesis",
            "retry": "output_guard",
            "done": "trace_finalizer",
        },
    )

    graph_builder.add_edge(
        "failure_response",
        "trace_finalizer",
    )
    graph_builder.add_edge("trace_finalizer", END)

    return graph_builder.compile(
        checkpointer=checkpointer
    )
