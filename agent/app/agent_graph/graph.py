from langgraph.graph import END, StateGraph

from app.agent_graph.nodes import (
    answer_quality_gate_node,
    finance_agent_node,
    general_finance_answer_node,
    general_finance_fallback_node,
    question_router_node,
)
from app.agent_graph.plan_compiler import (
    FINANCE_AGENT_STEP,
    GENERAL_FINANCE_ANSWER_STEP,
)
from app.agent_graph.state import FinanceAgentGraphState


def route_after_question_router(
    state: FinanceAgentGraphState,
) -> str:
    """
    问题路由之后的执行分发。

    general_finance_answer:
        进入通用金融知识直接回答节点。

    finance_agent:
        进入旧 FinanceAgent 节点。

    安全策略：
        如果 execution_plan 缺失或出现未知步骤，
        默认进入旧 FinanceAgent，避免请求中断。
    """
    execution_plan = state.get("execution_plan") or []

    if not execution_plan:
        return FINANCE_AGENT_STEP

    next_step = str(execution_plan[0]).strip()

    if next_step == GENERAL_FINANCE_ANSWER_STEP:
        return GENERAL_FINANCE_ANSWER_STEP

    if next_step == FINANCE_AGENT_STEP:
        return FINANCE_AGENT_STEP

    return FINANCE_AGENT_STEP


def route_after_general_finance_answer(
    state: FinanceAgentGraphState,
) -> str:
    """
    通用金融直接回答之后的路由。

    如果直接回答成功：
        END

    如果直接回答失败：
        general_finance_answer_node 会把 execution_plan 改成
        ["finance_agent"]，这里再转入旧 FinanceAgent。
    """
    execution_plan = state.get("execution_plan") or []

    if FINANCE_AGENT_STEP in execution_plan:
        return FINANCE_AGENT_STEP

    return "end"


def route_after_quality_gate(
    state: FinanceAgentGraphState,
) -> str:
    """
    质量门控之后的路由。

    fallback:
        进入通用金融解释兜底节点。

    end:
        直接结束。
    """
    if state.get("needs_general_finance_fallback"):
        return "fallback"

    return "end"


def build_finance_agent_graph():
    """
    Stage 4.1 LangGraph 编排图。

    当前图结构：

    user input
        ↓
    question_router
        ↓
    条件判断：
        - 单纯通用金融概念解释：
            general_finance_answer
                - 成功：END
                - 失败：finance_agent
        - 需要知识库 / 计算 / 记忆 / 复杂推理：
            finance_agent
        ↓
    finance_agent
        ↓
    answer_quality_gate
        ↓
    条件判断：
        - 需要兜底：general_finance_fallback
        - 不需要兜底：END
        ↓
    END
    """
    graph_builder = StateGraph(
        FinanceAgentGraphState
    )

    graph_builder.add_node(
        "question_router",
        question_router_node,
    )

    graph_builder.add_node(
        "general_finance_answer",
        general_finance_answer_node,
    )

    graph_builder.add_node(
        "finance_agent",
        finance_agent_node,
    )

    graph_builder.add_node(
        "answer_quality_gate",
        answer_quality_gate_node,
    )

    graph_builder.add_node(
        "general_finance_fallback",
        general_finance_fallback_node,
    )

    graph_builder.set_entry_point(
        "question_router"
    )

    graph_builder.add_conditional_edges(
        "question_router",
        route_after_question_router,
        {
            GENERAL_FINANCE_ANSWER_STEP: (
                "general_finance_answer"
            ),
            FINANCE_AGENT_STEP: "finance_agent",
        },
    )

    graph_builder.add_conditional_edges(
        "general_finance_answer",
        route_after_general_finance_answer,
        {
            FINANCE_AGENT_STEP: "finance_agent",
            "end": END,
        },
    )

    graph_builder.add_edge(
        "finance_agent",
        "answer_quality_gate",
    )

    graph_builder.add_conditional_edges(
        "answer_quality_gate",
        route_after_quality_gate,
        {
            "fallback": "general_finance_fallback",
            "end": END,
        },
    )

    graph_builder.add_edge(
        "general_finance_fallback",
        END,
    )

    return graph_builder.compile()
