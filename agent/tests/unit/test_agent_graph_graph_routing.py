from app.agent_graph.graph import (
    build_finance_agent_graph,
    route_after_general_finance_answer,
    route_after_question_router,
    route_after_quality_gate,
)
from app.agent_graph.plan_compiler import (
    FINANCE_AGENT_STEP,
    GENERAL_FINANCE_ANSWER_STEP,
)


def test_route_after_question_router_should_go_to_general_answer():
    route = route_after_question_router(
        {
            "user_message": "什么是紧急备用金？",
            "user_id": "user_001",
            "thread_id": "thread_001",
            "execution_plan": [
                GENERAL_FINANCE_ANSWER_STEP,
            ],
        }
    )

    assert route == GENERAL_FINANCE_ANSWER_STEP


def test_route_after_question_router_should_go_to_finance_agent():
    route = route_after_question_router(
        {
            "user_message": "帮我计算寿险缺口。",
            "user_id": "user_001",
            "thread_id": "thread_001",
            "execution_plan": [
                FINANCE_AGENT_STEP,
            ],
        }
    )

    assert route == FINANCE_AGENT_STEP


def test_route_after_question_router_missing_plan_should_use_finance_agent():
    route = route_after_question_router(
        {
            "user_message": "帮我规划一下。",
            "user_id": "user_001",
            "thread_id": "thread_001",
        }
    )

    assert route == FINANCE_AGENT_STEP


def test_route_after_question_router_unknown_plan_should_use_finance_agent():
    route = route_after_question_router(
        {
            "user_message": "帮我规划一下。",
            "user_id": "user_001",
            "thread_id": "thread_001",
            "execution_plan": [
                "unknown_step",
            ],
        }
    )

    assert route == FINANCE_AGENT_STEP


def test_route_after_general_finance_answer_success_should_end():
    route = route_after_general_finance_answer(
        {
            "user_message": "什么是紧急备用金？",
            "user_id": "user_001",
            "thread_id": "thread_001",
            "execution_plan": [
                GENERAL_FINANCE_ANSWER_STEP,
            ],
            "final_answer": "紧急备用金是预留的应急资金。",
            "finish_reason": "langgraph_general_finance_answer",
        }
    )

    assert route == "end"


def test_route_after_general_finance_answer_failure_should_go_to_finance_agent():
    route = route_after_general_finance_answer(
        {
            "user_message": "什么是紧急备用金？",
            "user_id": "user_001",
            "thread_id": "thread_001",
            "execution_plan": [
                FINANCE_AGENT_STEP,
            ],
            "final_answer": "",
            "finish_reason": "general_finance_answer_error",
        }
    )

    assert route == FINANCE_AGENT_STEP


def test_route_after_quality_gate_should_go_to_fallback():
    route = route_after_quality_gate(
        {
            "user_message": "什么是紧急备用金？",
            "user_id": "user_001",
            "thread_id": "thread_001",
            "needs_general_finance_fallback": True,
        }
    )

    assert route == "fallback"


def test_route_after_quality_gate_should_end():
    route = route_after_quality_gate(
        {
            "user_message": "什么是紧急备用金？",
            "user_id": "user_001",
            "thread_id": "thread_001",
            "needs_general_finance_fallback": False,
        }
    )

    assert route == "end"


def test_build_finance_agent_graph_should_compile():
    graph = build_finance_agent_graph()

    assert graph is not None
