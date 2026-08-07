import asyncio

from app.agent_graph import graph as graph_module
from app.agent_graph.plan_compiler import (
    FINANCE_AGENT_STEP,
    GENERAL_FINANCE_ANSWER_STEP,
)


def _build_input_state(
    user_message: str,
) -> dict[str, object]:
    return {
        "user_message": user_message,
        "user_id": "user_001",
        "thread_id": "thread_001",
        "request_id": "request_001",
        "tenant_id": "tenant_001",
        "knowledge_base_id": "kb_finance_basic",
        "history_messages": [],
    }


def test_graph_should_end_after_general_finance_answer(
    monkeypatch,
):
    called_nodes: list[str] = []

    async def fake_question_router_node(state):
        called_nodes.append("question_router")

        return {
            "question_capabilities": [
                "general_explanation",
            ],
            "question_router": "hard_rule",
            "question_router_confidence": "high",
            "question_router_reason": "测试：概念解释题",
            "question_router_used_fallback": False,
            "question_router_matched_rules": [
                "explicit_concept_explanation",
            ],
            "question_route_detail": {
                "router": "hard_rule",
            },
            "execution_plan": [
                GENERAL_FINANCE_ANSWER_STEP,
            ],
        }

    async def fake_general_finance_answer_node(state):
        called_nodes.append("general_finance_answer")

        return {
            "final_answer": "紧急备用金是为了应对突发支出预留的资金。",
            "agent_result": {
                "answer": "紧急备用金是为了应对突发支出预留的资金。",
            },
            "executed_tools": [],
            "usage": {
                "langgraph_general_finance_answer": {
                    "used": True,
                },
            },
            "finish_reason": (
                "langgraph_general_finance_answer"
            ),
            "message_count": 0,
            "safety_check": {},
            "fallback_used": False,
        }

    async def fake_finance_agent_node(state):
        called_nodes.append("finance_agent")

        return {
            "final_answer": "不应该执行到这里",
        }

    async def fake_answer_quality_gate_node(state):
        called_nodes.append("answer_quality_gate")

        return {
            "needs_general_finance_fallback": False,
        }

    monkeypatch.setattr(
        graph_module,
        "question_router_node",
        fake_question_router_node,
    )

    monkeypatch.setattr(
        graph_module,
        "general_finance_answer_node",
        fake_general_finance_answer_node,
    )

    monkeypatch.setattr(
        graph_module,
        "finance_agent_node",
        fake_finance_agent_node,
    )

    monkeypatch.setattr(
        graph_module,
        "answer_quality_gate_node",
        fake_answer_quality_gate_node,
    )

    graph = graph_module.build_finance_agent_graph()

    result = asyncio.run(
        graph.ainvoke(
            _build_input_state(
                "什么是紧急备用金？"
            )
        )
    )

    assert called_nodes == [
        "question_router",
        "general_finance_answer",
    ]

    assert result["final_answer"] == (
        "紧急备用金是为了应对突发支出预留的资金。"
    )

    assert result["finish_reason"] == (
        "langgraph_general_finance_answer"
    )

    assert result["execution_plan"] == [
        GENERAL_FINANCE_ANSWER_STEP,
    ]


def test_graph_should_run_finance_agent_for_complex_question(
    monkeypatch,
):
    called_nodes: list[str] = []

    async def fake_question_router_node(state):
        called_nodes.append("question_router")

        return {
            "question_capabilities": [
                "knowledge_retrieval",
                "financial_calculation",
                "complex_reasoning",
            ],
            "question_router": "llm_semantic_router",
            "question_router_confidence": "medium",
            "question_router_reason": "测试：复杂金融规划题",
            "question_router_used_fallback": False,
            "question_router_matched_rules": [],
            "question_route_detail": {
                "router": "llm_semantic_router",
            },
            "execution_plan": [
                FINANCE_AGENT_STEP,
            ],
        }

    async def fake_general_finance_answer_node(state):
        called_nodes.append("general_finance_answer")

        return {
            "final_answer": "不应该执行到这里",
        }

    async def fake_finance_agent_node(state):
        called_nodes.append("finance_agent")

        return {
            "final_answer": "这是旧 FinanceAgent 的复杂规划回答。",
            "agent_result": {
                "answer": "这是旧 FinanceAgent 的复杂规划回答。",
            },
            "executed_tools": [
                {
                    "tool_name": "search_knowledge_base",
                    "result": {
                        "retrieved_count": 1,
                        "evidence_assessment": {
                            "sufficient": True,
                        },
                    },
                }
            ],
            "usage": {},
            "finish_reason": "stop",
            "message_count": 1,
            "safety_check": {},
            "fallback_used": False,
        }

    async def fake_answer_quality_gate_node(state):
        called_nodes.append("answer_quality_gate")

        assert state["final_answer"] == (
            "这是旧 FinanceAgent 的复杂规划回答。"
        )

        return {
            "quality_gate": {
                "needs_general_finance_fallback": False,
                "reason": "测试：不需要兜底",
            },
            "needs_general_finance_fallback": False,
            "fallback_reason": "测试：不需要兜底",
        }

    monkeypatch.setattr(
        graph_module,
        "question_router_node",
        fake_question_router_node,
    )

    monkeypatch.setattr(
        graph_module,
        "general_finance_answer_node",
        fake_general_finance_answer_node,
    )

    monkeypatch.setattr(
        graph_module,
        "finance_agent_node",
        fake_finance_agent_node,
    )

    monkeypatch.setattr(
        graph_module,
        "answer_quality_gate_node",
        fake_answer_quality_gate_node,
    )

    graph = graph_module.build_finance_agent_graph()

    result = asyncio.run(
        graph.ainvoke(
            _build_input_state(
                "根据我的家庭情况计算寿险缺口并给出规划。"
            )
        )
    )

    assert called_nodes == [
        "question_router",
        "finance_agent",
        "answer_quality_gate",
    ]

    assert result["final_answer"] == (
        "这是旧 FinanceAgent 的复杂规划回答。"
    )

    assert result["execution_plan"] == [
        FINANCE_AGENT_STEP,
    ]

    assert result[
        "needs_general_finance_fallback"
    ] is False


def test_graph_should_fallback_to_finance_agent_when_direct_answer_fails(
    monkeypatch,
):
    called_nodes: list[str] = []

    async def fake_question_router_node(state):
        called_nodes.append("question_router")

        return {
            "question_capabilities": [
                "general_explanation",
            ],
            "question_router": "hard_rule",
            "question_router_confidence": "high",
            "question_router_reason": "测试：概念解释题",
            "question_router_used_fallback": False,
            "question_router_matched_rules": [
                "explicit_concept_explanation",
            ],
            "question_route_detail": {
                "router": "hard_rule",
            },
            "execution_plan": [
                GENERAL_FINANCE_ANSWER_STEP,
            ],
        }

    async def fake_general_finance_answer_node(state):
        called_nodes.append("general_finance_answer")

        return {
            "final_answer": "",
            "finish_reason": (
                "general_finance_answer_error"
            ),
            "question_route_detail": {
                "router": "hard_rule",
                "general_finance_answer": {
                    "used": False,
                    "error": "RuntimeError: direct answer failed",
                    "fallback_to": FINANCE_AGENT_STEP,
                },
            },
            "execution_plan": [
                FINANCE_AGENT_STEP,
            ],
        }

    async def fake_finance_agent_node(state):
        called_nodes.append("finance_agent")

        assert state["execution_plan"] == [
            FINANCE_AGENT_STEP,
        ]

        return {
            "final_answer": "这是旧 FinanceAgent 的兜底回答。",
            "agent_result": {
                "answer": "这是旧 FinanceAgent 的兜底回答。",
            },
            "executed_tools": [],
            "usage": {},
            "finish_reason": "stop",
            "message_count": 1,
            "safety_check": {},
            "fallback_used": False,
        }

    async def fake_answer_quality_gate_node(state):
        called_nodes.append("answer_quality_gate")

        return {
            "quality_gate": {
                "needs_general_finance_fallback": False,
                "reason": "测试：不需要兜底",
            },
            "needs_general_finance_fallback": False,
            "fallback_reason": "测试：不需要兜底",
        }

    monkeypatch.setattr(
        graph_module,
        "question_router_node",
        fake_question_router_node,
    )

    monkeypatch.setattr(
        graph_module,
        "general_finance_answer_node",
        fake_general_finance_answer_node,
    )

    monkeypatch.setattr(
        graph_module,
        "finance_agent_node",
        fake_finance_agent_node,
    )

    monkeypatch.setattr(
        graph_module,
        "answer_quality_gate_node",
        fake_answer_quality_gate_node,
    )

    graph = graph_module.build_finance_agent_graph()

    result = asyncio.run(
        graph.ainvoke(
            _build_input_state(
                "什么是紧急备用金？"
            )
        )
    )

    assert called_nodes == [
        "question_router",
        "general_finance_answer",
        "finance_agent",
        "answer_quality_gate",
    ]

    assert result["final_answer"] == (
        "这是旧 FinanceAgent 的兜底回答。"
    )

    assert result["execution_plan"] == [
        FINANCE_AGENT_STEP,
    ]

    assert result["finish_reason"] == "stop"
