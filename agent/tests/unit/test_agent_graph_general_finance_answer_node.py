import asyncio

from app.agent_graph import nodes
from app.agent_graph.plan_compiler import (
    FINANCE_AGENT_STEP,
)


def test_general_finance_answer_node_should_return_answer(
    monkeypatch,
):
    async def fake_general_finance_answer(
        user_message: str,
    ):
        assert user_message == "什么是紧急备用金？"

        return (
            "紧急备用金是为了应对失业、疾病或突发支出而预留的资金。",
            {
                "model": "fake-model",
                "usage": {
                    "total_tokens": 20,
                },
            },
        )

    monkeypatch.setattr(
        nodes,
        "_call_deepseek_for_general_finance_answer",
        fake_general_finance_answer,
    )

    result = asyncio.run(
        nodes.general_finance_answer_node(
            {
                "user_message": "什么是紧急备用金？",
                "user_id": "user_001",
                "thread_id": "thread_001",
                "question_route_detail": {
                    "router": "hard_rule",
                },
            }
        )
    )

    assert result["final_answer"] == (
        "紧急备用金是为了应对失业、疾病或突发支出而预留的资金。"
    )

    assert result["finish_reason"] == (
        "langgraph_general_finance_answer"
    )

    assert result["fallback_used"] is False

    assert result["executed_tools"] == []

    assert result[
        "usage"
    ]["langgraph_general_finance_answer"]["used"] is True

    assert result[
        "usage"
    ]["langgraph_general_finance_answer"]["model"] == (
        "fake-model"
    )

    assert result["agent_result"]["answer"] == (
        result["final_answer"]
    )


def test_general_finance_answer_failure_should_use_finance_agent_plan(
    monkeypatch,
):
    async def raise_general_finance_answer_error(
        user_message: str,
    ):
        raise RuntimeError(
            "direct answer failed"
        )

    monkeypatch.setattr(
        nodes,
        "_call_deepseek_for_general_finance_answer",
        raise_general_finance_answer_error,
    )

    result = asyncio.run(
        nodes.general_finance_answer_node(
            {
                "user_message": "什么是紧急备用金？",
                "user_id": "user_001",
                "thread_id": "thread_001",
                "question_route_detail": {
                    "router": "hard_rule",
                },
            }
        )
    )

    assert result["final_answer"] == ""

    assert result["finish_reason"] == (
        "general_finance_answer_error"
    )

    assert result["execution_plan"] == [
        FINANCE_AGENT_STEP,
    ]

    assert "error" not in result

    direct_answer_detail = result[
        "question_route_detail"
    ]["general_finance_answer"]

    assert direct_answer_detail["used"] is False

    assert direct_answer_detail[
        "fallback_to"
    ] == FINANCE_AGENT_STEP

    assert "RuntimeError" in direct_answer_detail[
        "error"
    ]


def test_general_finance_answer_empty_content_should_fallback(
    monkeypatch,
):
    async def return_empty_answer(
        user_message: str,
    ):
        return (
            "",
            {
                "model": "fake-model",
                "usage": {},
            },
        )

    monkeypatch.setattr(
        nodes,
        "_call_deepseek_for_general_finance_answer",
        return_empty_answer,
    )

    result = asyncio.run(
        nodes.general_finance_answer_node(
            {
                "user_message": "什么是紧急备用金？",
                "user_id": "user_001",
                "thread_id": "thread_001",
            }
        )
    )

    assert result["execution_plan"] == [
        FINANCE_AGENT_STEP,
    ]

    assert result["finish_reason"] == (
        "general_finance_answer_error"
    )

    assert "空内容" in result[
        "question_route_detail"
    ]["general_finance_answer"]["error"]
