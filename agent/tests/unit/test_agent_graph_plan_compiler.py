import pytest

from app.agent_graph.plan_compiler import (
    FINANCE_AGENT_STEP,
    GENERAL_FINANCE_ANSWER_STEP,
    compile_execution_plan,
)
from app.agent_graph.question_router import QuestionCapability


def test_only_general_explanation_should_use_general_finance_answer():
    plan = compile_execution_plan(
        [
            QuestionCapability.GENERAL_EXPLANATION,
        ]
    )

    assert plan == [GENERAL_FINANCE_ANSWER_STEP]


def test_general_explanation_string_should_be_supported():
    plan = compile_execution_plan(
        [
            "general_explanation",
        ]
    )

    assert plan == [GENERAL_FINANCE_ANSWER_STEP]


def test_financial_calculation_should_use_finance_agent():
    plan = compile_execution_plan(
        [
            QuestionCapability.FINANCIAL_CALCULATION,
        ]
    )

    assert plan == [FINANCE_AGENT_STEP]


def test_knowledge_retrieval_should_use_finance_agent():
    plan = compile_execution_plan(
        [
            QuestionCapability.KNOWLEDGE_RETRIEVAL,
        ]
    )

    assert plan == [FINANCE_AGENT_STEP]


def test_memory_read_should_use_finance_agent():
    plan = compile_execution_plan(
        [
            QuestionCapability.MEMORY_READ,
        ]
    )

    assert plan == [FINANCE_AGENT_STEP]


def test_complex_reasoning_should_use_finance_agent():
    plan = compile_execution_plan(
        [
            QuestionCapability.COMPLEX_REASONING,
        ]
    )

    assert plan == [FINANCE_AGENT_STEP]


def test_multiple_capabilities_should_use_finance_agent():
    plan = compile_execution_plan(
        [
            QuestionCapability.KNOWLEDGE_RETRIEVAL,
            QuestionCapability.FINANCIAL_CALCULATION,
            QuestionCapability.COMPLEX_REASONING,
        ]
    )

    assert plan == [FINANCE_AGENT_STEP]


def test_general_explanation_with_complex_reasoning_should_use_finance_agent():
    plan = compile_execution_plan(
        [
            QuestionCapability.GENERAL_EXPLANATION,
            QuestionCapability.COMPLEX_REASONING,
        ]
    )

    assert plan == [FINANCE_AGENT_STEP]


def test_empty_capabilities_should_raise_error():
    with pytest.raises(
        ValueError,
        match="capabilities 不能为空",
    ):
        compile_execution_plan([])


def test_unknown_capability_should_raise_error():
    with pytest.raises(
        ValueError,
        match="发现未知能力",
    ):
        compile_execution_plan(
            [
                "unknown_capability",
            ]
        )
