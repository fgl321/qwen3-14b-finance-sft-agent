
import pytest

from app.agent_graph.question_router import (
    QuestionCapability,
    RuleConfidence,
    normalize_capabilities,
    route_by_hard_rules,
)


def test_simple_finance_concept_should_use_general_explanation():
    result = route_by_hard_rules(
        "什么是紧急备用金？"
    )

    assert result.capabilities == (
        QuestionCapability.GENERAL_EXPLANATION,
    )
    assert result.confidence == RuleConfidence.MEDIUM
    assert result.needs_semantic_router is False


def test_explicit_document_request_should_require_knowledge():
    result = route_by_hard_rules(
        "请根据我上传的文档回答这个问题。"
    )

    assert result.has_capability(
        QuestionCapability.KNOWLEDGE_RETRIEVAL
    )
    assert result.confidence == RuleConfidence.HIGH
    assert result.needs_semantic_router is False


def test_explicit_calculation_should_require_calculator():
    result = route_by_hard_rules(
        "我的年度必要支出是18万元，"
        "紧急备用金应该准备多少？"
    )

    assert result.capabilities == (
        QuestionCapability.FINANCIAL_CALCULATION,
    )
    assert result.confidence == RuleConfidence.HIGH


def test_one_question_can_require_multiple_capabilities():
    result = route_by_hard_rules(
        "请根据上传的保单，计算寿险保障缺口，"
        "并给出调整方案。"
    )

    assert result.capabilities == (
        QuestionCapability.KNOWLEDGE_RETRIEVAL,
        QuestionCapability.FINANCIAL_CALCULATION,
        QuestionCapability.COMPLEX_REASONING,
    )

    assert result.confidence == RuleConfidence.HIGH
    assert result.needs_semantic_router is False


def test_previous_context_should_require_memory_read():
    result = route_by_hard_rules(
        "我刚才说的家庭年度支出是多少？"
    )

    assert result.capabilities == (
        QuestionCapability.MEMORY_READ,
    )


def test_number_alone_must_not_be_treated_as_calculation():
    result = route_by_hard_rules(
        "我年收入18万元，应该怎么规划？"
    )

    assert (
        QuestionCapability.FINANCIAL_CALCULATION
        not in result.capabilities
    )

    assert result.capabilities == (
        QuestionCapability.COMPLEX_REASONING,
    )


def test_ambiguous_natural_language_should_use_semantic_router():
    result = route_by_hard_rules(
        "我是不是应该先留一笔应急的钱？"
    )

    assert result.capabilities == ()
    assert result.needs_semantic_router is True
    assert result.confidence == RuleConfidence.LOW


def test_empty_input_should_use_conservative_fallback():
    result = route_by_hard_rules("   ")

    assert result.is_empty_input is True
    assert result.capabilities == (
        QuestionCapability.COMPLEX_REASONING,
    )
    assert result.confidence == RuleConfidence.LOW


def test_normalize_capabilities_should_validate_deduplicate_and_sort():
    result = normalize_capabilities(
        [
            "complex_reasoning",
            "knowledge_retrieval",
            "knowledge_retrieval",
            QuestionCapability.MEMORY_READ,
        ]
    )

    assert result == (
        QuestionCapability.MEMORY_READ,
        QuestionCapability.KNOWLEDGE_RETRIEVAL,
        QuestionCapability.COMPLEX_REASONING,
    )


def test_invalid_capability_should_raise_value_error():
    with pytest.raises(ValueError):
        normalize_capabilities(
            ["insurance_question"]
        )
