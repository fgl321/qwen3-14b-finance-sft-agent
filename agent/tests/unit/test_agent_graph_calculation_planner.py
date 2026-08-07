

import pytest
from app.agent_graph.calculation_planner import (
    EMERGENCY_FUND_RANGE_TOOL,
    YEARLY_EXPENSE_TO_MONTHLY_TOOL,
    build_calculation_plan,
)


def test_build_plan_for_yearly_expense_to_monthly_only():
    plan = build_calculation_plan(
        "我家年度必要支出是18万元，请帮我换算成月度必要支出。"
    )

    assert plan.supported is True
    assert len(plan.steps) == 1

    assert plan.steps[0].tool_name == YEARLY_EXPENSE_TO_MONTHLY_TOOL
    assert plan.steps[0].arguments == {
        "yearly_necessary_expense": 180000,
    }


def test_build_plan_for_monthly_expense_to_emergency_fund():
    plan = build_calculation_plan(
        "我家月度必要支出是15000元，请计算3到6个月紧急备用金范围。"
    )

    assert plan.supported is True
    assert len(plan.steps) == 1

    assert plan.steps[0].tool_name == EMERGENCY_FUND_RANGE_TOOL
    assert plan.steps[0].arguments == {
        "monthly_necessary_expense": 15000,
        "min_months": 3,
        "max_months": 6,
    }

@pytest.mark.skip(reason="Stage 4.2 改为 LLM calculation planner 处理复杂链式计算")
def test_build_plan_for_yearly_expense_to_emergency_fund_chain():
    plan = build_calculation_plan(
        "我家年度必要支出是18万元，请帮我换算成月度必要支出，并计算3到6个月紧急备用金范围。"
    )

    assert plan.supported is True
    assert len(plan.steps) == 2

    assert plan.steps[0].tool_name == YEARLY_EXPENSE_TO_MONTHLY_TOOL
    assert plan.steps[0].arguments == {
        "yearly_necessary_expense": 180000,
    }

    assert plan.steps[1].tool_name == EMERGENCY_FUND_RANGE_TOOL
    assert plan.steps[1].arguments == {
        "monthly_necessary_expense": {
            "from_step": YEARLY_EXPENSE_TO_MONTHLY_TOOL,
            "field": "monthly_necessary_expense",
        },
        "min_months": 3,
        "max_months": 6,
    }


def test_build_plan_should_default_emergency_fund_month_range_to_3_and_6():
    plan = build_calculation_plan(
        "我家月度必要支出是15000元，请计算紧急备用金范围。"
    )

    assert plan.supported is True
    assert len(plan.steps) == 1
    assert plan.steps[0].arguments["min_months"] == 3
    assert plan.steps[0].arguments["max_months"] == 6


def test_build_plan_should_report_missing_monthly_expense_for_emergency_fund():
    plan = build_calculation_plan(
        "请帮我计算紧急备用金范围。"
    )

    assert plan.supported is False
    assert plan.steps == []
    assert plan.missing_fields == [
        "monthly_necessary_expense",
    ]


def test_build_plan_should_not_support_unmatched_question():
    plan = build_calculation_plan(
        "什么是资产配置？"
    )

    assert plan.supported is False
    assert plan.steps == []
    assert plan.missing_fields == []


def test_calculation_plan_to_dict():
    plan = build_calculation_plan(
        "我家年度必要支出是18万元，请帮我换算成月度必要支出。"
    )

    data = plan.to_dict()

    assert data["supported"] is True
    assert data["steps"] == [
        {
            "tool_name": YEARLY_EXPENSE_TO_MONTHLY_TOOL,
            "arguments": {
                "yearly_necessary_expense": 180000,
            },
        }
    ]
    assert data["missing_fields"] == []
