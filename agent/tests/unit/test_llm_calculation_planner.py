from __future__ import annotations

from app.agent_graph.llm_calculation_planner import (
    _plan_from_dict,
    validate_basic_llm_calculation_plan,
)


def test_plan_from_dict_builds_steps() -> None:
    plan = _plan_from_dict(
        {
            "supported": True,
            "steps": [
                {
                    "tool_name": "yearly_expense_to_monthly",
                    "arguments": {
                        "yearly_necessary_expense": 180000
                    },
                },
                {
                    "tool_name": "emergency_fund_range",
                    "arguments": {
                        "monthly_necessary_expense": 15000
                    },
                },
            ],
            "missing_fields": [],
            "reason": "两步计算。",
        }
    )

    assert plan.supported is True
    assert len(plan.steps) == 2
    assert plan.steps[0].tool_name == "yearly_expense_to_monthly"
    assert (
        plan.steps[0].arguments["yearly_necessary_expense"]
        == 180000
    )
    assert plan.reason == "两步计算。"


def test_plan_from_dict_defaults_reason() -> None:
    plan = _plan_from_dict(
        {
            "supported": False,
            "steps": [],
            "missing_fields": ["yearly_necessary_expense"],
        }
    )
    assert plan.supported is False
    assert plan.reason == "LLM 已生成金融计算计划。"
    assert plan.missing_fields == ["yearly_necessary_expense"]


def test_validate_rejects_empty_steps_when_supported() -> None:
    plan = _plan_from_dict(
        {
            "supported": True,
            "steps": [],
            "missing_fields": [],
            "reason": "空计划。",
        }
    )
    result = validate_basic_llm_calculation_plan(plan)
    assert result.supported is False
    assert "steps 为空" in result.reason


def test_validate_rejects_unknown_tool() -> None:
    plan = _plan_from_dict(
        {
            "supported": True,
            "steps": [
                {
                    "tool_name": "not_a_real_tool",
                    "arguments": {},
                }
            ],
            "missing_fields": [],
            "reason": "未知工具。",
        }
    )
    result = validate_basic_llm_calculation_plan(plan)
    assert result.supported is False
    assert "不允许的计算工具" in result.reason


def test_validate_passes_allowed_plan() -> None:
    plan = _plan_from_dict(
        {
            "supported": True,
            "steps": [
                {
                    "tool_name": "yearly_expense_to_monthly",
                    "arguments": {
                        "yearly_necessary_expense": 180000
                    },
                }
            ],
            "missing_fields": [],
            "reason": "合法计划。",
        }
    )
    result = validate_basic_llm_calculation_plan(plan)
    assert result.supported is True
    assert len(result.steps) == 1
