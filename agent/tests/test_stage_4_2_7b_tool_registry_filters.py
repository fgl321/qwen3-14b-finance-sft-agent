from __future__ import annotations

from app.tools.runtime_registry import (
    build_production_tool_registry,
)


ALL_FINANCIAL_TOOLS = [
    "asset_allocation_rebalance",
    "bond_analytics",
    "cashflow_npv_irr",
    "compound_interest_projection",
    "emergency_fund_range",
    "financial_ratio_analysis",
    "life_insurance_gap",
    "loan_amortization_compare",
    "portfolio_risk_metrics",
    "yearly_expense_to_monthly",
]


def _tool_names(
    definitions: list[dict],
) -> list[str]:
    return [
        str(item["function"]["name"])
        for item in definitions
    ]


def test_empty_names_do_not_override_allowed_group() -> None:
    registry = build_production_tool_registry()

    definitions = registry.get_llm_tool_definitions(
        allowed_tool_names=frozenset(),
        allowed_tool_groups=frozenset(
            {"financial_calculation"}
        ),
    )

    assert _tool_names(definitions) == ALL_FINANCIAL_TOOLS


def test_non_empty_names_and_groups_use_intersection() -> None:
    registry = build_production_tool_registry()

    definitions = registry.get_llm_tool_definitions(
        allowed_tool_names=frozenset(
            {
                "yearly_expense_to_monthly",
                "life_insurance_gap",
            }
        ),
        allowed_tool_groups=frozenset(
            {"financial_calculation"}
        ),
    )

    assert _tool_names(definitions) == [
        "life_insurance_gap",
        "yearly_expense_to_monthly",
    ]


def test_empty_filters_mean_no_filter() -> None:
    registry = build_production_tool_registry()

    definitions = registry.get_llm_tool_definitions(
        allowed_tool_names=frozenset(),
        allowed_tool_groups=frozenset(),
    )

    assert _tool_names(definitions) == ALL_FINANCIAL_TOOLS
