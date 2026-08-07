from app.tools.tool_registry import execute_tool


def test_execute_yearly_expense_to_monthly():
    result = execute_tool(
        "yearly_expense_to_monthly",
        {
            "yearly_necessary_expense": 180000,
        },
    )

    assert result["monthly_necessary_expense"] == "15000"


def test_execute_emergency_fund_range():
    result = execute_tool(
        "emergency_fund_range",
        {
            "monthly_necessary_expense": 15000,
            "min_months": 3,
            "max_months": 6,
        },
    )

    assert result["min_amount"] == "45000"
    assert result["max_amount"] == "90000"


def test_execute_life_insurance_gap():
    result = execute_tool(
        "life_insurance_gap",
        {
            "family_required_funds": 1480000,
            "available_assets": 250000,
            "existing_life_insurance": 300000,
            "other_available_funds": 0,
        },
    )

    assert result["life_insurance_gap"] == "930000"
