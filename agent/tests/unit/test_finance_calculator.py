from app.tools.finance_calculator import (
    emergency_fund_range,
    life_insurance_gap,
    yearly_expense_to_monthly,
)


def test_emergency_fund_range_with_yearly_expense_converted_to_monthly():
    result = emergency_fund_range(
        monthly_necessary_expense=15000,
        min_months=3,
        max_months=6,
    )

    assert result["monthly_necessary_expense"] == "15000"
    assert result["min_amount"] == "45000"
    assert result["max_amount"] == "90000"


def test_life_insurance_gap_normal_case():
    result = life_insurance_gap(
        family_required_funds=1480000,
        available_assets=250000,
        existing_life_insurance=300000,
        other_available_funds=0,
    )

    assert result["life_insurance_gap"] == "930000"


def test_life_insurance_gap_should_not_be_negative():
    result = life_insurance_gap(
        family_required_funds=500000,
        available_assets=600000,
        existing_life_insurance=300000,
        other_available_funds=0,
    )

    assert result["life_insurance_gap"] == "0"


def test_life_insurance_gap_accepts_string_numbers():
    result = life_insurance_gap(
        family_required_funds="1480000",
        available_assets="250000",
        existing_life_insurance="300000",
        other_available_funds="0",
    )

    assert result["life_insurance_gap"] == "930000"

def test_yearly_expense_to_monthly():
    result = yearly_expense_to_monthly(180000)

    assert result["yearly_necessary_expense"] == "180000"
    assert result["monthly_necessary_expense"] == "15000"
