from decimal import Decimal, InvalidOperation


class FinanceCalculationError(ValueError):
    pass


def to_decimal(value: str | int | float) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise FinanceCalculationError(f"非法数字：{value}") from exc


def emergency_fund_range(
    monthly_necessary_expense: str | int | float,
    min_months: int = 3,
    max_months: int = 6,
) -> dict[str, str]:
    monthly = to_decimal(monthly_necessary_expense)

    min_amount = monthly * Decimal(min_months)
    max_amount = monthly * Decimal(max_months)

    return {
        "monthly_necessary_expense": str(monthly),
        "min_months": str(min_months),
        "max_months": str(max_months),
        "min_amount": str(min_amount),
        "max_amount": str(max_amount),
        "formula": "monthly_necessary_expense × months",
    }


def life_insurance_gap(
    family_required_funds: str | int | float,
    available_assets: str | int | float,
    existing_life_insurance: str | int | float,
    other_available_funds: str | int | float = 0,
) -> dict[str, str]:
    required = to_decimal(family_required_funds)
    assets = to_decimal(available_assets)
    insurance = to_decimal(existing_life_insurance)
    other = to_decimal(other_available_funds)

    gap = required - assets - insurance - other

    if gap < 0:
        gap = Decimal("0")

    return {
        "family_required_funds": str(required),
        "available_assets": str(assets),
        "existing_life_insurance": str(insurance),
        "other_available_funds": str(other),
        "life_insurance_gap": str(gap),
        "formula": (
            "family_required_funds - available_assets "
            "- existing_life_insurance - other_available_funds"
        ),
    }

def yearly_expense_to_monthly(
    yearly_necessary_expense: str | int | float,
) -> dict[str, str]:
    yearly = to_decimal(yearly_necessary_expense)
    monthly = yearly / Decimal("12")

    return {
        "yearly_necessary_expense": str(yearly),
        "monthly_necessary_expense": str(monthly),
        "formula": "yearly_necessary_expense / 12",
    }
