from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from pydantic import BaseModel, ConfigDict, Field, model_validator


_MONEY_QUANTIZER = Decimal("0.01")


def normalize_money(value: Decimal) -> Decimal:
    """
    金额统一保留两位小数。

    计算过程使用 Decimal，避免 float 的二进制精度问题。
    """

    return value.quantize(
        _MONEY_QUANTIZER,
        rounding=ROUND_HALF_UP,
    )


class YearlyExpenseToMonthlyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yearly_necessary_expense: Decimal = Field(
        gt=0,
        max_digits=18,
        decimal_places=2,
        description="家庭年度必要支出，单位为人民币元。",
    )


class EmergencyFundRangeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_necessary_expense: Decimal = Field(
        gt=0,
        max_digits=18,
        decimal_places=2,
        description="家庭月度必要支出，单位为人民币元。",
    )

    min_months: int = Field(
        default=3,
        ge=1,
        le=120,
        description="紧急备用金建议覆盖的最小月份数。",
    )

    max_months: int = Field(
        default=6,
        ge=1,
        le=120,
        description="紧急备用金建议覆盖的最大月份数。",
    )

    @model_validator(mode="after")
    def validate_month_range(self) -> "EmergencyFundRangeInput":
        if self.min_months > self.max_months:
            raise ValueError(
                "min_months 不能大于 max_months。"
            )

        return self


class LifeInsuranceGapInput(BaseModel):
    """
    寿险保障缺口的确定性计算输入。

    保障需求 =
        年度必要支出 × 收入保障年数
        + 未偿债务
        + 子女教育资金
        + 其他家庭责任

    可抵扣资源 =
        可用于承担责任的资产
        + 已有寿险保额

    寿险缺口 =
        max(保障需求 - 可抵扣资源, 0)
    """

    model_config = ConfigDict(extra="forbid")

    annual_necessary_expense: Decimal = Field(
        gt=0,
        max_digits=18,
        decimal_places=2,
        description="家庭年度必要支出，单位为人民币元。",
    )

    coverage_years: int = Field(
        default=10,
        ge=1,
        le=50,
        description="需要由寿险覆盖家庭支出的年数。",
    )

    outstanding_debt: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        max_digits=18,
        decimal_places=2,
        description="房贷、消费贷等尚未偿还的家庭债务。",
    )

    education_fund: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        max_digits=18,
        decimal_places=2,
        description="需要预留的子女教育资金。",
    )

    other_family_responsibilities: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        max_digits=18,
        decimal_places=2,
        description="赡养、医疗等其他家庭责任金额。",
    )

    available_assets: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        max_digits=18,
        decimal_places=2,
        description="可实际用于承担家庭责任的资产。",
    )

    existing_life_insurance: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        max_digits=18,
        decimal_places=2,
        description="当前已经拥有的寿险保额。",
    )


def yearly_expense_to_monthly(
    *,
    yearly_necessary_expense: Decimal,
) -> dict:
    monthly_expense = normalize_money(
        yearly_necessary_expense / Decimal("12")
    )

    return {
        "yearly_necessary_expense": normalize_money(
            yearly_necessary_expense
        ),
        "monthly_necessary_expense": monthly_expense,
        "currency": "CNY",
        "formula": "yearly_necessary_expense / 12",
    }


def emergency_fund_range(
    *,
    monthly_necessary_expense: Decimal,
    min_months: int = 3,
    max_months: int = 6,
) -> dict:
    if min_months <= 0 or max_months <= 0:
        raise ValueError("备用金覆盖月份必须大于 0。")

    if min_months > max_months:
        raise ValueError(
            "最小覆盖月份不能大于最大覆盖月份。"
        )

    min_amount = normalize_money(
        monthly_necessary_expense * Decimal(min_months)
    )

    max_amount = normalize_money(
        monthly_necessary_expense * Decimal(max_months)
    )

    return {
        "monthly_necessary_expense": normalize_money(
            monthly_necessary_expense
        ),
        "min_months": min_months,
        "max_months": max_months,
        "min_amount": min_amount,
        "max_amount": max_amount,
        "currency": "CNY",
        "formula": "monthly_necessary_expense × months",
    }


def life_insurance_gap(
    *,
    annual_necessary_expense: Decimal,
    coverage_years: int = 10,
    outstanding_debt: Decimal = Decimal("0"),
    education_fund: Decimal = Decimal("0"),
    other_family_responsibilities: Decimal = Decimal("0"),
    available_assets: Decimal = Decimal("0"),
    existing_life_insurance: Decimal = Decimal("0"),
) -> dict:
    if coverage_years <= 0:
        raise ValueError("收入保障年数必须大于 0。")

    living_expense_need = normalize_money(
        annual_necessary_expense * Decimal(coverage_years)
    )

    total_responsibility = normalize_money(
        living_expense_need
        + outstanding_debt
        + education_fund
        + other_family_responsibilities
    )

    deductible_resources = normalize_money(
        available_assets + existing_life_insurance
    )

    raw_gap = total_responsibility - deductible_resources

    insurance_gap = normalize_money(
        max(raw_gap, Decimal("0"))
    )

    return {
        "annual_necessary_expense": normalize_money(
            annual_necessary_expense
        ),
        "coverage_years": coverage_years,
        "living_expense_need": living_expense_need,
        "outstanding_debt": normalize_money(outstanding_debt),
        "education_fund": normalize_money(education_fund),
        "other_family_responsibilities": normalize_money(
            other_family_responsibilities
        ),
        "total_responsibility": total_responsibility,
        "available_assets": normalize_money(available_assets),
        "existing_life_insurance": normalize_money(
            existing_life_insurance
        ),
        "deductible_resources": deductible_resources,
        "life_insurance_gap": insurance_gap,
        "currency": "CNY",
        "formula": (
            "annual_expense × coverage_years "
            "+ debt + education + other_responsibilities "
            "- available_assets - existing_life_insurance"
        ),
    }
