from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP, localcontext

from pydantic import BaseModel, ConfigDict, Field, model_validator


MONEY = Decimal("0.01")
RATIO = Decimal("0.000001")
PERCENT = Decimal("0.0001")


def _decimal(value: float | int | Decimal) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY, rounding=ROUND_HALF_UP)


def _ratio(value: Decimal) -> Decimal:
    return value.quantize(RATIO, rounding=ROUND_HALF_UP)


def _pct(value: Decimal) -> Decimal:
    return value.quantize(PERCENT, rounding=ROUND_HALF_UP)


class StrictFinancialInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompoundInterestInput(StrictFinancialInput):
    initial_principal: Decimal = Field(default=Decimal("0"), ge=0, le=10**15)
    monthly_contribution: Decimal = Field(default=Decimal("0"), ge=0, le=10**12)
    annual_rate_percent: Decimal = Field(ge=-99, le=1000)
    years: int = Field(ge=1, le=100)
    contribution_at_period_start: bool = False

    @model_validator(mode="after")
    def require_cashflow(self) -> "CompoundInterestInput":
        if self.initial_principal == 0 and self.monthly_contribution == 0:
            raise ValueError("初始本金和每月投入不能同时为 0。")
        return self


def compound_interest_projection(
    *,
    initial_principal: Decimal = Decimal("0"),
    monthly_contribution: Decimal = Decimal("0"),
    annual_rate_percent: Decimal,
    years: int,
    contribution_at_period_start: bool = False,
) -> dict:
    months = years * 12
    monthly_rate = annual_rate_percent / Decimal("1200")
    with localcontext() as ctx:
        ctx.prec = 40
        factor = (Decimal("1") + monthly_rate) ** months
        principal_future = initial_principal * factor
        if monthly_rate == 0:
            contribution_future = monthly_contribution * months
        else:
            contribution_future = monthly_contribution * (
                (factor - Decimal("1")) / monthly_rate
            )
            if contribution_at_period_start:
                contribution_future *= Decimal("1") + monthly_rate
    total_contributed = initial_principal + monthly_contribution * months
    future_value = principal_future + contribution_future
    return {
        "future_value": _money(future_value),
        "total_contributed": _money(total_contributed),
        "estimated_return": _money(future_value - total_contributed),
        "months": months,
        "annual_rate_percent": _pct(annual_rate_percent),
        "contribution_timing": (
            "period_start" if contribution_at_period_start else "period_end"
        ),
        "currency": "CNY",
        "formula_version": "compound_monthly_v1",
    }


class LoanAmortizationInput(StrictFinancialInput):
    principal: Decimal = Field(gt=0, le=10**15)
    annual_rate_percent: Decimal = Field(ge=0, le=100)
    years: int = Field(ge=1, le=50)
    upfront_prepayment: Decimal = Field(default=Decimal("0"), ge=0, le=10**15)

    @model_validator(mode="after")
    def validate_prepayment(self) -> "LoanAmortizationInput":
        if self.upfront_prepayment >= self.principal:
            raise ValueError("提前还款金额必须小于贷款本金。")
        return self


def _loan_summary(principal: Decimal, rate: Decimal, months: int) -> dict:
    monthly_rate = rate / Decimal("1200")
    with localcontext() as ctx:
        ctx.prec = 40
        if monthly_rate == 0:
            equal_payment = principal / months
        else:
            factor = (Decimal("1") + monthly_rate) ** months
            equal_payment = principal * monthly_rate * factor / (factor - 1)
        equal_payment_total_interest = equal_payment * months - principal

        equal_principal_first = principal / months + principal * monthly_rate
        equal_principal_last = principal / months + (
            principal / months
        ) * monthly_rate
        equal_principal_total_interest = (
            principal * monthly_rate * Decimal(months + 1) / Decimal("2")
        )
    return {
        "equal_payment": {
            "monthly_payment": _money(equal_payment),
            "total_interest": _money(equal_payment_total_interest),
            "total_payment": _money(principal + equal_payment_total_interest),
        },
        "equal_principal": {
            "first_month_payment": _money(equal_principal_first),
            "last_month_payment": _money(equal_principal_last),
            "monthly_principal": _money(principal / months),
            "total_interest": _money(equal_principal_total_interest),
            "total_payment": _money(principal + equal_principal_total_interest),
        },
    }


def loan_amortization_compare(
    *,
    principal: Decimal,
    annual_rate_percent: Decimal,
    years: int,
    upfront_prepayment: Decimal = Decimal("0"),
) -> dict:
    months = years * 12
    baseline = _loan_summary(principal, annual_rate_percent, months)
    remaining = principal - upfront_prepayment
    after = _loan_summary(remaining, annual_rate_percent, months)
    savings = {
        method: _money(
            baseline[method]["total_interest"] - after[method]["total_interest"]
        )
        for method in ("equal_payment", "equal_principal")
    }
    return {
        "principal": _money(principal),
        "upfront_prepayment": _money(upfront_prepayment),
        "remaining_principal": _money(remaining),
        "annual_rate_percent": _pct(annual_rate_percent),
        "months": months,
        "baseline": baseline,
        "after_prepayment": after,
        "estimated_interest_savings": savings,
        "currency": "CNY",
        "assumption": "提前还款发生在首期前，期限保持不变。",
        "formula_version": "loan_amortization_v1",
    }


class CashflowNpvIrrInput(StrictFinancialInput):
    cash_flows: list[Decimal] = Field(min_length=2, max_length=600)
    discount_rate_percent: Decimal = Field(gt=-100, le=1000)

    @model_validator(mode="after")
    def require_mixed_signs(self) -> "CashflowNpvIrrInput":
        if not any(value < 0 for value in self.cash_flows):
            raise ValueError("IRR 计算至少需要一个负现金流。")
        if not any(value > 0 for value in self.cash_flows):
            raise ValueError("IRR 计算至少需要一个正现金流。")
        return self


def _npv_float(cash_flows: list[Decimal], rate: float) -> float:
    return sum(float(value) / ((1.0 + rate) ** index) for index, value in enumerate(cash_flows))


def _irr_bisection(cash_flows: list[Decimal]) -> float | None:
    # Scan a broad rate domain first; NPV can have multiple roots for
    # non-conventional cash flows, so this tool reports the first detected root.
    points = [-0.9999]
    points.extend(-0.99 + index * 0.01 for index in range(99))
    points.extend(index / 100 for index in range(0, 101))
    points.extend(10 ** (index / 20) - 1 for index in range(1, 81))
    left = points[0]
    left_value = _npv_float(cash_flows, left)
    for right in points[1:]:
        right_value = _npv_float(cash_flows, right)
        if math.isfinite(left_value) and math.isfinite(right_value):
            if left_value == 0:
                return left
            if left_value * right_value < 0:
                for _ in range(160):
                    middle = (left + right) / 2
                    middle_value = _npv_float(cash_flows, middle)
                    if abs(middle_value) < 1e-8:
                        return middle
                    if left_value * middle_value <= 0:
                        right = middle
                    else:
                        left, left_value = middle, middle_value
                return (left + right) / 2
        left, left_value = right, right_value
    return None


def cashflow_npv_irr(
    *, cash_flows: list[Decimal], discount_rate_percent: Decimal
) -> dict:
    rate = discount_rate_percent / Decimal("100")
    with localcontext() as ctx:
        ctx.prec = 40
        npv = sum(
            value / ((Decimal("1") + rate) ** index)
            for index, value in enumerate(cash_flows)
        )
    irr = _irr_bisection(cash_flows)
    return {
        "npv": _money(npv),
        "irr_percent": None if irr is None else _pct(_decimal(irr * 100)),
        "discount_rate_percent": _pct(discount_rate_percent),
        "period_count": len(cash_flows) - 1,
        "currency": "CNY",
        "irr_note": "返回搜索区间内检测到的第一个 IRR 根；非常规现金流可能存在多个根。",
        "formula_version": "npv_irr_v1",
    }


class BondAnalyticsInput(StrictFinancialInput):
    face_value: Decimal = Field(gt=0, le=10**15)
    annual_coupon_rate_percent: Decimal = Field(ge=0, le=100)
    annual_yield_percent: Decimal = Field(gt=-100, le=1000)
    years_to_maturity: int = Field(ge=1, le=100)
    payments_per_year: int = Field(default=2, ge=1, le=12)


def bond_analytics(
    *,
    face_value: Decimal,
    annual_coupon_rate_percent: Decimal,
    annual_yield_percent: Decimal,
    years_to_maturity: int,
    payments_per_year: int = 2,
) -> dict:
    periods = years_to_maturity * payments_per_year
    coupon = face_value * annual_coupon_rate_percent / Decimal("100") / payments_per_year
    periodic_yield = annual_yield_percent / Decimal("100") / payments_per_year
    if periodic_yield <= Decimal("-1"):
        raise ValueError("每期收益率必须大于 -100%。")
    with localcontext() as ctx:
        ctx.prec = 40
        cashflows = [coupon] * periods
        cashflows[-1] += face_value
        present_values = [
            cashflow / ((Decimal("1") + periodic_yield) ** index)
            for index, cashflow in enumerate(cashflows, start=1)
        ]
        price = sum(present_values)
        macaulay_periods = sum(
            Decimal(index) * value for index, value in enumerate(present_values, start=1)
        ) / price
        macaulay_years = macaulay_periods / payments_per_year
        modified_years = macaulay_years / (Decimal("1") + periodic_yield)
        convexity = sum(
            Decimal(index * (index + 1)) * value
            for index, value in enumerate(present_values, start=1)
        ) / (
            price
            * (Decimal("1") + periodic_yield) ** 2
            * Decimal(payments_per_year**2)
        )
    return {
        "dirty_price": _money(price),
        "macaulay_duration_years": _ratio(macaulay_years),
        "modified_duration_years": _ratio(modified_years),
        "convexity_years_squared": _ratio(convexity),
        "periods": periods,
        "currency": "CNY",
        "assumption": "估值日位于付息日，不含应计利息。",
        "formula_version": "fixed_coupon_bond_v1",
    }


class PortfolioRiskInput(StrictFinancialInput):
    periodic_returns_percent: list[Decimal] = Field(min_length=2, max_length=5000)
    risk_free_rate_percent_per_period: Decimal = Field(default=Decimal("0"), ge=-100, le=100)
    periods_per_year: int = Field(default=12, ge=1, le=366)


def portfolio_risk_metrics(
    *,
    periodic_returns_percent: list[Decimal],
    risk_free_rate_percent_per_period: Decimal = Decimal("0"),
    periods_per_year: int = 12,
) -> dict:
    returns = [value / Decimal("100") for value in periodic_returns_percent]
    count = Decimal(len(returns))
    mean = sum(returns) / count
    variance = sum((value - mean) ** 2 for value in returns) / Decimal(len(returns) - 1)
    volatility = Decimal(str(math.sqrt(float(variance))))
    annualized_return = (Decimal("1") + mean) ** periods_per_year - Decimal("1")
    annualized_volatility = volatility * Decimal(str(math.sqrt(periods_per_year)))
    rf = risk_free_rate_percent_per_period / Decimal("100")
    sharpe = None if volatility == 0 else (mean - rf) / volatility * Decimal(str(math.sqrt(periods_per_year)))

    wealth = Decimal("1")
    peak = Decimal("1")
    max_drawdown = Decimal("0")
    for value in returns:
        wealth *= Decimal("1") + value
        peak = max(peak, wealth)
        drawdown = wealth / peak - Decimal("1")
        max_drawdown = min(max_drawdown, drawdown)
    return {
        "periodic_mean_return_percent": _pct(mean * 100),
        "annualized_return_percent": _pct(annualized_return * 100),
        "annualized_volatility_percent": _pct(annualized_volatility * 100),
        "sharpe_ratio": None if sharpe is None else _ratio(sharpe),
        "cumulative_return_percent": _pct((wealth - 1) * 100),
        "max_drawdown_percent": _pct(max_drawdown * 100),
        "observation_count": len(returns),
        "formula_version": "portfolio_risk_v1",
    }


class AssetRebalanceInput(StrictFinancialInput):
    current_amounts: dict[str, Decimal] = Field(min_length=1, max_length=50)
    target_weights_percent: dict[str, Decimal] = Field(min_length=1, max_length=50)
    tolerance_percent: Decimal = Field(default=Decimal("1"), ge=0, le=100)

    @model_validator(mode="after")
    def validate_allocations(self) -> "AssetRebalanceInput":
        if set(self.current_amounts) != set(self.target_weights_percent):
            raise ValueError("当前资产与目标权重必须包含完全相同的资产名称。")
        if any(value < 0 for value in self.current_amounts.values()):
            raise ValueError("当前资产金额不能为负数。")
        if any(value < 0 or value > 100 for value in self.target_weights_percent.values()):
            raise ValueError("目标权重必须位于 0 到 100 之间。")
        if abs(sum(self.target_weights_percent.values()) - Decimal("100")) > Decimal("0.01"):
            raise ValueError("目标权重合计必须为 100%。")
        if sum(self.current_amounts.values()) <= 0:
            raise ValueError("当前资产总额必须大于 0。")
        return self


def asset_allocation_rebalance(
    *,
    current_amounts: dict[str, Decimal],
    target_weights_percent: dict[str, Decimal],
    tolerance_percent: Decimal = Decimal("1"),
) -> dict:
    total = sum(current_amounts.values())
    trades = []
    for asset in sorted(current_amounts):
        current = current_amounts[asset]
        current_weight = current / total * 100
        target_weight = target_weights_percent[asset]
        target_amount = total * target_weight / 100
        trade_amount = target_amount - current
        drift = current_weight - target_weight
        trades.append(
            {
                "asset": asset,
                "current_amount": _money(current),
                "current_weight_percent": _pct(current_weight),
                "target_weight_percent": _pct(target_weight),
                "target_amount": _money(target_amount),
                "trade": "hold" if abs(drift) <= tolerance_percent else ("buy" if trade_amount > 0 else "sell"),
                "trade_amount": _money(abs(trade_amount)) if abs(drift) > tolerance_percent else Decimal("0.00"),
                "drift_percent": _pct(drift),
            }
        )
    return {
        "portfolio_value": _money(total),
        "tolerance_percent": _pct(tolerance_percent),
        "trades": trades,
        "currency": "CNY",
        "assumption": "忽略税费、交易费和最小交易单位。",
        "formula_version": "allocation_rebalance_v1",
    }


class FinancialRatioInput(StrictFinancialInput):
    revenue: Decimal = Field(gt=0, le=10**18)
    net_income: Decimal = Field(le=10**18, ge=-(10**18))
    total_assets: Decimal = Field(gt=0, le=10**18)
    total_liabilities: Decimal = Field(ge=0, le=10**18)
    total_equity: Decimal = Field(gt=0, le=10**18)
    current_assets: Decimal = Field(ge=0, le=10**18)
    current_liabilities: Decimal = Field(gt=0, le=10**18)
    gross_profit: Decimal | None = Field(default=None, ge=-(10**18), le=10**18)
    average_assets: Decimal | None = Field(default=None, gt=0, le=10**18)
    average_equity: Decimal | None = Field(default=None, gt=0, le=10**18)


def financial_ratio_analysis(
    *,
    revenue: Decimal,
    net_income: Decimal,
    total_assets: Decimal,
    total_liabilities: Decimal,
    total_equity: Decimal,
    current_assets: Decimal,
    current_liabilities: Decimal,
    gross_profit: Decimal | None = None,
    average_assets: Decimal | None = None,
    average_equity: Decimal | None = None,
) -> dict:
    assets_base = average_assets or total_assets
    equity_base = average_equity or total_equity
    net_margin = net_income / revenue
    asset_turnover = revenue / assets_base
    equity_multiplier = assets_base / equity_base
    return {
        "current_ratio": _ratio(current_assets / current_liabilities),
        "debt_to_assets_percent": _pct(total_liabilities / total_assets * 100),
        "net_margin_percent": _pct(net_margin * 100),
        "return_on_assets_percent": _pct(net_income / assets_base * 100),
        "return_on_equity_percent": _pct(net_income / equity_base * 100),
        "gross_margin_percent": (
            None if gross_profit is None else _pct(gross_profit / revenue * 100)
        ),
        "dupont": {
            "net_margin": _ratio(net_margin),
            "asset_turnover": _ratio(asset_turnover),
            "equity_multiplier": _ratio(equity_multiplier),
            "roe_product_percent": _pct(
                net_margin * asset_turnover * equity_multiplier * 100
            ),
        },
        "formula_version": "financial_ratios_dupont_v1",
    }
