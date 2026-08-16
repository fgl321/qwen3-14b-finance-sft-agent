from __future__ import annotations

from threading import RLock
from typing import Iterable

from app.tools.production_finance_tools import (
    EmergencyFundRangeInput,
    LifeInsuranceGapInput,
    YearlyExpenseToMonthlyInput,
    emergency_fund_range,
    life_insurance_gap,
    yearly_expense_to_monthly,
)
from app.tools.financial_analytics_tools import (
    AssetRebalanceInput,
    BondAnalyticsInput,
    CashflowNpvIrrInput,
    CompoundInterestInput,
    FinancialRatioInput,
    LoanAmortizationInput,
    PortfolioRiskInput,
    asset_allocation_rebalance,
    bond_analytics,
    cashflow_npv_irr,
    compound_interest_projection,
    financial_ratio_analysis,
    loan_amortization_compare,
    portfolio_risk_metrics,
)
from app.tools.tool_specs import ToolSpec


class ToolRegistry:
    """
    显式工具白名单。

    禁止通过 globals、getattr 或 eval 动态寻找并执行函数。
    所有可被 Agent 调用的工具必须在启动阶段显式注册。
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._lock = RLock()
        self._frozen = False

    def register(self, spec: ToolSpec) -> None:
        with self._lock:
            if self._frozen:
                raise RuntimeError(
                    "工具注册表已冻结，不能继续注册工具。"
                )

            if spec.name in self._tools:
                raise ValueError(
                    f"工具名称重复注册：{spec.name}"
                )

            self._tools[spec.name] = spec

    def register_many(
        self,
        specs: Iterable[ToolSpec],
    ) -> None:
        for spec in specs:
            self.register(spec)

    def freeze(self) -> None:
        """
        应用启动完成后冻结注册表。

        防止运行过程中被意外添加或替换工具。
        """

        with self._lock:
            self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def get(self, tool_name: str) -> ToolSpec | None:
        return self._tools.get(tool_name)

    def require(self, tool_name: str) -> ToolSpec:
        spec = self.get(tool_name)

        if spec is None:
            raise KeyError(f"未注册工具：{tool_name}")

        return spec

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def list_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            self._tools[name]
            for name in sorted(self._tools)
        )

    def get_llm_tool_definitions(
        self,
        *,
        allowed_tool_names: set[str] | frozenset[str] | None = None,
        allowed_tool_groups: set[str] | frozenset[str] | None = None,
    ) -> list[dict]:
        """
        只向 Planner 暴露当前路由允许使用的工具。

        Router 可以通过 allowed_tool_groups 缩小工具范围。
        """

        # API/Graph State 使用 list 表达可选过滤条件。
        # 未传字段时，边界层通常会把它序列化成空列表。
        # 因此这里统一约定：
        #
        # - None 或空集合：该维度不参与过滤；
        # - 非空 names：只允许这些工具名称；
        # - 非空 groups：只允许这些工具组；
        # - names 与 groups 都非空：取二者交集。
        #
        # 这样 allowed_tool_names=[] 不会错误覆盖一个有效的
        # allowed_tool_groups=["financial_calculation"]。
        active_tool_names = (
            frozenset(allowed_tool_names)
            if allowed_tool_names
            else None
        )
        active_tool_groups = (
            frozenset(allowed_tool_groups)
            if allowed_tool_groups
            else None
        )

        definitions: list[dict] = []

        for spec in self.list_specs():
            if (
                active_tool_names is not None
                and spec.name not in active_tool_names
            ):
                continue

            if (
                active_tool_groups is not None
                and spec.tool_group not in active_tool_groups
            ):
                continue

            definitions.append(
                spec.to_llm_tool_definition()
            )

        return definitions


def build_production_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register_many(
        [
            ToolSpec(
                name="yearly_expense_to_monthly",
                description=(
                    "将家庭年度必要支出换算为月度必要支出。"
                    "用户提供年度必要支出且需要月度金额时调用。"
                    "不要自行完成除法，应使用本工具。"
                ),
                input_model=YearlyExpenseToMonthlyInput,
                handler=yearly_expense_to_monthly,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                max_infrastructure_retries=0,
                risk_level="low",
                side_effect=False,
                idempotent=True,
                parallel_safe=True,
                source_class="pure_math",
            ),
            ToolSpec(
                name="emergency_fund_range",
                description=(
                    "根据月度必要支出和覆盖月份，计算紧急备用金"
                    "的最小金额和最大金额。"
                    "默认覆盖3到6个月，也可使用用户明确提供的月份。"
                ),
                input_model=EmergencyFundRangeInput,
                handler=emergency_fund_range,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                max_infrastructure_retries=0,
                risk_level="low",
                side_effect=False,
                idempotent=True,
                parallel_safe=True,
                source_class="domain_heuristic",
            ),
            ToolSpec(
                name="life_insurance_gap",
                description=(
                    "根据家庭必要支出、保障年数、债务、教育责任、"
                    "可用资产和已有寿险保额，计算寿险保障缺口。"
                    "用户未指定保障年数时不要追问或默认10年，"
                    "应省略 coverage_years 以返回5/10/15年情景。"
                ),
                input_model=LifeInsuranceGapInput,
                handler=life_insurance_gap,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                max_infrastructure_retries=0,
                risk_level="medium",
                side_effect=False,
                idempotent=True,
                parallel_safe=True,
                source_class="user_fact_transform",
            ),
            ToolSpec(
                name="compound_interest_projection",
                description="计算复利、定投和目标储蓄的未来价值、累计投入与收益。",
                input_model=CompoundInterestInput,
                handler=compound_interest_projection,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                source_class="pure_math",
            ),
            ToolSpec(
                name="loan_amortization_compare",
                description="比较等额本息、等额本金以及期初提前还款后的月供和利息。",
                input_model=LoanAmortizationInput,
                handler=loan_amortization_compare,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                source_class="pure_math",
            ),
            ToolSpec(
                name="cashflow_npv_irr",
                description="根据逐期现金流计算净现值 NPV 和内部收益率 IRR。",
                input_model=CashflowNpvIrrInput,
                handler=cashflow_npv_irr,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                source_class="pure_math",
            ),
            ToolSpec(
                name="bond_analytics",
                description="计算固定利率债券价格、麦考利久期、修正久期和凸性。",
                input_model=BondAnalyticsInput,
                handler=bond_analytics,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                source_class="pure_math",
            ),
            ToolSpec(
                name="portfolio_risk_metrics",
                description="根据周期收益序列计算年化收益、波动率、夏普比率和最大回撤。",
                input_model=PortfolioRiskInput,
                handler=portfolio_risk_metrics,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                source_class="pure_math",
            ),
            ToolSpec(
                name="asset_allocation_rebalance",
                description="比较当前资产配置与目标权重并生成确定性的买入、卖出或保持金额。",
                input_model=AssetRebalanceInput,
                handler=asset_allocation_rebalance,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                risk_level="medium",
                source_class="pure_math",
            ),
            ToolSpec(
                name="financial_ratio_analysis",
                description="计算偿债、盈利、资产效率指标以及三因素杜邦分析。",
                input_model=FinancialRatioInput,
                handler=financial_ratio_analysis,
                tool_group="financial_calculation",
                timeout_seconds=3.0,
                source_class="pure_math",
            ),
        ]
    )

    registry.freeze()

    return registry
