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
            ),
            ToolSpec(
                name="life_insurance_gap",
                description=(
                    "根据家庭必要支出、保障年数、债务、教育责任、"
                    "可用资产和已有寿险保额，计算寿险保障缺口。"
                    "缺少完成当前计算所必需的信息时，不应编造参数，"
                    "应先向用户追问。"
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
            ),
        ]
    )

    registry.freeze()

    return registry
