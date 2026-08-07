from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


YEARLY_EXPENSE_TO_MONTHLY_TOOL = "yearly_expense_to_monthly"
EMERGENCY_FUND_RANGE_TOOL = "emergency_fund_range"
LIFE_INSURANCE_GAP_TOOL = "life_insurance_gap"


@dataclass(frozen=True)
class CalculationStep:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class CalculationPlan:
    supported: bool
    steps: list[CalculationStep]
    missing_fields: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "steps": [
                {
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                }
                for step in self.steps
            ],
            "missing_fields": list(self.missing_fields),
            "reason": self.reason,
        }


def _normalize_text(text: str) -> str:
    return (
        text.replace(",", "")
        .replace("，", "")
        .replace("。", "")
        .replace("；", "")
        .replace("：", "")
        .replace(" ", "")
        .strip()
    )


def _amount_to_yuan(
    number_text: str,
    unit_text: str | None,
) -> int:
    try:
        value = Decimal(number_text)
    except InvalidOperation as exc:
        raise ValueError(
            f"金额数字无法解析：{number_text}"
        ) from exc

    unit = unit_text or ""

    if unit in {"万", "万元"}:
        value = value * Decimal("10000")

    return int(value)

def _looks_like_month_count(
    *,
    normalized_text: str,
    number_start: int,
    number_end: int,
    unit_text: str | None,
) -> bool:
    """
    判断一个数字是不是“3到6个月”这种月份数字。

    目的：
    避免把“月度必要支出，并计算3到6个月备用金”里的 3
    错识别成 monthly_necessary_expense。
    """
    if unit_text:
        return False

    following_text = normalized_text[
        number_end:number_end + 12
    ]

    previous_text = normalized_text[
        max(0, number_start - 4):number_start
    ]

    if following_text.startswith("个月"):
        return True

    if re.match(
        r"^(到|至|-|~)\d+个月",
        following_text,
    ):
        return True

    if previous_text.endswith(
        ("到", "至", "-", "~")
    ) and following_text.startswith("个月"):
        return True

    return False


def _extract_amount_after_keywords(
    text: str,
    keywords: list[str],
) -> int | None:
    """
    从关键词附近提取金额。

    例：
    年度必要支出是18万元
    家庭年度必要支出180000元
    月度必要支出15000元

    注意：
    这里要避免把“3到6个月”这种月份数字误识别成金额。
    """
    normalized_text = _normalize_text(text)

    amount_pattern = (
        r"(?P<number>\d+(?:\.\d+)?)"
        r"(?P<unit>万元|万|元)?"
    )

    for keyword in keywords:
        patterns = [
            rf"{keyword}[^0-9]{{0,12}}{amount_pattern}",
            rf"{amount_pattern}[^0-9]{{0,12}}{keyword}",
        ]

        for pattern in patterns:
            for match in re.finditer(
                pattern,
                normalized_text,
            ):
                number = match.group("number")
                unit = match.group("unit")

                number_start = match.start("number")

                if unit:
                    number_end = match.end("unit")
                else:
                    number_end = match.end("number")

                if _looks_like_month_count(
                    normalized_text=normalized_text,
                    number_start=number_start,
                    number_end=number_end,
                    unit_text=unit,
                ):
                    continue

                return _amount_to_yuan(
                    number_text=number,
                    unit_text=unit,
                )

    return None


def _extract_yearly_necessary_expense(
    text: str,
) -> int | None:
    return _extract_amount_after_keywords(
        text,
        keywords=[
            "年度必要支出",
            "年必要支出",
            "每年必要支出",
            "年度生活支出",
            "年生活支出",
            "每年生活支出",
            "年度支出",
            "年支出",
        ],
    )


def _extract_monthly_necessary_expense(
    text: str,
) -> int | None:
    return _extract_amount_after_keywords(
        text,
        keywords=[
            "月度必要支出",
            "月必要支出",
            "每月必要支出",
            "月度生活支出",
            "月生活支出",
            "每月生活支出",
            "月度支出",
            "月支出",
        ],
    )


def _extract_emergency_fund_month_range(
    text: str,
) -> tuple[int, int]:
    normalized_text = _normalize_text(text)

    range_patterns = [
        r"(?P<min>\d+)到(?P<max>\d+)个月",
        r"(?P<min>\d+)至(?P<max>\d+)个月",
        r"(?P<min>\d+)-(?P<max>\d+)个月",
        r"(?P<min>\d+)~(?P<max>\d+)个月",
    ]

    for pattern in range_patterns:
        match = re.search(
            pattern,
            normalized_text,
        )

        if match:
            min_months = int(match.group("min"))
            max_months = int(match.group("max"))

            if min_months <= 0 or max_months <= 0:
                continue

            if min_months > max_months:
                min_months, max_months = (
                    max_months,
                    min_months,
                )

            return min_months, max_months

    return 3, 6


def _wants_emergency_fund_range(
    text: str,
) -> bool:
    return any(
        keyword in text
        for keyword in [
            "紧急备用金",
            "备用金",
            "应急金",
            "应急备用金",
        ]
    )


def build_calculation_plan(
    user_message: str,
) -> CalculationPlan:
    """
    根据用户问题生成金融计算执行计划。

    当前 Stage 4.2.1 先支持：
    1. 年度必要支出 -> 月度必要支出
    2. 月度必要支出 -> 3到6个月紧急备用金范围
    3. 年度必要支出 -> 月度必要支出 -> 紧急备用金范围
    """
    message = user_message.strip()

    if not message:
        return CalculationPlan(
            supported=False,
            steps=[],
            missing_fields=["user_message"],
            reason="用户问题为空，无法生成计算计划。",
        )

    yearly_expense = _extract_yearly_necessary_expense(
        message
    )

    monthly_expense = _extract_monthly_necessary_expense(
        message
    )

    wants_emergency_fund = _wants_emergency_fund_range(
        message
    )

    min_months, max_months = (
        _extract_emergency_fund_month_range(message)
    )

    steps: list[CalculationStep] = []

    if yearly_expense is not None:
        steps.append(
            CalculationStep(
                tool_name=YEARLY_EXPENSE_TO_MONTHLY_TOOL,
                arguments={
                    "yearly_necessary_expense": yearly_expense,
                },
            )
        )

    if wants_emergency_fund:
        if monthly_expense is not None:
            steps.append(
                CalculationStep(
                    tool_name=EMERGENCY_FUND_RANGE_TOOL,
                    arguments={
                        "monthly_necessary_expense": monthly_expense,
                        "min_months": min_months,
                        "max_months": max_months,
                    },
                )
            )

        elif yearly_expense is not None:
            steps.append(
                CalculationStep(
                    tool_name=EMERGENCY_FUND_RANGE_TOOL,
                    arguments={
                        "monthly_necessary_expense": {
                            "from_step": YEARLY_EXPENSE_TO_MONTHLY_TOOL,
                            "field": "monthly_necessary_expense",
                        },
                        "min_months": min_months,
                        "max_months": max_months,
                    },
                )
            )

        else:
            return CalculationPlan(
                supported=False,
                steps=steps,
                missing_fields=[
                    "monthly_necessary_expense",
                ],
                reason=(
                    "用户想计算紧急备用金，"
                    "但没有提供月度必要支出或年度必要支出。"
                ),
            )

    if not steps:
        return CalculationPlan(
            supported=False,
            steps=[],
            missing_fields=[],
            reason="当前问题没有命中已支持的确定性金融计算场景。",
        )

    return CalculationPlan(
        supported=True,
        steps=steps,
        missing_fields=[],
        reason="已生成确定性金融计算计划。",
    )
