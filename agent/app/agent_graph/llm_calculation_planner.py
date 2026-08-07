from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any

from app.agent_graph.calculation_planner import (
    CalculationPlan,
    CalculationStep,
    EMERGENCY_FUND_RANGE_TOOL,
    LIFE_INSURANCE_GAP_TOOL,
    YEARLY_EXPENSE_TO_MONTHLY_TOOL,
)


ALLOWED_CALCULATION_TOOLS = {
    YEARLY_EXPENSE_TO_MONTHLY_TOOL,
    EMERGENCY_FUND_RANGE_TOOL,
    LIFE_INSURANCE_GAP_TOOL,
}


LLM_CALCULATION_PLANNER_SYSTEM_PROMPT = """
你是一个金融计算计划解析器。

你的任务不是回答用户问题，也不是自己计算结果。
你的任务是把用户问题解析成固定 JSON 格式的金融计算工具执行计划。

你只能输出 JSON，不能输出 Markdown，不能输出解释文字。

允许使用的工具只有：

1. yearly_expense_to_monthly
用途：把年度必要支出换算成月度必要支出。
参数：
- yearly_necessary_expense: number，单位是元。

2. emergency_fund_range
用途：根据月度必要支出，计算紧急备用金区间。
参数：
- monthly_necessary_expense: number 或引用上一步结果。
- min_months: number，默认 3。
- max_months: number，默认 6。

3. life_insurance_gap
用途：计算寿险保障缺口。
参数：
- family_required_funds: number，单位是元。
- available_assets: number，单位是元。
- existing_life_insurance: number，单位是元。
- other_available_funds: number，单位是元，缺失时可用 0。

重要规则：

1. 所有金额必须统一换算成“元”。
例如：
18万元 -> 180000
80万 -> 800000
1.5万元 -> 15000

2. “3到6个月”“3-6个月”“3至6个月”是月份范围，不是金额。

3. 如果用户只提供年度必要支出，并要求计算紧急备用金：
先调用 yearly_expense_to_monthly；
再调用 emergency_fund_range；
第二步的 monthly_necessary_expense 必须引用第一步结果。

4. 不要自己计算 180000 / 12，也不要自己计算备用金金额。
计算必须交给工具。

5. 如果信息不足，supported 必须是 false，并在 missing_fields 里列出缺失字段。

6. 如果问题不是金融计算问题，supported 必须是 false。

输出 JSON 格式必须是：

{
  "supported": true,
  "steps": [
    {
      "tool_name": "yearly_expense_to_monthly",
      "arguments": {
        "yearly_necessary_expense": 180000
      }
    }
  ],
  "missing_fields": [],
  "reason": "简短说明为什么这样规划"
}

如果需要引用上一步结果，格式必须是：

{
  "from_step": "yearly_expense_to_monthly",
  "field": "monthly_necessary_expense"
}
""".strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    从模型输出中提取 JSON object。

    兼容两种情况：
    1. 纯 JSON
    2. ```json ... ``` 包裹的 JSON
    """
    content = text.strip()

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        content,
        flags=re.DOTALL,
    )

    if fence_match:
        content = fence_match.group(1).strip()

    try:
        data = json.loads(content)
    except JSONDecodeError:
        object_match = re.search(
            r"\{.*\}",
            content,
            flags=re.DOTALL,
        )

        if not object_match:
            raise

        data = json.loads(object_match.group(0))

    if not isinstance(data, dict):
        raise ValueError("LLM calculation planner 输出不是 JSON object。")

    return data


def _normalize_missing_fields(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def _normalize_arguments(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    return value


def _plan_from_dict(
    data: dict[str, Any],
) -> CalculationPlan:
    supported = bool(data.get("supported", False))
    raw_steps = data.get("steps") or []

    if not isinstance(raw_steps, list):
        raw_steps = []

    steps: list[CalculationStep] = []

    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue

        tool_name = str(raw_step.get("tool_name") or "").strip()

        if not tool_name:
            continue

        arguments = _normalize_arguments(
            raw_step.get("arguments")
        )

        steps.append(
            CalculationStep(
                tool_name=tool_name,
                arguments=arguments,
            )
        )

    missing_fields = _normalize_missing_fields(
        data.get("missing_fields")
    )

    reason = str(data.get("reason") or "").strip()

    if not reason:
        reason = "LLM 已生成金融计算计划。"

    return CalculationPlan(
        supported=supported,
        steps=steps,
        missing_fields=missing_fields,
        reason=reason,
    )


def validate_basic_llm_calculation_plan(
    plan: CalculationPlan,
) -> CalculationPlan:
    """
    LLM 计划的基础校验。

    注意：
    这里只做轻量校验。
    更严格的参数校验会放到 Stage 4.2.3 validator。
    """
    if not plan.supported:
        return plan

    if not plan.steps:
        return CalculationPlan(
            supported=False,
            steps=[],
            missing_fields=plan.missing_fields,
            reason="LLM 返回 supported=true，但 steps 为空。",
        )

    unknown_tools = [
        step.tool_name
        for step in plan.steps
        if step.tool_name not in ALLOWED_CALCULATION_TOOLS
    ]

    if unknown_tools:
        return CalculationPlan(
            supported=False,
            steps=[],
            missing_fields=[],
            reason=(
                "LLM 返回了不允许的计算工具："
                f"{unknown_tools}"
            ),
        )

    return plan


class LLMCalculationPlanner:
    """
    基于 DeepSeek 的金融计算结构化计划器。

    它只负责理解问题并生成工具执行计划。
    不负责执行工具，也不负责最终回答。
    """

    def __init__(
        self,
        llm_client: Any,
    ) -> None:
        self.llm_client = llm_client

    async def build_plan(
        self,
        user_message: str,
    ) -> CalculationPlan:
        message = user_message.strip()

        if not message:
            return CalculationPlan(
                supported=False,
                steps=[],
                missing_fields=["user_message"],
                reason="用户问题为空，无法生成 LLM 计算计划。",
            )

        try:
            result = await self.llm_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": LLM_CALCULATION_PLANNER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": message,
                    },
                ],
                thinking_enabled=False,
                max_completion_tokens=1024,
            )

            assistant_message = result.get("message") or {}
            content = str(
                assistant_message.get("content") or ""
            ).strip()

            data = _extract_json_object(
                content
            )

            plan = _plan_from_dict(
                data
            )

            return validate_basic_llm_calculation_plan(
                plan
            )

        except Exception as exc:
            return CalculationPlan(
                supported=False,
                steps=[],
                missing_fields=[],
                reason=(
                    "LLM calculation planner 执行失败："
                    f"{type(exc).__name__}: {exc}"
                ),
            )
