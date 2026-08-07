from __future__ import annotations

from typing import Any

from app.tools.finance_calculator import (
    emergency_fund_range,
    life_insurance_gap,
    yearly_expense_to_monthly,
)


RAG_TOOL_NAME = "search_knowledge_base"


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "yearly_expense_to_monthly",
            "description": (
                "将家庭年度必要支出换算为月度必要支出。"
                "当用户提供的是一年必要支出、年度支出、年支出时，应先调用该工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "yearly_necessary_expense": {
                        "type": "number",
                        "description": "家庭年度必要支出，单位为元。例如 180000。",
                    }
                },
                "required": ["yearly_necessary_expense"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emergency_fund_range",
            "description": (
                "根据家庭月度必要支出计算紧急备用金建议区间。"
                "通常按 3 到 6 个月必要支出计算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_necessary_expense": {
                        "type": "number",
                        "description": "家庭月度必要支出，单位为元。例如 15000。",
                    },
                    "min_months": {
                        "type": "number",
                        "description": "最低覆盖月数，默认 3。",
                        "default": 3,
                    },
                    "max_months": {
                        "type": "number",
                        "description": "最高覆盖月数，默认 6。",
                        "default": 6,
                    },
                },
                "required": ["monthly_necessary_expense"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "life_insurance_gap",
            "description": (
                "计算家庭寿险保障缺口。"
                "适用于用户提供家庭负债、子女教育责任、未来必要支出、已有资产、已有寿险保额等信息时。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "family_debt": {
                        "type": "number",
                        "description": "家庭负债总额，单位元。",
                    },
                    "education_responsibility": {
                        "type": "number",
                        "description": "子女教育责任或其他家庭责任金额，单位元。",
                    },
                    "future_living_expense": {
                        "type": "number",
                        "description": "家庭未来必要生活支出，单位元。",
                    },
                    "available_assets": {
                        "type": "number",
                        "description": "已有可用资产，单位元。",
                    },
                    "existing_life_insurance": {
                        "type": "number",
                        "description": "已有寿险保额，单位元。",
                    },
                },
                "required": [
                    "family_debt",
                    "education_responsibility",
                    "future_living_expense",
                    "available_assets",
                    "existing_life_insurance",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": RAG_TOOL_NAME,
            "description": (
                "从金融知识库中检索资料并生成带引用的回答。"
                "当用户询问金融概念、规则、公式、知识库中的说明、项目文档内容时，应调用该工具。"
                "注意：该工具只负责知识库问答，不负责金额计算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要检索知识库的问题。应尽量保留用户原始问题。",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def is_rag_tool(tool_name: str) -> bool:
    return tool_name == RAG_TOOL_NAME


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    执行本地确定性工具。

    注意：
    search_knowledge_base 是异步 RAG 工具，不在这里执行。
    它由 FinanceAgent 单独调用 RagAnswerService。
    """

    if tool_name == "yearly_expense_to_monthly":
        return yearly_expense_to_monthly(**arguments)

    if tool_name == "emergency_fund_range":
        return emergency_fund_range(**arguments)

    if tool_name == "life_insurance_gap":
        return life_insurance_gap(**arguments)

    if tool_name == RAG_TOOL_NAME:
        raise ValueError(
            "search_knowledge_base 是异步 RAG 工具，不能通过 execute_tool 执行。"
        )

    raise ValueError(f"未知工具：{tool_name}")
