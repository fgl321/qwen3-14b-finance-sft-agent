from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


ToolRiskLevel = Literal[
    "low",
    "medium",
    "high",
]
ToolSourceClass = Literal[
    "pure_math",
    "user_fact_transform",
    "domain_heuristic",
    "external_data",
]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """
    一个工具在生产 Agent 中的完整运行定义。

    handler:
        真正执行工具的 Python 函数。

    input_model:
        工具自己的输入协议。
        这只检查运行参数，不重新理解用户自然语言。

    side_effect:
        是否会修改数据库、发送消息或产生其他副作用。

    idempotent:
        相同参数重复调用是否产生相同结果。
        只有幂等工具才允许基础设施自动重试。

    parallel_safe:
        是否允许与其他工具并行执行。
    """

    name: str
    description: str

    input_model: type[BaseModel]
    handler: Callable[..., Any]

    tool_group: str = "general"

    timeout_seconds: float = 10.0

    max_infrastructure_retries: int = 1

    risk_level: ToolRiskLevel = "low"

    side_effect: bool = False
    idempotent: bool = True
    parallel_safe: bool = True

    source_class: ToolSourceClass = "user_fact_transform"

    allowed_roles: frozenset[str] = frozenset({"user", "system"})

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("工具名称不能为空。")

        if not self.description:
            raise ValueError(
                f"工具 {self.name} 的 description 不能为空。"
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                f"工具 {self.name} 的 timeout_seconds 必须大于 0。"
            )

        if self.max_infrastructure_retries < 0:
            raise ValueError(
                "max_infrastructure_retries 不能小于 0。"
            )

        if not issubclass(self.input_model, BaseModel):
            raise TypeError(
                f"工具 {self.name} 的 input_model 必须继承 BaseModel。"
            )

        if not callable(self.handler):
            raise TypeError(
                f"工具 {self.name} 的 handler 必须可调用。"
            )

    def to_llm_tool_definition(self) -> dict[str, Any]:
        """
        转换成 OpenAI/DeepSeek 兼容的工具说明结构。

        后续 Planner 可以直接获得工具名、说明和 JSON Schema。
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
                "source_class": self.source_class,
            },
        }


def tool_allowed(
    source_class: str,
    source_contract: Any,
) -> bool:
    """Deterministic source gate: unknown classes fail closed."""

    if source_class == "pure_math":
        return (
            getattr(
                source_contract,
                "deterministic_derivation",
                "forbidden",
            )
            == "allowed"
        )
    if source_class == "user_fact_transform":
        return (
            getattr(
                source_contract,
                "current_user_facts",
                "forbidden",
            )
            == "allowed"
            and getattr(
                source_contract,
                "deterministic_derivation",
                "forbidden",
            )
            == "allowed"
        )
    if source_class == "domain_heuristic":
        return (
            getattr(
                source_contract,
                "domain_heuristics",
                "forbidden",
            )
            == "allowed"
        )
    if source_class == "external_data":
        return (
            getattr(source_contract, "web", "forbidden") == "allowed"
        )
    return False
