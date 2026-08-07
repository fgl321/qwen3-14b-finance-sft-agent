from __future__ import annotations

from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


PlannerAction = Literal[
    "call_tools",
    "respond",
    "clarify",
    "fallback",
]

PlannerConfidence = Literal[
    "low",
    "medium",
    "high",
]



ExecutionPolicy = Literal[
    "direct_allowed",
    "auto",
    "prefer_tool",
    "require_tool",
]

DEFAULT_EXECUTION_POLICY: ExecutionPolicy = "auto"

VALID_EXECUTION_POLICIES = frozenset(
    {
        "direct_allowed",
        "auto",
        "prefer_tool",
        "require_tool",
    }
)


def normalize_execution_policy(
    value: str | None,
) -> ExecutionPolicy:
    """
    将 API、Service 或 Checkpoint 中的执行策略统一归一化。

    不认识的值直接拒绝，避免静默降级为另一种策略。
    """

    cleaned = str(
        value or DEFAULT_EXECUTION_POLICY
    ).strip().lower()

    if cleaned not in VALID_EXECUTION_POLICIES:
        allowed = ", ".join(
            sorted(VALID_EXECUTION_POLICIES)
        )
        raise ValueError(
            "execution_policy 不合法："
            f"{cleaned!r}。允许值：{allowed}。"
        )

    return cast(ExecutionPolicy, cleaned)


def _new_tool_call_id() -> str:
    return f"call_{uuid4().hex[:16]}"


class ToolCallRequest(BaseModel):
    """
    大模型计划器生成的一次工具调用请求。

    这里不判断参数的金融语义是否正确。
    参数是否符合工具输入结构，由 Tool Executor 在运行时处理。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    tool_call_id: str = Field(
        default_factory=_new_tool_call_id,
        min_length=1,
        max_length=128,
    )

    tool_name: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$",
    )

    arguments: dict[str, Any] = Field(default_factory=dict)


class PlannerDecision(BaseModel):
    """
    LLM Task Planner 每一轮只决定下一步行动。

    不生成完整 DAG；
    不使用 from_step；
    不预先规划所有未来步骤。
    """

    model_config = ConfigDict(extra="forbid")

    action: PlannerAction

    tool_calls: list[ToolCallRequest] = Field(
        default_factory=list,
        max_length=4,
    )

    clarification_question: str | None = Field(
        default=None,
        max_length=500,
    )

    decision_reason: str = Field(
        default="",
        max_length=1000,
    )

    confidence: PlannerConfidence = "medium"

    needs_review: bool = False

    plan_version: int = Field(
        default=1,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_action_payload(self) -> "PlannerDecision":
        if self.action == "call_tools":
            if not self.tool_calls:
                raise ValueError(
                    "action=call_tools 时必须至少包含一个 tool_call。"
                )

            if self.clarification_question is not None:
                raise ValueError(
                    "action=call_tools 时不能同时包含 clarification_question。"
                )

        else:
            if self.tool_calls:
                raise ValueError(
                    f"action={self.action} 时不能携带 tool_calls。"
                )

        if self.action == "clarify":
            if not self.clarification_question:
                raise ValueError(
                    "action=clarify 时必须提供 clarification_question。"
                )
        elif self.clarification_question is not None:
            raise ValueError(
                "只有 action=clarify 时才能提供 clarification_question。"
            )

        return self
