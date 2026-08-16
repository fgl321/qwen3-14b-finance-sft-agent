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

    step_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$",
    )

    depends_on: list[str] = Field(default_factory=list, max_length=8)

    @property
    def effective_step_id(self) -> str:
        return self.step_id or self.tool_call_id


def iter_typed_references(value: Any):
    """Yield strict references shaped as {"$ref": {"step_id": ..., "path": [...]}}."""
    if isinstance(value, dict):
        if set(value) == {"$ref"} and isinstance(value["$ref"], dict):
            payload = value["$ref"]
            if set(payload).issubset({"step_id", "path"}) and payload.get("step_id"):
                path = payload.get("path") or []
                if not isinstance(path, list) or not all(isinstance(item, (str, int)) for item in path):
                    raise ValueError("typed reference path must be a string/integer list")
                yield str(payload["step_id"]), list(path)
                return
        for child in value.values():
            yield from iter_typed_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_typed_references(child)


def resolve_typed_references(value: Any, outputs: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        references = list(iter_typed_references(value))
        if set(value) == {"$ref"} and references:
            step_id, path = references[0]
            if step_id not in outputs:
                raise KeyError(f"dependency output is unavailable: {step_id}")
            resolved = outputs[step_id]
            for segment in path:
                if isinstance(resolved, dict) and segment in resolved:
                    resolved = resolved[segment]
                elif isinstance(resolved, list) and isinstance(segment, int):
                    resolved = resolved[segment]
                else:
                    raise KeyError(f"invalid dependency output path: {step_id}.{path}")
            return resolved
        return {key: resolve_typed_references(child, outputs) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_typed_references(child, outputs) for child in value]
    return value


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

        step_ids = [call.effective_step_id for call in self.tool_calls]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("tool call step_id values must be unique")
        known_steps = set(step_ids)
        dependencies = {
            call.effective_step_id: set(call.depends_on)
            for call in self.tool_calls
        }
        for call in self.tool_calls:
            step_id = call.effective_step_id
            if step_id in dependencies[step_id]:
                raise ValueError(f"step cannot depend on itself: {step_id}")
            unknown = dependencies[step_id] - known_steps
            if unknown:
                raise ValueError(f"unknown dependency for {step_id}: {sorted(unknown)}")
            referenced = {ref_step for ref_step, _ in iter_typed_references(call.arguments)}
            if not referenced.issubset(dependencies[step_id]):
                raise ValueError(
                    f"typed references must be declared in depends_on for {step_id}: "
                    f"{sorted(referenced - dependencies[step_id])}"
                )
        visiting: set[str] = set()
        visited: set[str] = set()
        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("tool dependency graph contains a cycle")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in dependencies.get(step_id, set()):
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)
        for step_id in step_ids:
            visit(step_id)
        return self
