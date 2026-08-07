from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """
    Agent 运行硬预算。

    这些限制由 Python 运行时控制，
    不能由大模型自行修改。
    """

    max_agent_rounds: int = 6
    max_total_tool_calls: int = 12
    max_parallel_tool_calls: int = 4

    max_same_error_count: int = 2
    max_consecutive_no_progress_rounds: int = 2
    max_plan_revisions: int = 2
    max_output_rewrites: int = 1

    default_tool_timeout_seconds: float = 10.0
    total_run_timeout_seconds: float = 120.0

    max_context_messages: int = 20
    max_tool_result_chars: int = 12000

    def __post_init__(self) -> None:
        integer_fields = {
            "max_agent_rounds": self.max_agent_rounds,
            "max_total_tool_calls": self.max_total_tool_calls,
            "max_parallel_tool_calls": self.max_parallel_tool_calls,
            "max_same_error_count": self.max_same_error_count,
            "max_consecutive_no_progress_rounds": (
                self.max_consecutive_no_progress_rounds
            ),
            "max_plan_revisions": self.max_plan_revisions,
            "max_output_rewrites": self.max_output_rewrites,
            "max_context_messages": self.max_context_messages,
            "max_tool_result_chars": self.max_tool_result_chars,
        }

        for field_name, field_value in integer_fields.items():
            if field_value <= 0:
                raise ValueError(
                    f"{field_name} 必须大于 0，当前值为 {field_value}。"
                )

        if self.default_tool_timeout_seconds <= 0:
            raise ValueError(
                "default_tool_timeout_seconds 必须大于 0。"
            )

        if self.total_run_timeout_seconds <= 0:
            raise ValueError(
                "total_run_timeout_seconds 必须大于 0。"
            )

        if (
            self.default_tool_timeout_seconds
            > self.total_run_timeout_seconds
        ):
            raise ValueError(
                "单个工具超时不能大于整个 Agent 运行超时。"
            )


DEFAULT_AGENT_LIMITS = AgentLimits()
