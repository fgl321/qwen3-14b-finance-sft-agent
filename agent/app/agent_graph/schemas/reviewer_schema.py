from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ReviewVerdict = Literal[
    "approve",
    "revise",
    "clarify",
    "reject",
]


class ReviewDecision(BaseModel):
    """
    高复杂度或高风险计划的复核结果。

    Reviewer 不执行工具，不生成最终回答，
    只决定当前计划能否继续。
    """

    model_config = ConfigDict(extra="forbid")

    verdict: ReviewVerdict

    feedback: str = Field(
        default="",
        max_length=1500,
    )

    @model_validator(mode="after")
    def validate_feedback(self) -> "ReviewDecision":
        if self.verdict in {"revise", "clarify", "reject"}:
            if not self.feedback.strip():
                raise ValueError(
                    f"verdict={self.verdict} 时必须给出 feedback。"
                )

        return self
