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

    issues: list[str] = Field(
        default_factory=list,
        max_length=12,
    )

    repair_instructions: list[str] = Field(
        default_factory=list,
        max_length=12,
    )

    clarification_question: str | None = Field(
        default=None,
        max_length=500,
    )

    # Deterministic internal summary. It is not part of the LLM response
    # schema, so free-form prose cannot contradict verdict.
    feedback: str = Field(
        default="",
        max_length=1500,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_feedback(cls, value):
        """Accept the v3.1 feedback-only wire form, then enforce one schema.

        The normalized object never retains an independent prose verdict: the
        same feedback becomes the issue/repair or clarification field selected
        by the structured verdict.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)
        verdict = data.get("verdict")
        feedback = str(data.get("feedback") or "").strip()
        if verdict == "revise" and feedback:
            data.setdefault("issues", [feedback])
            data.setdefault("repair_instructions", [feedback])
        elif verdict == "clarify" and feedback:
            data.setdefault("clarification_question", feedback)
        elif verdict == "reject" and feedback:
            data.setdefault("issues", [feedback])
        return data

    @model_validator(mode="after")
    def validate_feedback(self) -> "ReviewDecision":
        if self.verdict == "approve":
            if self.issues or self.repair_instructions or self.clarification_question:
                raise ValueError("verdict=approve 时不得携带问题或修复要求。")
        elif self.verdict == "revise":
            if not self.issues or not self.repair_instructions:
                raise ValueError("verdict=revise 时必须提供 issues 和 repair_instructions。")
        elif self.verdict == "clarify":
            if not self.clarification_question:
                raise ValueError("verdict=clarify 时必须提供 clarification_question。")
        elif self.verdict == "reject" and not self.issues:
            raise ValueError("verdict=reject 时必须提供 issues。")

        return self
