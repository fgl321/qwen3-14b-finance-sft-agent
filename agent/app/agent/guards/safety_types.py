from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SafetyDecision = Literal[
    "allow",
    "refuse",
    "rewrite",
    "ask_clarification",
]


RiskLevel = Literal[
    "low",
    "medium",
    "high",
]


class SafetyFinding(BaseModel):
    category: str = Field(description="风险类别")
    severity: RiskLevel = Field(description="风险等级")
    evidence: str = Field(description="触发风险的原文片段")
    reason: str = Field(description="为什么这个片段有风险")


class SafetyAssessment(BaseModel):
    safe: bool
    decision: SafetyDecision
    risk_level: RiskLevel
    findings: list[SafetyFinding] = Field(default_factory=list)
    explanation: str = Field(description="给开发者看的判断说明")
    user_message: str = Field(description="可以展示给用户的安全提示")


class SafetyGuardResult(BaseModel):
    assessment: SafetyAssessment
    usage: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    finish_reason: str | None = None


class SafetyRewriteResult(BaseModel):
    answer: str
    usage: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    finish_reason: str | None = None
