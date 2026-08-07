from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent_graph.schemas.synthesis_schema import (
    OutputGuardResult,
    SynthesisResult,
)


FinalResponseStatus = Literal[
    "completed",
    "clarification_required",
    "fallback",
]


class ModelInvocationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "synthesis",
        "output_guard",
    ]

    model: str | None = None
    finish_reason: str = ""

    usage: dict[str, Any] = Field(default_factory=dict)

    attempts: int = Field(default=1, ge=1)
    protocol_repaired: bool = False

    error: str | None = None


class FinalResponsePipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: FinalResponseStatus

    answer: str

    synthesis: SynthesisResult | None = None
    guard: OutputGuardResult | None = None

    model_invocations: list[
        ModelInvocationAudit
    ] = Field(default_factory=list)

    output_rewrites: int = Field(default=0, ge=0)

    usage: dict[str, Any] = Field(default_factory=dict)

    finish_reason: str
