from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


OutputGuardVerdict = Literal[
    "pass",
    "rewrite",
    "fallback",
]


class SynthesisResult(BaseModel):
    """
    最终回答生成器的结构化输出。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    answer: str = Field(
        min_length=1,
        max_length=8000,
    )

    used_tool_call_ids: list[str] = Field(
        default_factory=list
    )

    used_citation_ids: list[str] = Field(
        default_factory=list
    )

    uncertainty: str | None = None

    disclaimer_required: bool = False

    @field_validator("answer")
    @classmethod
    def validate_answer(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "answer 不能为空。"
            )

        if "<think" in cleaned.lower():
            raise ValueError(
                "answer 不能包含思考标签。"
            )

        return cleaned

    @field_validator(
        "used_tool_call_ids",
        "used_citation_ids",
    )
    @classmethod
    def normalize_identifier_list(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = str(value).strip()

            if not cleaned:
                continue

            if cleaned in seen:
                continue

            seen.add(cleaned)
            normalized.append(cleaned)

        return normalized

    @field_validator("uncertainty")
    @classmethod
    def normalize_uncertainty(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        if cleaned.lower() in {
            "null",
            "none",
            "nil",
        }:
            return None

        return cleaned or None


class OutputGuardResult(BaseModel):
    """
    最终输出检查器的统一协议。

    pass：
        草稿可以直接返回。

    rewrite：
        草稿需要重写，并且必须提供重写要求。

    fallback：
        当前草稿无法安全修复。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    verdict: OutputGuardVerdict

    reason: str = Field(
        min_length=1,
        max_length=1500,
    )

    risk_flags: list[str] = Field(
        default_factory=list
    )

    rewrite_instructions: str | None = None

    @field_validator("reason")
    @classmethod
    def validate_reason(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "reason 不能为空。"
            )

        return cleaned

    @field_validator("risk_flags")
    @classmethod
    def normalize_risk_flags(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = str(value).strip()

            if not cleaned:
                continue

            if cleaned in seen:
                continue

            seen.add(cleaned)
            normalized.append(cleaned)

        return normalized

    @field_validator("rewrite_instructions")
    @classmethod
    def normalize_rewrite_instructions(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()

        if cleaned.lower() in {
            "null",
            "none",
            "nil",
        }:
            return None

        return cleaned or None

    @model_validator(mode="after")
    def validate_verdict_consistency(
        self,
    ) -> "OutputGuardResult":
        if (
            self.verdict == "rewrite"
            and not self.rewrite_instructions
        ):
            raise ValueError(
                "verdict=rewrite 时必须提供 "
                "rewrite_instructions。"
            )

        if (
            self.verdict != "rewrite"
            and self.rewrite_instructions is not None
        ):
            raise ValueError(
                "只有 verdict=rewrite 时才能提供 "
                "rewrite_instructions。"
            )

        return self
