from __future__ import annotations

from typing import Any, Literal

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


class ProposedActionPayload(BaseModel):
    """Structured action proposal; Python assigns the handle and status."""

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    proposed_by: Literal[
        "assistant", "planner", "system"
    ] = "assistant"


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

    # First-class grounding bindings: user facts (canonical fact fields) and
    # deterministic derivations that support the answer without citations.
    used_fact_refs: list[str] = Field(
        default_factory=list, max_length=40
    )
    used_derivation_ids: list[str] = Field(
        default_factory=list, max_length=40
    )
    used_result_artifact_refs: list[str] = Field(
        default_factory=list, max_length=40
    )
    claim_bindings: list[dict[str, Any]] = Field(
        default_factory=list, max_length=40
    )
    primary_response_focus: dict[str, Any] | None = None

    # New-artifact proposals.  The model only proposes content with a
    # local_key; Python's ArtifactAllocator assigns the real handle.
    new_artifacts: list[dict[str, Any]] = Field(
        default_factory=list, max_length=40
    )
    focus_candidate: dict[str, Any] | None = Field(default=None)

    uncertainty: str | None = None

    disclaimer_required: bool = False

    # 每个案例的结构化最终标签（由模型生成，Python 只校验 enum/基数）。
    case_verdicts: dict[str, str] = Field(
        default_factory=dict
    )

    # Present only when the assistant needs user confirmation before executing
    # a concrete action (e.g. run a catalog query).  Python turns this into a
    # PendingAction in ConversationState; the model never supplies the handle.
    proposed_action: ProposedActionPayload | None = None

    # Typed state-claim bindings: the model declares whether a state-related
    # sentence is a current-turn mutation ACK (needs this turn's mutation
    # receipt) or a reference to already-committed state (needs an active
    # canonical fact).  Python validates the binding; it never relies on
    # keyword lists to decide the claim type.
    state_claim_bindings: list[dict[str, Any]] = Field(
        default_factory=list, max_length=20
    )

    @field_validator("state_claim_bindings")
    @classmethod
    def validate_state_claim_bindings(
        cls,
        value: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        allowed = {
            "committed_state_reference",
            "current_turn_mutation_ack",
        }
        cleaned: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            claim_type = str(item.get("claim_type") or "")
            if claim_type not in allowed:
                raise ValueError(
                    f"state_claim_bindings[{index}].claim_type 必须是 "
                    "committed_state_reference / "
                    "current_turn_mutation_ack"
                )
            cleaned.append(
                {
                    "claim_id": str(item.get("claim_id") or ""),
                    "claim_type": claim_type,
                    "fact_refs": [
                        str(ref)
                        for ref in (
                            item.get("fact_refs")
                            or []
                        )
                    ],
                    "mutation_receipt_ref": str(
                        item.get("mutation_receipt_ref")
                        or ""
                    ),
                }
            )
        return cleaned

    @field_validator("case_verdicts")
    @classmethod
    def validate_case_verdicts(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        allowed = {
            "determined",
            "conditional",
            "insufficient_evidence",
        }
        for case_id, verdict in value.items():
            if verdict not in allowed:
                raise ValueError(
                    f"case_verdicts[{case_id}] 必须是 "
                    "determined / conditional / insufficient_evidence"
                )
        return value

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
        "used_fact_refs",
        "used_derivation_ids",
        "used_result_artifact_refs",
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
