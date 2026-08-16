from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agent_graph.prompts.synthesis_prompt import (
    OUTPUT_GUARD_REPAIR_PROMPT,
    OUTPUT_GUARD_SYSTEM_PROMPT,
)
from app.agent_graph.source_authority_prompt import (
    normalize_authority,
    source_authority_contract_message,
)
from app.agent_graph.schemas.loop_schema import AgentLoopResult
from app.agent_graph.schemas.synthesis_schema import (
    OutputGuardResult,
    SynthesisResult,
)
from app.core.logging import get_logger
from app.llm.structured_gateway import StructuredLLMGateway
from app.rag.context_governance import (
    DEFAULT_CONTEXT_BUDGET,
    compact_citation,
    compact_tool_results,
    trim_text,
)


logger = get_logger(__name__)


SUBMIT_GUARD_TOOL = "submit_output_guard_result"


_FORBIDDEN_PATTERNS = {
    "hidden_reasoning": re.compile(
        r"<\s*/?\s*think\b",
        re.IGNORECASE,
    ),
}


class OutputGuardLLMClient(Protocol):
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        thinking_enabled: bool = False,
        max_completion_tokens: int = 800,
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class OutputGuardRequest:
    request_id: str
    run_id: str

    user_message: str

    loop_result: AgentLoopResult
    synthesis: SynthesisResult

    citations: list[dict[str, Any]] = field(
        default_factory=list
    )
    # 用户上下文（短期记忆历史摘要 + 长期记忆事实）。
    # 这些是用户明确提供或已保存的个人事实，属于合法回答依据。
    context_summary: str = field(default="")
    completion_contract: dict[str, Any] = field(default_factory=dict)
    source_authority: Any | None = field(default=None)
    requirement_observations: list[dict[str, Any]] = field(
        default_factory=list
    )


class OutputGuardInvocationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: OutputGuardResult

    model: str | None = None
    finish_reason: str = ""

    usage: dict[str, Any] = Field(default_factory=dict)

    attempts: int = Field(default=1, ge=1)
    protocol_repaired: bool = False

    error: str | None = None


class OutputGuardProtocolError(ValueError):
    pass


def _guard_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_GUARD_TOOL,
            "description": (
                "提交最终金融回答的安全与一致性检查结果。"
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "pass",
                            "rewrite",
                            "fallback",
                        ],
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1500,
                    },
                    "risk_flags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "rewrite_instructions": {
                        "type": [
                            "string",
                            "null",
                        ],
                    },
                },
                "required": [
                    "verdict",
                    "reason",
                    "risk_flags",
                    "rewrite_instructions",
                ],
            },
        },
    }


def _parse_arguments(
    raw_arguments: Any,
) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments

    if not isinstance(raw_arguments, str):
        raise OutputGuardProtocolError(
            "Output Guard arguments 格式错误。"
        )

    try:
        payload = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise OutputGuardProtocolError(
            "Output Guard arguments 不是合法 JSON。"
        ) from exc

    if not isinstance(payload, dict):
        raise OutputGuardProtocolError(
            "Output Guard arguments 顶层必须是对象。"
        )

    return payload

def _normalize_guard_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    兼容模型可能产生的常见同义字段。

    最终仍统一转换为：
    - verdict
    - reason
    - risk_flags
    - rewrite_instructions
    """

    normalized = dict(payload)

    if "verdict" not in normalized:
        normalized["verdict"] = (
            normalized.pop("decision", None)
            or normalized.pop("action", None)
            or normalized.pop("result", None)
        )

    raw_verdict = str(
        normalized.get("verdict") or ""
    ).strip().lower()

    verdict_aliases = {
        "pass": "pass",
        "allow": "pass",
        "allowed": "pass",
        "approve": "pass",
        "approved": "pass",
        "safe": "pass",
        "ok": "pass",

        "rewrite": "rewrite",
        "revise": "rewrite",
        "modify": "rewrite",
        "repair": "rewrite",

        "fallback": "fallback",
        "reject": "fallback",
        "rejected": "fallback",
        "block": "fallback",
        "blocked": "fallback",
        "unsafe": "fallback",
    }

    normalized["verdict"] = (
        verdict_aliases.get(
            raw_verdict,
            raw_verdict,
        )
    )

    if "reason" not in normalized:
        normalized["reason"] = (
            normalized.pop(
                "explanation",
                None,
            )
            or normalized.pop(
                "message",
                None,
            )
            or normalized.pop(
                "feedback",
                None,
            )
            or "输出检查已完成。"
        )

    if "risk_flags" not in normalized:
        normalized["risk_flags"] = (
            normalized.pop("issues", None)
            or normalized.pop("findings", None)
            or normalized.pop("risks", None)
            or []
        )

    risk_flags = normalized.get(
        "risk_flags"
    )

    if risk_flags is None:
        normalized["risk_flags"] = []

    elif isinstance(risk_flags, str):
        cleaned_flag = risk_flags.strip()

        normalized["risk_flags"] = (
            [cleaned_flag]
            if cleaned_flag
            else []
        )

    elif not isinstance(risk_flags, list):
        normalized["risk_flags"] = [
            str(risk_flags)
        ]

    if "rewrite_instructions" not in normalized:
        normalized[
            "rewrite_instructions"
        ] = (
            normalized.pop(
                "rewrite_instruction",
                None,
            )
            or normalized.pop(
                "instructions",
                None,
            )
            or normalized.pop(
                "revision_instructions",
                None,
            )
        )

    rewrite_instructions = normalized.get(
        "rewrite_instructions"
    )

    if isinstance(rewrite_instructions, str):
        cleaned_rewrite = rewrite_instructions.strip()

        if (
            not cleaned_rewrite
            or cleaned_rewrite.lower()
            in {"null", "none", "nil"}
        ):
            normalized[
                "rewrite_instructions"
            ] = None
        else:
            normalized[
                "rewrite_instructions"
            ] = cleaned_rewrite

    if (
        normalized.get("verdict")
        == "rewrite"
        and not normalized.get(
            "rewrite_instructions"
        )
    ):
        normalized[
            "rewrite_instructions"
        ] = (
            str(normalized.get("reason") or "").strip()
            or "根据输出检查结果修正回答。"
        )

    # 删除模型可能附带、但不属于正式协议的字段。
    allowed_fields = {
        "verdict",
        "reason",
        "risk_flags",
        "rewrite_instructions",
    }

    return {
        key: value
        for key, value in normalized.items()
        if key in allowed_fields
    }


_SUCCESSFUL_RESULT_CLAIM_PATTERN = re.compile(
    r"计算结果|结果为|结果是|合理区间|应为|等于|"
    r"\d+(?:\.\d+)?\s*(?:元|万元|%|％)",
    re.IGNORECASE,
)

_FAILURE_DISCLOSURE_PATTERN = re.compile(
    r"无法|失败|未能|不能|暂时不能|没有产生|未产生",
    re.IGNORECASE,
)

_SEMANTIC_FIELD_SUBSTITUTION_PATTERN = re.compile(
    r"(?:必要支出.{0,24}(?:近似|代替|替代).{0,12}(?:年)?收入|"
    r"(?:年)?收入.{0,24}(?:以|用).{0,12}必要支出.{0,12}(?:近似|代替|替代)|"
    r"房贷余额.{0,24}(?:近似|代替|替代|作为).{0,12}(?:月供|每月还款))",
    re.IGNORECASE | re.DOTALL,
)

_UNVERIFIED_FACT_REUSE_PATTERN = re.compile(
    r"(?:无法确认|没有足够证据|未检索到).{0,40}(?:最高偿付限额|存款保险)"
    r"[\s\S]{0,600}(?:超过最高偿付限额|分散(?:至|到|在)不同投保机构)",
    re.IGNORECASE,
)

_STATE_MUTATION_CLAIM_PATTERN = re.compile(
    r"(?:"
    r"已(?:保存|记录|记住|更新|修改|删除|确认|写入|持久化|同步|添加|存入)|"
    r"保存(?:成功|完成)|"
    r"已(?:为|作为).{0,16}(?:保存|记录|写入)|"
    r"(?:信息|资料|内容|事实).{0,8}已(?:保存|记录)"
    r")",
    re.IGNORECASE,
)

_COMMITTED_STATE_REFERENCE_PATTERN = re.compile(
    r"(?:已生效|当前已|已经生效|当前有效|已确认的|已记录在案|目前是|"
    r"已(?:更新|修改|调整|变更|改为)为|"
    r"已(?:修改|更新|调整|变更)(?:后|过)?的|已保存的)",
    re.IGNORECASE,
)

_TECHNICAL_FAILURE_NEGATION_PATTERN = re.compile(
    r"(?:没有|无|不存在|未发生|未出现)(?:任何)?"
    r"技术(?:异常|故障|问题)|"
    r"技术(?:异常|故障|问题).{0,8}(?:没有|无|不存在)",
    re.IGNORECASE,
)

_DOCUMENT_CITATION_REQUIRED_PATTERN = re.compile(
    r"(?:必须|务必|严格).{0,24}(?:检索|文档|引用).{0,80}(?:上传|知识库|文档|资料)|"
    r"(?:上传|知识库|文档|资料).{0,80}(?:必须|务必|严格).{0,24}(?:检索|引用)",
    re.IGNORECASE | re.DOTALL,
)
_DOCUMENT_UNAVAILABLE_DISCLOSURE_PATTERN = re.compile(
    r"(?:未通过|没有|缺少|无法).{0,30}(?:文档证据|已验证引用|citation)|"
    r"(?:文档证据|已验证引用).{0,30}(?:未完成|不可用|不足)",
    re.IGNORECASE | re.DOTALL,
)
_GENERAL_KNOWLEDGE_SUBSTITUTION_PATTERN = re.compile(
    r"(?:通常|一般|通用).{0,30}(?:最高偿付限额|保存期限|存款保险|征信).{0,20}"
    r"(?:\d+\s*(?:万|年)|五十万|五年)",
    re.IGNORECASE | re.DOTALL,
)

_CLAIM_DATE_PATTERN = re.compile(
    r"20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
)
_CLAIM_SUBJECT_PATTERN = re.compile(r"《([^》]{1,40})》")


def _evidence_contains_date_for_subject(
    *,
    evidence: str,
    subject: str,
    date_text: str,
) -> bool:
    """Date and subject must co-occur in the same sentence/paragraph."""
    compact_evidence = re.sub(r"\s+", "", evidence)
    compact_date = re.sub(r"\s+", "", date_text)
    compact_subject = re.sub(r"\s+", "", subject)
    if compact_date not in compact_evidence:
        return False
    bare_subject = subject.strip("《》")
    for unit in re.split(r"[\n。；]", evidence):
        compact_unit = re.sub(r"\s+", "", unit)
        if compact_date in compact_unit and (
            compact_subject in compact_unit
            or bare_subject in compact_unit
        ):
            return True
    return False


def _detect_claim_evidence_attribution_mismatches(
    *,
    answer: str,
    citations: list[dict[str, Any]] | None,
) -> list[str]:
    """Citation existence does not entail claim attribution.

    A claim that binds a specific subject (e.g. 《存款保险条例》) to a
    specific date must be supported by the cited evidence within the same
    sentence/paragraph; otherwise the date may belong to a neighbouring
    section (parent-chunk boundary pollution).
    """
    citations_by_id = {
        str(item.get("citation_id") or ""): item
        for item in (citations or [])
        if item.get("citation_id") is not None
    }
    violations: list[str] = []
    for marker in re.finditer(r"\[(\d+)\]", answer):
        citation_id = marker.group(1)
        citation = citations_by_id.get(citation_id)
        if citation is None:
            continue
        claim_window = answer[
            max(0, marker.start() - 220) : marker.end() + 80
        ]
        claim_dates = [
            match.group(0)
            for match in _CLAIM_DATE_PATTERN.finditer(claim_window)
        ]
        subjects = [
            match.group(1)
            for match in _CLAIM_SUBJECT_PATTERN.finditer(claim_window)
        ]
        if not claim_dates or not subjects:
            continue
        evidence = str(
            citation.get("quote")
            or citation.get("text")
            or (citation.get("metadata") or {}).get(
                "evidence_excerpt"
            )
            or ""
        )[:1200]
        if not evidence:
            continue
        for subject in subjects:
            bare_subject = subject.strip("《》")
            compact_evidence = re.sub(r"\s+", "", evidence)
            if (
                subject not in evidence
                and bare_subject not in evidence
                and subject not in compact_evidence
                and bare_subject not in compact_evidence
            ):
                violations.append(
                    f"citation:{citation_id}:subject_missing:{subject}"
                )
                continue
            for date_text in claim_dates:
                if not _evidence_contains_date_for_subject(
                    evidence=evidence,
                    subject=subject,
                    date_text=date_text,
                ):
                    violations.append(
                        f"citation:{citation_id}:date_not_attached:"
                        f"{date_text}:{subject}"
                    )
    return list(dict.fromkeys(violations))


def deterministic_output_flags(
    synthesis: SynthesisResult,
    *,
    loop_result: AgentLoopResult | None = None,
    citations: list[dict[str, Any]] | None = None,
    user_message: str = "",
    source_authority: Any | None = None,
    requirement_observations: list[dict[str, Any]] | None = None,
    delivery_contract: str = "",
    result_reference_context: dict[str, Any] | None = None,
    canonical_fact_fields: list[str] | None = None,
    known_derivation_ids: list[str] | None = None,
    known_sub_artifact_ids: list[str] | None = None,
) -> list[str]:
    """
    不依赖第二个模型判断的最小证据一致性检查。

    这里不规定所有请求都必须调用工具：
    - 没有走工具路径时，允许 used_tool_call_ids 为空；
    - 一旦存在成功工具结果，引用的工具调用编号必须真实；
    - 工具全部失败时，不允许生成看起来成功的确定性计算结论。
    """

    flags: list[str] = []

    answer = synthesis.answer

    if not answer.strip():
        flags.append("empty_answer")

    if len(answer) > 8000:
        flags.append("answer_too_long")

    if _SEMANTIC_FIELD_SUBSTITUTION_PATTERN.search(answer):
        flags.append("semantic_field_substitution")

    if _UNVERIFIED_FACT_REUSE_PATTERN.search(answer):
        flags.append("unverified_fact_reused_downstream")

    mutation_context = result_reference_context or {}
    technical_failures = list(
        mutation_context.get("technical_failures") or []
    )
    if (
        technical_failures
        and _TECHNICAL_FAILURE_NEGATION_PATTERN.search(answer)
    ):
        flags.append("delivery_truth_conflict")
    requirement_observations = list(
        mutation_context.get("requirement_observations") or []
    )
    blocked_citation_ids = {
        str(citation_id)
        for item in requirement_observations
        if str(item.get("status") or "")
        in {
            "not_observed",
            "technical_unavailable",
            "assessment_protocol_failed",
        }
        for citation_id in (item.get("citation_ids") or [])
    }
    if blocked_citation_ids and (
        set(synthesis.used_citation_ids) & blocked_citation_ids
    ):
        flags.append("blocked_evidence_citation_used")
    committed_fields = {
        str(item)
        for item in (
            mutation_context.get("committed_fact_fields")
            or []
        )
    }
    current_turn_fields = {
        str(item)
        for item in (
            mutation_context.get(
                "current_turn_mutation_fields"
            )
            or []
        )
    }
    for binding in (
        synthesis.state_claim_bindings or []
    ):
        claim_type = str(binding.get("claim_type") or "")
        refs = [
            str(item)
            for item in (binding.get("fact_refs") or [])
        ]
        if claim_type == "committed_state_reference":
            if not refs or not (set(refs) <= committed_fields):
                flags.append(
                    "unverified_committed_state_reference"
                )
        elif claim_type == "current_turn_mutation_ack":
            if (
                not bool(
                    mutation_context.get(
                        "has_mutation_intent"
                    )
                )
                or (
                    refs
                    and not (
                        set(refs) <= current_turn_fields
                    )
                )
            ):
                flags.append(
                    "unverified_state_mutation_claim"
                )

    if (
        _STATE_MUTATION_CLAIM_PATTERN.search(answer)
        and not bool(
            mutation_context.get("has_mutation_intent")
        )
        and not (synthesis.state_claim_bindings or [])
    ):
        committed_fields_list = [
            str(item)
            for item in committed_fields
        ]
        committed_reference = bool(
            _COMMITTED_STATE_REFERENCE_PATTERN.search(answer)
            and bool(committed_fields_list)
        )
        if not committed_reference:
            flags.append("unverified_state_mutation_claim")

    for flag_name, pattern in (
        _FORBIDDEN_PATTERNS.items()
    ):
        if pattern.search(answer):
            flags.append(flag_name)

    if loop_result is not None:
        successful_tool_ids = {
            item.tool_call_id
            for item in loop_result.tool_results
            if item.success
        }
        failed_tool_ids = {
            item.tool_call_id
            for item in loop_result.tool_results
            if not item.success
        }
        used_tool_ids = set(
            synthesis.used_tool_call_ids
        )

        if (
            not successful_tool_ids
            and re.search(r"已验证(?:的)?工具计算|工具验证(?:结果|计算)", answer)
        ):
            flags.append("verified_tool_claim_without_tool_result")

        if used_tool_ids - successful_tool_ids:
            flags.append(
                "invalid_used_tool_call_ids"
            )

        if (
            successful_tool_ids
            and not used_tool_ids
        ):
            flags.append(
                "missing_used_tool_call_ids"
            )

        if used_tool_ids & failed_tool_ids:
            flags.append(
                "failed_tool_result_referenced"
            )

        if (
            failed_tool_ids
            and not successful_tool_ids
            and _SUCCESSFUL_RESULT_CLAIM_PATTERN.search(
                answer
            )
            and not _FAILURE_DISCLOSURE_PATTERN.search(
                answer
            )
        ):
            flags.append(
                "successful_conclusion_after_tool_failure"
            )

    allowed_citation_ids = {
        str(item.get("citation_id"))
        for item in (citations or [])
        if item.get("citation_id")
    }
    used_citation_ids = set(
        synthesis.used_citation_ids
    )

    if used_citation_ids - allowed_citation_ids:
        flags.append(
            "invalid_used_citation_ids"
        )

    if _detect_claim_evidence_attribution_mismatches(
        answer=answer,
        citations=citations,
    ):
        flags.append("claim_evidence_attribution_mismatch")

    document_citations_required = bool(
        _DOCUMENT_CITATION_REQUIRED_PATTERN.search(user_message)
    )
    if document_citations_required and not allowed_citation_ids:
        if not _DOCUMENT_UNAVAILABLE_DISCLOSURE_PATTERN.search(answer):
            flags.append("required_citations_unavailable_not_disclosed")
        if _GENERAL_KNOWLEDGE_SUBSTITUTION_PATTERN.search(answer):
            flags.append("required_document_fact_substituted_from_memory")

    flags.extend(
        _source_authority_deterministic_flags(
            synthesis=synthesis,
            citations=citations,
            source_authority=source_authority,
            delivery_contract=delivery_contract,
            result_reference_context=result_reference_context,
            canonical_fact_fields=canonical_fact_fields,
            known_derivation_ids=known_derivation_ids,
            known_sub_artifact_ids=known_sub_artifact_ids,
        )
    )

    return list(dict.fromkeys(flags))


def _source_authority_deterministic_flags(
    *,
    synthesis: SynthesisResult,
    citations: list[dict[str, Any]] | None,
    source_authority: Any | None,
    delivery_contract: str,
    result_reference_context: dict[str, Any] | None,
    canonical_fact_fields: list[str] | None,
    known_derivation_ids: list[str] | None,
    known_sub_artifact_ids: list[str] | None,
) -> list[str]:
    """Minimal structural Source Authority checks on the final answer.

    These checks use only structured synthesis metadata (used citations,
    used tool ids, the delivery contract) and the typed authority contract.
    They do not re-interpret the user's natural language.
    """

    authority = normalize_authority(source_authority)
    if authority is None:
        return []

    flags: list[str] = []

    if (
        authority.selected_documents == "forbidden"
        and synthesis.used_citation_ids
    ):
        flags.append("source_authority_citation_forbidden")

    allowed_citation_ids = {
        str(item.get("citation_id") or "")
        for item in (citations or [])
        if item.get("citation_id")
    }
    used_citations = set(synthesis.used_citation_ids) & allowed_citation_ids
    used_tools = set(synthesis.used_tool_call_ids)
    used_fact_refs = set(synthesis.used_fact_refs)
    used_derivation_ids = set(synthesis.used_derivation_ids)
    used_result_artifact_refs = set(
        synthesis.used_result_artifact_refs
    )
    has_grounding_bindings = bool(
        used_citations
        or used_tools
        or used_fact_refs
        or used_derivation_ids
        or used_result_artifact_refs
    )

    known_fact_fields = {
        str(item) for item in (canonical_fact_fields or [])
    }
    known_derivations = {
        str(item) for item in (known_derivation_ids or [])
    }
    if used_fact_refs - known_fact_fields:
        flags.append("invalid_used_fact_refs")
    if used_derivation_ids - known_derivations:
        flags.append("invalid_used_derivation_ids")
    known_sub_artifacts = {
        str(item)
        for item in (known_sub_artifact_ids or [])
    }
    if used_result_artifact_refs - known_sub_artifacts:
        flags.append("invalid_used_result_artifact_refs")

    document_grounding_required = bool(
        delivery_contract.strip()
        and "required_outputs" in delivery_contract
    )

    if (
        document_grounding_required
        and not has_grounding_bindings
        and (
            authority.general_model_knowledge == "forbidden"
            or authority.domain_heuristics == "forbidden"
        )
        and not _DOCUMENT_UNAVAILABLE_DISCLOSURE_PATTERN.search(
            synthesis.answer
        )
    ):
        flags.append("source_authority_ungrounded_answer")

    if (
        authority.general_model_knowledge == "forbidden"
        and any(
            marker in synthesis.answer
            for marker in (
                "通用金融知识",
                "通用金融原则",
                "一般金融原则",
                "通用金融常识",
                "金融常识",
                "一般常识",
            )
        )
    ):
        flags.append(
            "source_authority_general_knowledge_used"
        )

    if (
        authority.general_model_knowledge == "forbidden"
        and not has_grounding_bindings
        and bool(
            (result_reference_context or {}).get(
                "has_claims"
            )
        )
    ):
        flags.append(
            "source_authority_result_reference_not_cited"
        )

    if (
        authority.general_model_knowledge == "forbidden"
        and authority.domain_heuristics == "forbidden"
        and not has_grounding_bindings
        and len(synthesis.answer) > 200
    ):
        flags.append(
            "source_authority_ungrounded_explanation"
        )

    return flags


def _deterministic_rewrite_instructions(
    flags: list[str],
) -> str:
    instructions: list[str] = []

    if "missing_used_tool_call_ids" in flags:
        instructions.append(
            "如果回答使用了成功工具结果，"
            "必须填写真实的 used_tool_call_ids。"
        )

    if (
        "invalid_used_tool_call_ids" in flags
        or "failed_tool_result_referenced" in flags
    ):
        instructions.append(
            "删除不存在或失败的工具调用编号，"
            "只能引用成功工具结果。"
        )

    if (
        "successful_conclusion_after_tool_failure"
        in flags
    ):
        instructions.append(
            "工具没有产生成功结果，"
            "不得输出成功计算结论；"
            "应明确说明本次无法得到可靠结果。"
        )

    if "verified_tool_claim_without_tool_result" in flags:
        instructions.append(
            "当前没有任何成功工具调用，删除“已验证工具计算”等来源标签；"
            "不得把模型心算或文档规则推导冒充为工具验证结果。"
        )

    if "semantic_field_substitution" in flags:
        instructions.append(
            "删除跨金融概念替代：家庭必要支出不能近似为家庭收入，"
            "房贷余额不能替代月供。缺少规则要求的字段时停止该项数值推导。"
        )

    if "claim_evidence_attribution_mismatch" in flags:
        instructions.append(
            "回答中引用具体法规/制度并给出日期时，"
            "该日期必须与同一主体出现在证据的同一句或同一段内；"
            "若证据无法支持该归属，删除该引用，"
            "并把结论改为通用说明或明确标注证据不足。"
        )


    if "unverified_fact_reused_downstream" in flags:
        instructions.append(
            "前文已声明某制度事实未获得文档证据，后文不得继续把该事实作为"
            "建议前提；删除该推导，或明确移入【通用金融建议】且不得暗示来自文档。"
        )

    if "unverified_state_mutation_claim" in flags:
        instructions.append(
            "回答声称已保存/已记住/已更新/已修改/已删除/已确认/已记录了状态，"
            "但本轮没有经过确定性状态提交，必须删除这些“已执行”表述，"
            "或改为明确的“未确认/需要用户确认”。"
        )

    if "unverified_committed_state_reference" in flags:
        instructions.append(
            "state_claim_bindings 中声明了 committed_state_reference，"
            "但 fact_refs 引用的字段不在当前已提交 canonical facts 中；"
            "删除该 binding，并把正文中的“已生效/已修改后”表述"
            "改为“本轮提供的/未提交的”或直接删除。"
        )

    if "delivery_truth_conflict" in flags:
        instructions.append(
            "本轮存在已记录的技术失败（检索服务异常/证据评估协议失败/"
            "记忆读取异常），必须如实披露，不得声称“没有技术异常”；"
            "列出失败项并说明对应部分未完成。"
        )

    if "blocked_evidence_citation_used" in flags:
        instructions.append(
            "回答引用了证据状态为 not_observed / technical_unavailable / "
            "assessment_protocol_failed 的要求所绑定的引用；"
            "删除这些引用，并把对应文档结论改为“未覆盖/技术异常/未验证”。"
        )

    if "invalid_used_citation_ids" in flags:
        instructions.append(
            "删除不存在的引用编号，"
            "只能使用系统提供的 citation_id。"
        )

    if "required_citations_unavailable_not_disclosed" in flags:
        instructions.append(
            "明确说明文档证据审核未完成或当前没有已验证引用，并将文档要求项标记为未完成。"
        )

    if "required_document_fact_substituted_from_memory" in flags:
        instructions.append(
            "删除以‘通常/一般’等模型记忆替代 required document evidence 的制度数值；"
            "只保留已验证工具计算，文档制度结论标记为未完成。"
        )

    if "source_authority_citation_forbidden" in flags:
        instructions.append(
            "Source Authority 禁止文档引用时，删除所有 used_citation_ids 和正文中的 [n] 引用，"
            "不得声称“根据文档/条款”。"
        )

    if "source_authority_ungrounded_answer" in flags:
        instructions.append(
            "delivery_contract 要求文档证据，但答案既没有 used_tool_call_ids 也没有 used_citation_ids；"
            "必须明确说明未找到足够文档证据（或标记为未完成），"
            "不得用模型常识或经验法则补出确定性结论。"
        )

    if "source_authority_general_knowledge_used" in flags:
        instructions.append(
            "general_model_knowledge=forbidden 时，删除所有以“通用金融知识/原则”"
            "名义输出的无支撑规则；只保留用户事实、已验证工具结果和文档引用支撑的内容，"
            "无法支撑的结论改为明确说明当前证据不足。"
        )

    if "source_authority_result_reference_not_cited" in flags:
        instructions.append(
            "本轮引用了先前结果（该结果包含结构化 claims/citations）；"
            "general_model_knowledge=forbidden 时，必须基于该结果的 claims 回答"
            "并绑定其 citations，或明确说明该结果无法支撑当前问题；"
            "不得改用通用金融知识重述。"
        )

    if "source_authority_ungrounded_explanation" in flags:
        instructions.append(
            "general_model_knowledge/domain_heuristics=forbidden 且没有引用和工具结果时，"
            "如果答案基于用户事实与确定性推导（例如 90−20=70），"
            "重写时必须同时填写 used_fact_refs（如 cash、down_payment）"
            "与 used_derivation_ids（如 CALC_1），并保持简短；"
            "如果没有任何可绑定来源，才输出‘当前证据不足’并结束。"
            "不得再输出任何风险规律、平台机制或金融原理的陈述。"
        )

    if "invalid_used_fact_refs" in flags:
        instructions.append(
            "used_fact_refs 中包含了当前 EffectiveTaskContract 不存在的字段，"
            "删除不存在的 fact 引用，只保留 canonical_facts 中真实存在的字段。"
        )

    if "invalid_used_derivation_ids" in flags:
        instructions.append(
            "used_derivation_ids 中包含了当前结构化结果不存在的推导句柄，"
            "删除不存在的 derivation 引用，只保留计算结果中的 CALC_n。"
        )

    if "invalid_used_result_artifact_refs" in flags:
        instructions.append(
            "used_result_artifact_refs 中包含了不存在的 RESULT_n.CLAIM_n/"
            "CALC_n/CONCLUSION_n 引用，删除不存在的子产物引用。"
        )

    if "hidden_reasoning" in flags:
        instructions.append(
            "删除思考标签，不要输出 <think> 等隐藏推理内容。"
        )

    if not instructions:
        instructions.append(
            "根据确定性输出检查结果修正回答，"
            "只保留有真实依据的内容。"
        )

    return "".join(instructions)


class LLMOutputGuard:
    def __init__(
        self,
        *,
        llm_client: OutputGuardLLMClient,
        max_completion_tokens: int = 800,
        max_protocol_repairs: int = 1,
    ) -> None:
        if max_completion_tokens <= 0:
            raise ValueError(
                "max_completion_tokens 必须大于 0。"
            )

        if max_protocol_repairs < 0:
            raise ValueError(
                "max_protocol_repairs 不能小于 0。"
            )

        self.llm_client = llm_client
        self.max_completion_tokens = (
            max_completion_tokens
        )
        self.max_protocol_repairs = (
            max_protocol_repairs
        )

    def build_messages(
        self,
        request: OutputGuardRequest,
    ) -> list[dict[str, Any]]:
        successful_tool_results, tool_governance = (
            compact_tool_results(
                [
                    item.model_dump(mode="json")
                    for item in request.loop_result.tool_results
                    if item.success
                ]
            )
        )

        failed_tool_results, _failed_tool_governance = (
            compact_tool_results(
                [
                    item.model_dump(mode="json")
                    for item in request.loop_result.tool_results
                    if not item.success
                ]
            )
        )

        used_citation_ids = set(
            request.synthesis.used_citation_ids
        )
        compacted_citations = [
            compact_citation(item)
            for item in request.citations
            if item.get("citation_id")
            and str(item.get("citation_id"))
            in used_citation_ids
        ]

        observation_summary = [
            {
                "requirement_id": (
                    observation.get("requirement_id")
                ),
                "status": observation.get("status"),
                "citation_ids": list(
                    observation.get("citation_ids") or []
                ),
                "conflict_ids": list(
                    observation.get("conflict_ids") or []
                ),
                "reason": (
                    str(observation.get("reason") or "")[:200]
                    if observation.get("reason")
                    else None
                ),
            }
            for observation in (
                request.requirement_observations
            )
        ]

        guarded_context_summary = trim_text(
            request.context_summary,
            DEFAULT_CONTEXT_BUDGET.guard_context_tokens,
        )

        payload = {
            "user_message": request.user_message,
            "draft_synthesis": (
                request.synthesis.model_dump(
                    mode="json"
                )
            ),
            "successful_tool_results": successful_tool_results,
            "failed_tool_results": failed_tool_results,
            "evidence_contract": {
                "used_tool_call_ids": list(
                    request.synthesis.used_tool_call_ids
                ),
                "successful_tool_call_ids": [
                    item.tool_call_id
                    for item
                    in request.loop_result.tool_results
                    if item.success
                ],
                "failed_tool_call_ids": [
                    item.tool_call_id
                    for item
                    in request.loop_result.tool_results
                    if not item.success
                ],
                "direct_answer_without_tools_allowed": (
                    not request.loop_result.tool_results
                ),
            },
            "citations": compacted_citations,
            "completion_contract": request.completion_contract,
            "source_authority": (
                normalize_authority(
                    request.source_authority
                ).model_dump(mode="json")
                if normalize_authority(
                    request.source_authority
                )
                is not None
                else None
            ),
            "requirement_observations": (
                observation_summary
            ),
            "user_context": {
                "context_summary": guarded_context_summary,
                "note": (
                    "context_summary 中的内容来自短期对话历史或"
                    "用户已确认的长期记忆事实，是合法回答依据。"
                    "基于这些事实回答（例如用户自己提供的收入、"
                    "家庭支出）不得判定为伪造或缺少证据。"
                ),
            },
            "context_governance": {
                "tool_results": tool_governance,
                "citation_count": len(compacted_citations),
                "observation_count": len(observation_summary),
                "guard_context_tokens": (
                    DEFAULT_CONTEXT_BUDGET.guard_context_tokens
                ),
            },
        }

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": OUTPUT_GUARD_SYSTEM_PROMPT,
            },
        ]

        authority_message = source_authority_contract_message(
            request.source_authority
        )
        if authority_message:
            messages.append(
                {
                    "role": "system",
                    "content": authority_message,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    "请检查以下最终回答草稿：\n"
                    f"{json.dumps(payload, ensure_ascii=False)}"
                ),
            }
        )

        return messages

    async def guard(
        self,
        request: OutputGuardRequest,
    ) -> OutputGuardInvocationResult:
        logger.info(
            "output_guard_deterministic_input",
            request_id=request.request_id,
            run_id=request.run_id,
            source_authority=(
                request.source_authority
                if isinstance(request.source_authority, dict)
                else (
                    normalize_authority(
                        request.source_authority
                    ).model_dump(mode="json")
                    if normalize_authority(
                        request.source_authority
                    )
                    is not None
                    else None
                )
            ),
            answer_len=len(request.synthesis.answer),
            used_tools=list(
                request.synthesis.used_tool_call_ids
            ),
            used_citations=list(
                request.synthesis.used_citation_ids
            ),
            citations_count=len(request.citations),
        )
        deterministic_flags = (
            deterministic_output_flags(
                request.synthesis,
                loop_result=request.loop_result,
                citations=request.citations,
                user_message=request.user_message,
                source_authority=request.source_authority,
                requirement_observations=(
                    request.requirement_observations
                ),
                delivery_contract=str(
                    (request.completion_contract or {}).get(
                        "delivery_contract"
                    )
                    or ""
                ),
                result_reference_context=(
                    (request.completion_contract or {}).get(
                        "result_reference_context"
                    )
                    or {}
                ),
                canonical_fact_fields=(
                    (request.completion_contract or {}).get(
                        "canonical_fact_fields"
                    )
                    or []
                ),
                known_derivation_ids=(
                    (request.completion_contract or {}).get(
                        "known_derivation_ids"
                    )
                    or []
                ),
                known_sub_artifact_ids=(
                    (request.completion_contract or {}).get(
                        "known_sub_artifact_ids"
                    )
                    or []
                ),
            )
        )

        if deterministic_flags:
            return OutputGuardInvocationResult(
                result=OutputGuardResult(
                    verdict="rewrite",
                    reason=(
                        "确定性输出规则检测到安全或"
                        "证据一致性问题。"
                    ),
                    risk_flags=deterministic_flags,
                    rewrite_instructions=(
                        _deterministic_rewrite_instructions(
                            deterministic_flags
                        )
                    ),
                )
            )

        messages = self.build_messages(request)

        logger.info(
            "llm_output_guard_started",
            request_id=request.request_id,
            run_id=request.run_id,
        )

        gateway = StructuredLLMGateway(self.llm_client)
        structured = await gateway.invoke_tool(
            schema=OutputGuardResult,
            messages=messages,
            tools=[_guard_tool_definition()],
            expected_tool_name=SUBMIT_GUARD_TOOL,
            stage="output_guard",
            max_completion_tokens=max(self.max_completion_tokens, 1200),
            max_protocol_repairs=self.max_protocol_repairs,
            normalize=_normalize_guard_payload,
        )
        if structured.parsed is None:
            service_failed = structured.status == "service_failed"
            risk_flag = (
                "guard_service_unavailable"
                if service_failed
                else "guard_protocol_failure"
            )
            logger.warning(
                "llm_output_guard_degraded",
                request_id=request.request_id,
                run_id=request.run_id,
                status=structured.status,
                attempts=structured.attempts,
                validation_errors=structured.validation_errors,
            )
            return OutputGuardInvocationResult(
                result=OutputGuardResult(
                    verdict="fallback",
                    reason=(
                        "输出安全检查服务不可用。"
                        if service_failed
                        else "输出检查器连续返回无效协议。"
                    ),
                    risk_flags=[risk_flag],
                ),
                model=structured.model,
                finish_reason=structured.finish_reason,
                usage=structured.usage,
                attempts=structured.attempts,
                protocol_repaired=structured.attempts > 1,
                error=(structured.validation_errors[-1]["error"]
                       if structured.validation_errors else structured.status),
            )

        logger.info(
            "llm_output_guard_finished",
            request_id=request.request_id,
            run_id=request.run_id,
            verdict=structured.parsed.verdict,
            attempts=structured.attempts,
            protocol_repaired=structured.status == "repaired",
            model=structured.model,
            finish_reason=structured.finish_reason,
            usage=structured.usage,
        )
        return OutputGuardInvocationResult(
            result=structured.parsed,
            model=structured.model,
            finish_reason=structured.finish_reason,
            usage=structured.usage,
            attempts=structured.attempts,
            protocol_repaired=structured.status == "repaired",
        )

    def _parse_assistant_message(
        self,
        assistant_message: dict[str, Any],
    ) -> OutputGuardResult:
        if not isinstance(assistant_message, dict):
            raise OutputGuardProtocolError(
                "Output Guard message 不是对象。"
            )

        tool_calls = (
            assistant_message.get("tool_calls") or []
        )

        if (
            not isinstance(tool_calls, list)
            or len(tool_calls) != 1
        ):
            raise OutputGuardProtocolError(
                "Output Guard 必须调用一次 "
                "submit_output_guard_result。"
            )

        function_payload = tool_calls[0].get(
            "function"
        )

        if not isinstance(function_payload, dict):
            raise OutputGuardProtocolError(
                "Output Guard 工具调用缺少 function。"
            )

        tool_name = str(
            function_payload.get("name") or ""
        )

        if tool_name != SUBMIT_GUARD_TOOL:
            raise OutputGuardProtocolError(
                f"Output Guard 调用了非法工具："
                f"{tool_name}"
            )

        raw_payload = _parse_arguments(
            function_payload.get("arguments")
        )

        payload = _normalize_guard_payload(
            raw_payload
        )

        try:
            return OutputGuardResult.model_validate(
                payload
            )
        except Exception as exc:
            raise OutputGuardProtocolError(
                "OutputGuardResult 校验失败："
                f"{type(exc).__name__}: {exc}"
            ) from exc
