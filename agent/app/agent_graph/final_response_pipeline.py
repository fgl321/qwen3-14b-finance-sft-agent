from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

from app.agent_graph.llm_output_guard import (
    LLMOutputGuard,
    OutputGuardRequest,
)
from app.agent_graph.llm_synthesizer import (
    LLMAnswerSynthesizer,
    SynthesisRequest,
)
from app.agent_graph.runtime.agent_limits import (
    AgentLimits,
    DEFAULT_AGENT_LIMITS,
)
from app.agent_graph.schemas.final_response_schema import (
    FinalResponsePipelineResult,
    ModelInvocationAudit,
)
from app.agent_graph.schemas.loop_schema import AgentLoopResult
from app.agent_graph.schemas.synthesis_schema import (
    SynthesisResult,
)


_STANDARD_DISCLAIMER = (
    "以上内容用于一般性金融知识与规划参考，"
    "不构成具体投资、保险或交易建议。"
)

# Guard 无法通过整篇重写修复的上游约束类 violation：
# 没有可用文档证据时，Synthesis 重写再多遍也造不出真实 citation。
_IMMUTABLE_GUARD_FLAGS = frozenset(
    {
        "required_citations_unavailable_not_disclosed",
    }
)

logger = get_logger(__name__)


@dataclass(slots=True)
class FinalResponseRequest:
    request_id: str
    run_id: str

    user_message: str
    loop_result: AgentLoopResult

    context_summary: str = ""

    citations: list[dict] = field(
        default_factory=list
    )
    # Resolved document scope from the request boundary.  When non-empty, the
    # pipeline enforces a deterministic citation scope guard and may regenerate
    # the answer exactly once before failing closed.
    allowed_document_ids: list[str] = field(
        default_factory=list
    )
    scope_snapshot: dict[str, Any] = field(
        default_factory=dict
    )
    delivery_contract: str = field(default="")
    source_authority: Any = field(default=None)
    requirement_observations: list[dict[str, Any]] = field(
        default_factory=list
    )
    result_reference_context: dict[str, Any] = field(
        default_factory=dict
    )
    canonical_fact_fields: list[str] = field(
        default_factory=list
    )
    known_derivation_ids: list[str] = field(
        default_factory=list
    )
    known_sub_artifact_ids: list[str] = field(
        default_factory=list
    )


def _citation_scope_violations(
    citations: list[dict],
    synthesis: SynthesisResult,
    *,
    allowed_document_ids: list[str],
    scope_snapshot: dict[str, Any],
) -> list[str]:
    """Deterministic citation scope check (ids + document version)."""
    if not allowed_document_ids:
        return []
    allowed = set(allowed_document_ids)
    citation_by_id = {
        str(citation.get("citation_id") or ""): citation
        for citation in citations
        if citation.get("citation_id") is not None
    }
    violations: list[str] = []
    for used_id in (synthesis.used_citation_ids or []):
        citation = citation_by_id.get(str(used_id))
        if citation is None:
            violations.append(f"unknown_citation:{used_id}")
            continue
        document_id = str(citation.get("document_id") or "").strip()
        if not document_id:
            continue
        if document_id not in allowed:
            violations.append(f"citation_scope:{document_id}")
            continue
        snapshot = scope_snapshot.get(document_id)
        if not snapshot:
            continue
        citation_version = str(
            citation.get("document_version")
            or citation.get("version")
            or ""
        ).strip()
        snapshot_version = str(
            snapshot.get("document_version") or ""
        ).strip()
        if citation_version and snapshot_version and (
            citation_version != snapshot_version
        ):
            violations.append(
                f"citation_version:{document_id}"
            )
    return violations


def _guard_violation_fingerprint(
    guard: Any,
) -> str:
    """Fingerprint a guard rewrite so immutable violations do not loop."""
    risk_flags = sorted(str(item) for item in (guard.risk_flags or []))
    instructions = str(guard.rewrite_instructions or "")[:300]
    return f"{','.join(risk_flags)}::{instructions}"


def _append_disclaimer(
    synthesis: SynthesisResult,
) -> SynthesisResult:
    if not synthesis.disclaimer_required:
        return synthesis

    answer = synthesis.answer.rstrip()

    if "不构成" not in answer:
        answer = (
            f"{answer}\n\n{_STANDARD_DISCLAIMER}"
        )

    return synthesis.model_copy(
        update={
            "answer": answer,
        }
    )


def _safe_fallback_answer(
    loop_result: AgentLoopResult,
) -> str:
    successful_tools = [
        item.tool_name
        for item in loop_result.tool_results
        if item.success
    ]

    if successful_tools:
        return (
            "相关计算工具已经完成，但最终回答的安全或一致性"
            "检查未通过，因此暂时不能给出确定结论。"
            "请稍后重试，或缩小问题范围后重新提问。"
        )

    return (
        "抱歉，我暂时无法安全完成这个问题的分析。"
        "可能的原因是信息不完整，或问题超出了当前能力范围。"
        "请换个问法，或补充必要信息后重试。"
    )


def _usage_totals(
    usage: dict[str, Any],
) -> dict[str, int]:
    """抽取与最终 usage 相同的三个统计键。"""
    totals: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            totals[key] = int(value)
    return totals


def _sum_invocations_usage(
    invocations: list[Any],
) -> dict[str, int]:
    """把 Planner / Reviewer 的多次调用 usage 汇总为单份统计。"""
    totals: dict[str, int] = defaultdict(int)
    for item in invocations:
        for key, value in _usage_totals(item.usage or {}).items():
            totals[key] += value
    return dict(totals)


def _add_usage_totals(
    totals: dict[str, int],
    usage: dict[str, Any],
) -> None:
    for key, value in _usage_totals(usage).items():
        totals[key] += value


class FinalResponsePipeline:
    def __init__(
        self,
        *,
        synthesizer: LLMAnswerSynthesizer,
        output_guard: LLMOutputGuard,
        limits: AgentLimits = DEFAULT_AGENT_LIMITS,
    ) -> None:
        self.synthesizer = synthesizer
        self.output_guard = output_guard
        self.limits = limits

    async def run(
        self,
        request: FinalResponseRequest,
    ) -> FinalResponsePipelineResult:
        loop_result = request.loop_result

        planner_usage = _sum_invocations_usage(
            loop_result.planner_invocations
        )
        reviewer_usage = _sum_invocations_usage(
            loop_result.review_invocations
        )
        usage_by_stage: dict[str, Any] = {}
        if planner_usage:
            usage_by_stage["planner"] = planner_usage
        if reviewer_usage:
            usage_by_stage["reviewer"] = reviewer_usage

        usage_totals: dict[str, int] = defaultdict(int)
        _add_usage_totals(usage_totals, planner_usage)
        _add_usage_totals(usage_totals, reviewer_usage)
        citation_regeneration_count = 0

        if (
            loop_result.status
            == "clarification_required"
        ):
            question = (
                loop_result.clarification_question
                or "请补充完成当前任务所需的信息。"
            )

            return FinalResponsePipelineResult(
                status="clarification_required",
                answer=question,
                finish_reason=(
                    "clarification_required"
                ),
                citation_regeneration_count=(
                    citation_regeneration_count
                ),
                usage=dict(usage_totals),
                usage_by_stage=usage_by_stage,
            )

        if loop_result.status != "completed":
            return FinalResponsePipelineResult(
                status="fallback",
                answer=_safe_fallback_answer(
                    loop_result
                ),
                citation_regeneration_count=(
                    citation_regeneration_count
                ),
                finish_reason=(
                    f"loop_{loop_result.finish_reason}"
                ),
                usage=dict(usage_totals),
                usage_by_stage=usage_by_stage,
            )

        invocation_audits: list[
            ModelInvocationAudit
        ] = []

        rewrite_instructions = ""
        rewrite_count = 0
        guard_retry_count = 0
        seen_guard_fingerprints: set[str] = set()

        last_synthesis: SynthesisResult | None = None
        last_guard = None

        while True:
            synthesis_invocation = (
                await self.synthesizer.synthesize(
                    SynthesisRequest(
                        request_id=request.request_id,
                        run_id=request.run_id,
                        user_message=(
                            request.user_message
                        ),
                        loop_result=loop_result,
                        context_summary=(
                            request.context_summary
                        ),
                        citations=request.citations,
                        rewrite_instructions=(
                            rewrite_instructions
                        ),
                        delivery_contract=(
                            request.delivery_contract
                        ),
                        source_authority=(
                            request.source_authority
                        ),
                        requirement_observations=(
                            request.requirement_observations
                        ),
                    )
                )
            )

            invocation_audits.append(
                ModelInvocationAudit(
                    stage="synthesis",
                    model=synthesis_invocation.model,
                    finish_reason=(
                        synthesis_invocation.finish_reason
                    ),
                    usage=synthesis_invocation.usage,
                    attempts=(
                        synthesis_invocation.attempts
                    ),
                    protocol_repaired=(
                        synthesis_invocation.protocol_repaired
                    ),
                    error=synthesis_invocation.error,
                )
            )

            _add_usage_totals(
                usage_totals,
                synthesis_invocation.usage,
            )
            usage_by_stage["synthesis"] = _usage_totals(
                synthesis_invocation.usage
            )

            if synthesis_invocation.result is None:
                return FinalResponsePipelineResult(
                    status="fallback",
                    answer=_safe_fallback_answer(
                        loop_result
                    ),
                    model_invocations=(
                        invocation_audits
                    ),
                    output_rewrites=(
                        rewrite_count
                    ),
                    citation_regeneration_count=(
                        citation_regeneration_count
                    ),
                    usage=dict(usage_totals),
                    usage_by_stage=usage_by_stage,
                    finish_reason=(
                        "synthesis_failed"
                    ),
                )

            last_synthesis = _append_disclaimer(
                synthesis_invocation.result
            )

            citation_violations = _citation_scope_violations(
                request.citations,
                last_synthesis,
                allowed_document_ids=(
                    request.allowed_document_ids
                ),
                scope_snapshot=request.scope_snapshot,
            )
            if citation_violations:
                if (
                    citation_regeneration_count
                    >= 1
                ):
                    return FinalResponsePipelineResult(
                        status="fallback",
                        delivery_status="rejected",
                        answer=_safe_fallback_answer(
                            loop_result
                        ),
                        synthesis=last_synthesis,
                        model_invocations=(
                            invocation_audits
                        ),
                        output_rewrites=(
                            rewrite_count
                        ),
                        citation_regeneration_count=(
                            citation_regeneration_count
                        ),
                        usage=dict(usage_totals),
                        usage_by_stage=usage_by_stage,
                        finish_reason=(
                            "citation_scope_violation"
                        ),
                    )
                citation_regeneration_count += 1
                rewrite_instructions = (
                    "回答只能引用本次指定范围内的文档，"
                    "当前回答引用了范围外文档："
                    + "、".join(citation_violations)
                    + "。请仅基于范围内证据重写，"
                    "不要引用范围外文档，也不要用通用知识替代。"
                )
                continue

            guard_invocation = (
                await self.output_guard.guard(
                    OutputGuardRequest(
                        request_id=request.request_id,
                        run_id=request.run_id,
                        user_message=(
                            request.user_message
                        ),
                        loop_result=loop_result,
                        synthesis=last_synthesis,
                        citations=request.citations,
                        context_summary=request.context_summary,
                        completion_contract={
                            "delivery_contract": (
                                request.delivery_contract
                            ),
                            "result_reference_context": (
                                request.result_reference_context
                            ),
                            "canonical_fact_fields": (
                                request.canonical_fact_fields
                            ),
                            "known_derivation_ids": (
                                request.known_derivation_ids
                            ),
                            "known_sub_artifact_ids": (
                                request.known_sub_artifact_ids
                            ),
                        },
                        source_authority=(
                            request.source_authority
                        ),
                        requirement_observations=(
                            request.requirement_observations
                        ),
                    )
                )
            )

            last_guard = guard_invocation.result

            invocation_audits.append(
                ModelInvocationAudit(
                    stage="output_guard",
                    model=guard_invocation.model,
                    finish_reason=(
                        guard_invocation.finish_reason
                    ),
                    usage=guard_invocation.usage,
                    attempts=(
                        guard_invocation.attempts
                    ),
                    protocol_repaired=(
                        guard_invocation.protocol_repaired
                    ),
                    error=guard_invocation.error,
                )
            )

            _add_usage_totals(
                usage_totals,
                guard_invocation.usage,
            )
            usage_by_stage["output_guard"] = _usage_totals(
                guard_invocation.usage
            )

            if last_guard.verdict == "pass":
                return FinalResponsePipelineResult(
                    status="completed",
                    answer=last_synthesis.answer,
                    synthesis=last_synthesis,
                    guard=last_guard,
                    model_invocations=(
                        invocation_audits
                    ),
                    output_rewrites=(
                        rewrite_count
                    ),
                    citation_regeneration_count=(
                        citation_regeneration_count
                    ),
                    usage=dict(usage_totals),
                    usage_by_stage=usage_by_stage,
                    finish_reason=(
                        "output_guard_passed"
                    ),
                )

            if last_guard.verdict == "fallback":
                risk_flags = (
                    last_guard.risk_flags or []
                )
                transient_failure = (
                    "guard_service_unavailable" in risk_flags
                    or "guard_protocol_failure" in risk_flags
                )
                if "guard_protocol_failure" in risk_flags:
                    # Deterministic checks have already run before the LLM
                    # guard. A malformed guard protocol is infrastructure
                    # degradation, not proof that the answer is unsafe.
                    return FinalResponsePipelineResult(
                        status="completed",
                        delivery_status="guard_degraded",
                        answer=last_synthesis.answer,
                        synthesis=last_synthesis,
                        guard=last_guard,
                        model_invocations=invocation_audits,
                        output_rewrites=rewrite_count,
                        usage=dict(usage_totals),
                        usage_by_stage=usage_by_stage,
                        finish_reason="output_guard_protocol_degraded",
                    )
                if (
                    transient_failure
                    and guard_retry_count < 2
                ):
                    # Guard 瞬时失败（API/协议）时不直接兜底，
                    # 对同一版草稿重试 Guard。
                    guard_retry_count += 1
                    logger.warning(
                        "output_guard_transient_retry",
                        request_id=request.request_id,
                        run_id=request.run_id,
                        retry_count=guard_retry_count,
                        risk_flags=risk_flags,
                    )
                    continue
                return FinalResponsePipelineResult(
                    status="fallback",
                    answer=_safe_fallback_answer(
                        loop_result
                    ),
                    synthesis=last_synthesis,
                    guard=last_guard,
                    model_invocations=(
                        invocation_audits
                    ),
                    output_rewrites=(
                        rewrite_count
                    ),
                    citation_regeneration_count=(
                        citation_regeneration_count
                    ),
                    usage=dict(usage_totals),
                    usage_by_stage=usage_by_stage,
                    finish_reason=(
                        "output_guard_fallback"
                    ),
                )

            if last_guard.verdict == "rewrite":
                fingerprint = _guard_violation_fingerprint(last_guard)
                immutable_flag = any(
                    flag in (last_guard.risk_flags or [])
                    for flag in _IMMUTABLE_GUARD_FLAGS
                )
                if (
                    rewrite_count >= 1
                    and (
                        immutable_flag
                        or (
                            fingerprint
                            and fingerprint in seen_guard_fingerprints
                        )
                    )
                ):
                    # 同 blocking violation 已出现两次：全文重写无法修复，
                    # 必须优先于 max-rewrite 分支进入带限制交付，
                    # 否则会先触发 fallback 把好答案丢掉。
                    return FinalResponsePipelineResult(
                        status="completed",
                        delivery_status="validated_with_limitations",
                        answer=last_synthesis.answer,
                        synthesis=last_synthesis,
                        guard=last_guard,
                        model_invocations=(
                            invocation_audits
                        ),
                        output_rewrites=rewrite_count,
                        citation_regeneration_count=(
                            citation_regeneration_count
                        ),
                        usage=dict(usage_totals),
                        usage_by_stage=usage_by_stage,
                        finish_reason=(
                            "structural_guard_violation_persisted"
                        ),
                    )
                seen_guard_fingerprints.add(fingerprint)

            if (
                rewrite_count
                >= self.limits.max_output_rewrites
            ):
                # Guard 反复要求“可修复的重写”（而非 fallback）时，
                # 直接返回最后一版草稿，而不是用“信息不足”兜底，
                # 避免用户面对个股咨询等安全拒绝类问题得到误导性回复。
                return FinalResponsePipelineResult(
                    status="completed",
                    answer=last_synthesis.answer,
                    synthesis=last_synthesis,
                    guard=last_guard,
                    model_invocations=(
                        invocation_audits
                    ),
                    output_rewrites=(
                        rewrite_count
                    ),
                    citation_regeneration_count=(
                        citation_regeneration_count
                    ),
                    usage=dict(usage_totals),
                    usage_by_stage=usage_by_stage,
                    finish_reason=(
                        "max_output_rewrites_exceeded"
                    ),
                )

            rewrite_count += 1

            rewrite_instructions = (
                last_guard.rewrite_instructions
                or "根据输出检查结果重写回答。"
            )
