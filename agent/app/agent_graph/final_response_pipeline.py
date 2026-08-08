from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

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
                usage=dict(usage_totals),
                usage_by_stage=usage_by_stage,
            )

        if loop_result.status != "completed":
            return FinalResponsePipelineResult(
                status="fallback",
                answer=_safe_fallback_answer(
                    loop_result
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
                    usage=dict(usage_totals),
                    usage_by_stage=usage_by_stage,
                    finish_reason=(
                        "synthesis_failed"
                    ),
                )

            last_synthesis = _append_disclaimer(
                synthesis_invocation.result
            )

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
                    usage=dict(usage_totals),
                    usage_by_stage=usage_by_stage,
                    finish_reason=(
                        "output_guard_fallback"
                    ),
                )

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
