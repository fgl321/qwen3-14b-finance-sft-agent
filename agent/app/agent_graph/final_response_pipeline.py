from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

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
        "当前信息不足，或系统暂时无法安全完成本次分析。"
        "请补充必要信息后重新提问。"
    )


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
            )

        invocation_audits: list[
            ModelInvocationAudit
        ] = []

        usage_totals: dict[str, int] = defaultdict(int)

        rewrite_instructions = ""
        rewrite_count = 0

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

            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                value = synthesis_invocation.usage.get(
                    key
                )

                if isinstance(value, int):
                    usage_totals[key] += value

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

            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                value = guard_invocation.usage.get(
                    key
                )

                if isinstance(value, int):
                    usage_totals[key] += value

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
                    finish_reason=(
                        "output_guard_passed"
                    ),
                )

            if last_guard.verdict == "fallback":
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
                    finish_reason=(
                        "output_guard_fallback"
                    ),
                )

            if (
                rewrite_count
                >= self.limits.max_output_rewrites
            ):
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
                    finish_reason=(
                        "max_output_rewrites_exceeded"
                    ),
                )

            rewrite_count += 1

            rewrite_instructions = (
                last_guard.rewrite_instructions
                or "根据输出检查结果重写回答。"
            )
