from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from app.agent_graph.llm_plan_reviewer import (
    PlanReviewInvocationResult,
    PlanReviewRequest,
)
from app.agent_graph.llm_task_planner import (
    PlannerInvocationResult,
    PlannerRequest,
)
from app.agent_graph.runtime_nodes.tool_executor_node import (
    build_tool_reuse_signature,
    execute_planner_tool_calls,
)
from app.agent_graph.runtime.agent_limits import (
    AgentLimits,
    DEFAULT_AGENT_LIMITS,
)
from app.agent_graph.runtime.error_policy import (
    build_error_signature,
)
from app.agent_graph.schemas.loop_schema import (
    AgentLoopResult,
    PlannerInvocationAudit,
    NoProgressAudit,
    PlanReviewInvocationAudit,
    ToolReuseAudit,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
)
from app.agent_graph.schemas.tool_schema import (
    ToolResult,
    ToolTraceEntry,
)
from app.core.logging import get_logger
from app.tools.tool_executor import (
    ProductionToolExecutor,
    ToolExecutionContext,
)


logger = get_logger(__name__)


class AgentPlanner(Protocol):
    async def plan(
        self,
        request: PlannerRequest,
    ) -> PlannerInvocationResult:
        ...


class AgentReviewer(Protocol):
    def should_review(
        self,
        *,
        decision: PlannerDecision,
        route_context: dict[str, Any],
        repeated_error_count: int,
        repairable_schema_error: bool = False,
    ) -> bool:
        ...

    async def review(
        self,
        request: PlanReviewRequest,
    ) -> PlanReviewInvocationResult:
        ...


@dataclass(slots=True)
class _LoopAccumulator:
    agent_messages: list[dict[str, Any]] = field(
        default_factory=list
    )

    tool_results: list[ToolResult] = field(
        default_factory=list
    )

    tool_traces: list[ToolTraceEntry] = field(
        default_factory=list
    )

    planner_invocations: list[
        PlannerInvocationAudit
    ] = field(default_factory=list)

    review_invocations: list[
        PlanReviewInvocationAudit
    ] = field(default_factory=list)

    error_counts: dict[str, int] = field(
        default_factory=dict
    )

    successful_tool_results_by_signature: dict[
        str,
        ToolResult,
    ] = field(default_factory=dict)

    reused_tool_calls: list[ToolReuseAudit] = field(
        default_factory=list
    )

    no_progress_events: list[NoProgressAudit] = field(
        default_factory=list
    )

    total_tool_calls: int = 0
    reused_tool_call_count: int = 0
    no_progress_round_count: int = 0
    consecutive_no_progress_rounds: int = 0
    max_consecutive_no_progress_rounds: int = 0
    last_progress_round: int | None = None
    repeated_error_count: int = 0
    plan_revision_count: int = 0

    last_error_signature: str | None = None
    last_repairable_schema_error: bool = False


def _fallback_decision(
    *,
    reason: str,
    plan_version: int,
) -> PlannerDecision:
    return PlannerDecision(
        action="fallback",
        decision_reason=reason,
        confidence="low",
        needs_review=False,
        plan_version=max(plan_version, 1),
    )


def _clarification_decision(
    *,
    question: str,
    reason: str,
    plan_version: int,
) -> PlannerDecision:
    return PlannerDecision(
        action="clarify",
        clarification_question=question,
        decision_reason=reason,
        confidence="medium",
        needs_review=False,
        plan_version=max(plan_version, 1),
    )


def _assistant_tool_message_for_state(
    assistant_message: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": assistant_message.get(
            "tool_calls",
            [],
        ),
    }


class AgentToolLoop:
    def __init__(
        self,
        *,
        planner: AgentPlanner,
        executor: ProductionToolExecutor,
        reviewer: AgentReviewer | None = None,
        limits: AgentLimits = DEFAULT_AGENT_LIMITS,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.reviewer = reviewer
        self.limits = limits

    async def run(
        self,
        *,
        request: PlannerRequest,
        execution_context: ToolExecutionContext,
    ) -> AgentLoopResult:
        accumulator = _LoopAccumulator(
            agent_messages=list(request.agent_messages)
        )

        try:
            async with asyncio.timeout(
                self.limits.total_run_timeout_seconds
            ):
                return await self._run_internal(
                    request=request,
                    execution_context=execution_context,
                    accumulator=accumulator,
                )

        except TimeoutError:
            logger.error(
                "agent_tool_loop_timeout",
                request_id=request.request_id,
                run_id=request.run_id,
                total_tool_calls=(
                    accumulator.total_tool_calls
                ),
                agent_rounds=len(
                    accumulator.planner_invocations
                ),
            )

            decision = _fallback_decision(
                reason="Agent 工具循环超过总运行时间限制。",
                plan_version=max(
                    len(accumulator.planner_invocations),
                    1,
                ),
            )

            return self._build_result(
                status="fallback",
                decision=decision,
                accumulator=accumulator,
                finish_reason="agent_run_timeout",
            )

    async def _run_internal(
        self,
        *,
        request: PlannerRequest,
        execution_context: ToolExecutionContext,
        accumulator: _LoopAccumulator,
    ) -> AgentLoopResult:
        initial_tool_budget = min(
            request.remaining_tool_calls,
            execution_context.remaining_tool_calls,
            self.limits.max_total_tool_calls,
        )

        active_review_feedback = (
            request.review_feedback
        )

        force_review_next_plan = False

        logger.info(
            "agent_tool_loop_started",
            request_id=request.request_id,
            run_id=request.run_id,
            initial_tool_budget=initial_tool_budget,
            max_agent_rounds=(
                self.limits.max_agent_rounds
            ),
            reviewer_enabled=(
                self.reviewer is not None
            ),
        )

        for round_index in range(
            1,
            self.limits.max_agent_rounds + 1,
        ):
            remaining_tool_calls = max(
                0,
                initial_tool_budget
                - accumulator.total_tool_calls,
            )

            round_request = replace(
                request,
                agent_messages=list(
                    accumulator.agent_messages
                ),
                agent_round=round_index,
                remaining_tool_calls=remaining_tool_calls,
                repeated_error_count=(
                    accumulator.repeated_error_count
                ),
                review_feedback=active_review_feedback,
            )

            active_review_feedback = ""

            planner_result = await self.planner.plan(
                round_request
            )

            decision = planner_result.decision

            accumulator.planner_invocations.append(
                PlannerInvocationAudit(
                    round_index=round_index,
                    action=decision.action,
                    tool_names=[
                        call.tool_name
                        for call in decision.tool_calls
                    ],
                    model=planner_result.model,
                    finish_reason=(
                        planner_result.finish_reason
                    ),
                    usage=planner_result.usage,
                    attempts=planner_result.attempts,
                    protocol_repaired=(
                        planner_result.protocol_repaired
                    ),
                    raw_tool_call_names=(
                        planner_result.raw_tool_call_names
                    ),
                    error=planner_result.error,
                )
            )

            logger.info(
                "agent_tool_loop_round_planned",
                request_id=request.request_id,
                run_id=request.run_id,
                round_index=round_index,
                action=decision.action,
                tool_names=[
                    call.tool_name
                    for call in decision.tool_calls
                ],
                remaining_tool_calls=(
                    remaining_tool_calls
                ),
            )

            if decision.action == "respond":
                return self._build_result(
                    status="completed",
                    decision=decision,
                    accumulator=accumulator,
                    finish_reason="planner_finished",
                )

            if decision.action == "clarify":
                return self._build_result(
                    status="clarification_required",
                    decision=decision,
                    accumulator=accumulator,
                    finish_reason=(
                        "planner_requested_clarification"
                    ),
                    clarification_question=(
                        decision.clarification_question
                    ),
                )

            if decision.action == "fallback":
                return self._build_result(
                    status="fallback",
                    decision=decision,
                    accumulator=accumulator,
                    finish_reason=(
                        "planner_requested_fallback"
                    ),
                )

            signature_context = replace(
                execution_context,
                remaining_tool_calls=(
                    remaining_tool_calls
                ),
            )

            planned_execution_count = sum(
                1
                for tool_call in decision.tool_calls
                if (
                    (
                        signature := (
                            build_tool_reuse_signature(
                                executor=self.executor,
                                tool_call=tool_call,
                                context=signature_context,
                            )
                        )
                    )
                    is None
                    or signature
                    not in (
                        accumulator
                        .successful_tool_results_by_signature
                    )
                )
            )

            if (
                planned_execution_count
                > remaining_tool_calls
            ):
                budget_decision = _fallback_decision(
                    reason=(
                        "Planner 请求的新增工具调用数量超过"
                        "当前剩余工具预算。"
                    ),
                    plan_version=round_index,
                )

                return self._build_result(
                    status="fallback",
                    decision=budget_decision,
                    accumulator=accumulator,
                    finish_reason=(
                        "tool_budget_exceeded"
                    ),
                )

            needs_review = decision.needs_review

            if self.reviewer is not None:
                needs_review = (
                    force_review_next_plan
                    or self.reviewer.should_review(
                        decision=decision,
                        route_context=(
                            request.route_context
                        ),
                        repeated_error_count=(
                            accumulator.repeated_error_count
                        ),
                        repairable_schema_error=(
                            accumulator.last_repairable_schema_error
                        ),
                    )
                )

            if needs_review:
                if self.reviewer is None:
                    return self._build_result(
                        status="review_required",
                        decision=decision,
                        accumulator=accumulator,
                        finish_reason=(
                            "plan_review_required"
                        ),
                    )

                review_result = (
                    await self.reviewer.review(
                        PlanReviewRequest(
                            request_id=request.request_id,
                            run_id=request.run_id,
                            user_message=(
                                request.user_message
                            ),
                            decision=decision,
                            context_summary=(
                                request.context_summary
                            ),
                            route_context=(
                                request.route_context
                            ),
                            agent_messages=list(
                                accumulator.agent_messages
                            ),
                            repeated_error_count=(
                                accumulator.repeated_error_count
                            ),
                            plan_revision_count=(
                                accumulator.plan_revision_count
                            ),
                        )
                    )
                )

                review_decision = (
                    review_result.decision
                )

                accumulator.review_invocations.append(
                    PlanReviewInvocationAudit(
                        round_index=round_index,
                        verdict=(
                            review_decision.verdict
                        ),
                        feedback=(
                            review_decision.feedback
                        ),
                        model=review_result.model,
                        finish_reason=(
                            review_result.finish_reason
                        ),
                        usage=review_result.usage,
                        attempts=review_result.attempts,
                        protocol_repaired=(
                            review_result.protocol_repaired
                        ),
                        error=review_result.error,
                    )
                )

                logger.info(
                    "agent_tool_loop_plan_reviewed",
                    request_id=request.request_id,
                    run_id=request.run_id,
                    round_index=round_index,
                    verdict=review_decision.verdict,
                    plan_revision_count=(
                        accumulator.plan_revision_count
                    ),
                )

                if review_decision.verdict == "approve":
                    force_review_next_plan = False

                elif review_decision.verdict == "revise":
                    if (
                        accumulator.plan_revision_count
                        >= self.limits.max_plan_revisions
                    ):
                        revision_decision = (
                            _fallback_decision(
                                reason=(
                                    "计划修改次数已经达到上限。"
                                ),
                                plan_version=round_index,
                            )
                        )

                        return self._build_result(
                            status="fallback",
                            decision=revision_decision,
                            accumulator=accumulator,
                            finish_reason=(
                                "max_plan_revisions_exceeded"
                            ),
                        )

                    accumulator.plan_revision_count += 1

                    active_review_feedback = (
                        review_decision.feedback
                    )

                    force_review_next_plan = True

                    continue

                elif review_decision.verdict == "clarify":
                    question = (
                        review_decision.feedback.strip()
                        or "请补充完成当前任务所需的信息。"
                    )

                    clarification_decision = (
                        _clarification_decision(
                            question=question,
                            reason=(
                                "Reviewer 判断当前计划"
                                "缺少必要用户信息。"
                            ),
                            plan_version=round_index,
                        )
                    )

                    return self._build_result(
                        status=(
                            "clarification_required"
                        ),
                        decision=clarification_decision,
                        accumulator=accumulator,
                        finish_reason=(
                            "reviewer_requested_clarification"
                        ),
                        clarification_question=question,
                    )

                else:
                    rejected_decision = (
                        _fallback_decision(
                            reason=(
                                review_decision.feedback
                                or "Reviewer 拒绝当前计划。"
                            ),
                            plan_version=round_index,
                        )
                    )

                    return self._build_result(
                        status="fallback",
                        decision=rejected_decision,
                        accumulator=accumulator,
                        finish_reason=(
                            "reviewer_rejected_plan"
                        ),
                    )

            if not planner_result.assistant_message:
                protocol_decision = _fallback_decision(
                    reason=(
                        "Planner 要求调用工具，"
                        "但没有返回对应 assistant_message。"
                    ),
                    plan_version=round_index,
                )

                return self._build_result(
                    status="fallback",
                    decision=protocol_decision,
                    accumulator=accumulator,
                    finish_reason=(
                        "missing_assistant_tool_message"
                    ),
                )

            accumulator.agent_messages.append(
                _assistant_tool_message_for_state(
                    planner_result.assistant_message
                )
            )

            current_execution_context = replace(
                execution_context,
                remaining_tool_calls=(
                    remaining_tool_calls
                ),
            )

            node_result = (
                await execute_planner_tool_calls(
                    decision=decision,
                    executor=self.executor,
                    context=current_execution_context,
                    successful_results_by_signature=(
                        accumulator
                        .successful_tool_results_by_signature
                    ),
                    round_index=round_index,
                )
            )

            accumulator.tool_results.extend(
                node_result.tool_results
            )

            accumulator.tool_traces.extend(
                node_result.tool_traces
            )

            accumulator.agent_messages.extend(
                node_result.feedback_messages
            )

            accumulator.reused_tool_calls.extend(
                node_result.reused_tool_calls
            )

            accumulator.reused_tool_call_count += (
                node_result.reused_call_count
            )

            accumulator.total_tool_calls += (
                node_result.executed_call_count
            )

            decision_calls_by_id = {
                tool_call.tool_call_id: tool_call
                for tool_call in decision.tool_calls
            }

            for tool_result in node_result.tool_results:
                if not tool_result.success:
                    continue

                original_call = decision_calls_by_id.get(
                    tool_result.tool_call_id
                )

                if original_call is None:
                    continue

                signature = (
                    build_tool_reuse_signature(
                        executor=self.executor,
                        tool_call=original_call,
                        context=current_execution_context,
                    )
                )

                if signature is not None:
                    (
                        accumulator
                        .successful_tool_results_by_signature
                        .setdefault(
                            signature,
                            tool_result,
                        )
                    )

            if node_result.reused_tool_calls:
                logger.info(
                    "agent_tool_loop_results_reused",
                    request_id=request.request_id,
                    run_id=request.run_id,
                    round_index=round_index,
                    reused_call_count=(
                        node_result.reused_call_count
                    ),
                    tool_names=[
                        audit.tool_name
                        for audit
                        in node_result.reused_tool_calls
                    ],
                )

            accumulator.last_repairable_schema_error = False
            for tool_result in reversed(node_result.tool_results):
                if tool_result.success:
                    continue
                error = tool_result.error
                if (
                    error is not None
                    and bool(
                        getattr(error, "model_repairable", False)
                    )
                    and str(getattr(error, "code", "") or "")
                    in {
                        "ARGUMENT_SCHEMA_ERROR",
                        "DOMAIN_INPUT_ERROR",
                    }
                ):
                    accumulator.last_repairable_schema_error = True
                break

            no_progress_exceeded = (
                self._record_round_progress(
                    accumulator=accumulator,
                    decision=decision,
                    current_results=(
                        node_result.tool_results
                    ),
                    executed_call_count=(
                        node_result.executed_call_count
                    ),
                    reused_call_count=(
                        node_result.reused_call_count
                    ),
                    round_index=round_index,
                )
            )

            repeated_error_exceeded = (
                self._update_error_counters(
                    accumulator=accumulator,
                    current_results=(
                        node_result.tool_results
                    ),
                )
            )

            if repeated_error_exceeded:
                repeated_decision = _fallback_decision(
                    reason=(
                        "相同工具错误重复出现，"
                        "已停止自动修复以防止死循环。"
                    ),
                    plan_version=round_index,
                )

                return self._build_result(
                    status="fallback",
                    decision=repeated_decision,
                    accumulator=accumulator,
                    finish_reason=(
                        "repeated_tool_error_limit"
                    ),
                )

            if no_progress_exceeded:
                no_progress_decision = _fallback_decision(
                    reason=(
                        "连续多轮没有产生新的工具结果或"
                        "错误诊断，已停止规划以防止空转。"
                    ),
                    plan_version=round_index,
                )

                return self._build_result(
                    status="fallback",
                    decision=no_progress_decision,
                    accumulator=accumulator,
                    finish_reason=(
                        "no_progress_loop_detected"
                    ),
                )

        max_round_decision = _fallback_decision(
            reason="Agent 已达到最大规划轮数。",
            plan_version=self.limits.max_agent_rounds,
        )

        return self._build_result(
            status="fallback",
            decision=max_round_decision,
            accumulator=accumulator,
            finish_reason=(
                "max_agent_rounds_exceeded"
            ),
        )

    def _record_round_progress(
        self,
        *,
        accumulator: _LoopAccumulator,
        decision: PlannerDecision,
        current_results: list[ToolResult],
        executed_call_count: int,
        reused_call_count: int,
        round_index: int,
    ) -> bool:
        """
        记录当前工具轮是否产生了新证据。

        新证据包括：
        - 至少一个本轮真实执行成功的工具结果；
        - 至少一个此前未出现过的结构化工具错误。

        只有复用旧结果，或只重复已有错误，都不算推进。
        返回 True 表示连续无进展轮数已经达到硬限制。
        """

        new_success_count = sum(
            1
            for result in current_results
            if result.success
        )

        new_error_signature_count = 0

        for result in current_results:
            if result.success or result.error is None:
                continue

            signature = build_error_signature(
                tool_name=result.tool_name,
                error=result.error,
            )

            if signature not in accumulator.error_counts:
                new_error_signature_count += 1

        made_progress = (
            new_success_count > 0
            or new_error_signature_count > 0
        )

        if made_progress:
            accumulator.consecutive_no_progress_rounds = 0
            accumulator.last_progress_round = round_index
            return False

        accumulator.no_progress_round_count += 1
        accumulator.consecutive_no_progress_rounds += 1
        accumulator.max_consecutive_no_progress_rounds = max(
            accumulator.max_consecutive_no_progress_rounds,
            accumulator.consecutive_no_progress_rounds,
        )

        if reused_call_count > 0 and executed_call_count == 0:
            reason = "本轮只复用了已有成功结果，没有新增工具证据。"
        elif current_results:
            reason = "本轮工具结果只包含已经出现过的错误，没有新增证据。"
        else:
            reason = "本轮没有产生新的工具执行结果或诊断信息。"

        accumulator.no_progress_events.append(
            NoProgressAudit(
                round_index=round_index,
                reason=reason,
                tool_names=[
                    call.tool_name
                    for call in decision.tool_calls
                ],
                executed_call_count=executed_call_count,
                reused_call_count=reused_call_count,
                consecutive_count=(
                    accumulator
                    .consecutive_no_progress_rounds
                ),
            )
        )

        log_fields = {
            "round_index": round_index,
            "reason": reason,
            "executed_call_count": executed_call_count,
            "reused_call_count": reused_call_count,
            "consecutive_no_progress_rounds": (
                accumulator.consecutive_no_progress_rounds
            ),
        }

        try:
            logger.warning(
                "agent_tool_loop_no_progress",
                **log_fields,
            )
        except TypeError:
            # 兼容单元测试中使用标准 logging.Logger 的替身。
            logger.warning(
                "agent_tool_loop_no_progress",
                extra=log_fields,
            )

        return (
            accumulator.consecutive_no_progress_rounds
            >= self.limits.max_consecutive_no_progress_rounds
        )

    def _update_error_counters(
        self,
        *,
        accumulator: _LoopAccumulator,
        current_results: list[ToolResult],
    ) -> bool:
        repeated_error_exceeded = False

        for tool_result in current_results:
            if tool_result.success:
                continue

            if tool_result.error is None:
                continue

            signature = build_error_signature(
                tool_name=tool_result.tool_name,
                error=tool_result.error,
            )

            new_count = (
                accumulator.error_counts.get(
                    signature,
                    0,
                )
                + 1
            )

            accumulator.error_counts[
                signature
            ] = new_count

            accumulator.last_error_signature = (
                signature
            )

            accumulator.repeated_error_count = max(
                accumulator.repeated_error_count,
                new_count,
            )

            if (
                new_count
                >= self.limits.max_same_error_count
            ):
                repeated_error_exceeded = True

        return repeated_error_exceeded

    @staticmethod
    def _build_result(
        *,
        status: str,
        decision: PlannerDecision,
        accumulator: _LoopAccumulator,
        finish_reason: str,
        clarification_question: str | None = None,
    ) -> AgentLoopResult:
        return AgentLoopResult(
            status=status,  # type: ignore[arg-type]
            final_decision=decision,
            agent_messages=list(
                accumulator.agent_messages
            ),
            tool_results=list(
                accumulator.tool_results
            ),
            tool_traces=list(
                accumulator.tool_traces
            ),
            planner_invocations=list(
                accumulator.planner_invocations
            ),
            review_invocations=list(
                accumulator.review_invocations
            ),
            agent_rounds=len(
                accumulator.planner_invocations
            ),
            total_tool_calls=(
                accumulator.total_tool_calls
            ),
            reused_tool_calls=list(
                accumulator.reused_tool_calls
            ),
            reused_tool_call_count=(
                accumulator.reused_tool_call_count
            ),
            no_progress_events=list(
                accumulator.no_progress_events
            ),
            no_progress_round_count=(
                accumulator.no_progress_round_count
            ),
            consecutive_no_progress_rounds=(
                accumulator
                .consecutive_no_progress_rounds
            ),
            max_consecutive_no_progress_rounds=(
                accumulator
                .max_consecutive_no_progress_rounds
            ),
            last_progress_round=(
                accumulator.last_progress_round
            ),
            repeated_error_count=(
                accumulator.repeated_error_count
            ),
            plan_revision_count=(
                accumulator.plan_revision_count
            ),
            last_error_signature=(
                accumulator.last_error_signature
            ),
            clarification_question=(
                clarification_question
            ),
            finish_reason=finish_reason,
        )
