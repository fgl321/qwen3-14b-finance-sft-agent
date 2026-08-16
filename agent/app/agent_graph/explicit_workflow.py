from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from app.agent_graph.events import publish_event
from app.agent_graph.llm_plan_reviewer import PlanReviewRequest
from app.agent_graph.llm_task_planner import (
    PlannerRequest,
    plan_semantic_signature,
)
from app.agent_graph.llm_output_guard import OutputGuardRequest
from app.agent_graph.llm_synthesizer import SynthesisRequest
from app.agent_graph.final_response_pipeline import (
    _append_disclaimer,
    _safe_fallback_answer,
)
from app.agent_graph.production_dependencies import ProductionGraphDependencies
from app.agent_graph.production_nodes import _optional_tool_filter
from app.agent_graph.production_state import ProductionFinanceGraphState
from app.agent_graph.runtime.error_policy import build_error_signature
from app.agent_graph.runtime_nodes.tool_executor_node import (
    build_tool_reuse_signature,
    execute_planner_tool_calls,
)
from app.agent_graph.schemas.loop_schema import AgentLoopResult
from app.agent_graph.schemas.planner_schema import PlannerDecision
from app.agent_graph.schemas.tool_schema import ToolResult
from app.agent_graph.schemas.synthesis_schema import SynthesisResult
from app.tools.tool_executor import (
    ToolExecutionContext,
    source_authority_from_route_context,
)


def _trace(
    state: ProductionFinanceGraphState,
    *,
    node: str,
    started: float,
    status: str = "completed",
    summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    entry = {
        "node": node,
        "status": status,
        "elapsed_ms": max(0, int((time.perf_counter() - started) * 1000)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary or {},
    }
    publish_event(
        "node_finished",
        request_id=str(state.get("request_id") or ""),
        node=node,
        status=status,
        detail=entry["summary"],
    )
    return [*(state.get("node_trace") or []), entry]


def _start(state: ProductionFinanceGraphState, node: str) -> float:
    publish_event(
        "node_started",
        request_id=str(state.get("request_id") or ""),
        node=node,
    )
    return time.perf_counter()


def _execution_context(
    state: ProductionFinanceGraphState,
    *,
    remaining: int,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        request_id=state["request_id"],
        run_id=state["run_id"],
        tenant_id=str(state.get("tenant_id") or "personal"),
        user_id=state["user_id"],
        role="user",
        allowed_tool_names=_optional_tool_filter(state.get("allowed_tool_names")),
        allowed_tool_groups=_optional_tool_filter(state.get("allowed_tool_groups")),
        allow_side_effects=bool(state.get("allow_side_effects", False)),
        remaining_tool_calls=remaining,
        source_authority=source_authority_from_route_context(
            state.get("route_context")
        ),
    )


def _delivery_contract_from_route_context(
    route_context: dict[str, Any],
) -> str:
    """Build the per-task delivery contract from the resolved route."""

    retrieval = dict(route_context.get("retrieval_outcome") or {})
    semantic_route = dict(route_context.get("semantic_route") or {})
    coverage_by_task = {
        str(item.get("requirement_id") or ""): item
        for item in (retrieval.get("requirement_coverage") or [])
    }
    lines = [
        "<delivery_contract>",
        "每个 retrieval task 的当前状态与必须逐条交付的子要求：",
    ]
    for task in (semantic_route.get("task_requirements") or []):
        task_id = str(task.get("id") or "")
        coverage = coverage_by_task.get(task_id) or {}
        lines.append(
            "- "
            + task_id
            + ": status="
            + str(coverage.get("status") or "not_covered")
            + "; required_outputs="
            + str(task.get("required_outputs") or [])
        )
    lines.append("</delivery_contract>")
    return "\n".join(lines)


def build_intent_router_node(dependencies: ProductionGraphDependencies):
    async def intent_router(state: ProductionFinanceGraphState) -> dict[str, Any]:
        started = _start(state, "intent_router")
        groups = set(state.get("allowed_tool_groups") or [])
        route_context = dict(state.get("route_context") or {})
        semantic_route = dict(route_context.get("semantic_route") or {})
        semantic_mode = str(semantic_route.get("orchestration_mode") or "")
        if semantic_mode:
            mode = semantic_mode
        elif "knowledge_retrieval" in groups and "financial_calculation" in groups:
            mode = "hybrid"
        elif "knowledge_retrieval" in groups:
            mode = "rag"
        elif "financial_calculation" in groups:
            mode = "tool_or_direct"
        else:
            mode = "direct"
        route_context.setdefault("orchestration_mode", mode)
        route_context.setdefault("complexity", "medium")
        route_context.setdefault("risk_level", "low")
        return {
            "route_context": route_context,
            "orchestration_mode": mode,
            "node_trace": _trace(
                state,
                node="intent_router",
                started=started,
                summary={"mode": mode},
            ),
        }

    return intent_router


def build_planner_node(dependencies: ProductionGraphDependencies):
    async def planner(state: ProductionFinanceGraphState) -> dict[str, Any]:
        started = _start(state, "planner")
        limits = dependencies.limits
        is_plan_repair = bool(str(state.get("review_feedback") or "").strip())
        execution_round = int(state.get("execution_round") or 0)
        next_execution_round = execution_round + 1
        invocation_index = int(state.get("planner_invocation_count") or 0) + 1
        plan_attempt = (
            int(state.get("plan_attempt_in_round") or 1) + 1
            if is_plan_repair
            else 1
        )
        if next_execution_round > limits.max_agent_rounds:
            has_verified_results = any(
                bool(item.get("success"))
                for item in (state.get("tool_results") or [])
            )
            decision = PlannerDecision(
                action="respond" if has_verified_results else "fallback",
                decision_reason=(
                    "已达到最大规划轮数；停止继续规划，并使用现有已验证结果生成回答。"
                    if has_verified_results
                    else "已达到最大规划轮数，且没有可安全返回的已验证结果。"
                ),
                confidence="medium" if has_verified_results else "low",
                plan_version=limits.max_agent_rounds,
            )
            return {
                "current_decision": decision.model_dump(mode="json"),
                "loop_status": "completed" if has_verified_results else "fallback",
                "loop_finish_reason": (
                    "max_agent_rounds_completed_with_verified_results"
                    if has_verified_results
                    else "max_agent_rounds_exceeded_without_results"
                ),
                "node_trace": _trace(
                    state,
                    node="planner",
                    started=started,
                    status="bounded",
                    summary={"execution_round": next_execution_round},
                ),
            }

        remaining = max(
            0,
            min(
                int(state.get("remaining_tool_calls") or limits.max_total_tool_calls),
                limits.max_total_tool_calls,
            )
            - int(state.get("total_tool_calls") or 0),
        )
        route_context = dict(state.get("route_context") or {})
        semantic_route = dict(
            route_context.get("semantic_route") or {}
        )
        orchestration_mode = str(
            semantic_route.get("orchestration_mode") or ""
        )
        executable_capabilities = set(
            semantic_route.get("required_capabilities") or []
        ) & {
            "knowledge_retrieval",
            "financial_calculation",
            "resource_catalog_read",
            "memory_read",
            "web_search",
        }
        blocking_executable = (
            executable_capabilities - {"financial_calculation"}
        )
        direct_respond = bool(
            orchestration_mode == "direct"
            and not blocking_executable
        )
        if direct_respond and not is_plan_repair:
            decision = PlannerDecision(
                action="respond",
                decision_reason=(
                    "direct orchestration; planner LLM not required"
                ),
                confidence="high",
                plan_version=next_execution_round,
            )
            return {
                "current_decision": decision.model_dump(
                    mode="json"
                ),
                "loop_status": "completed",
                "loop_finish_reason": (
                    "planner_direct_skipped"
                ),
                "planner_invocations": list(
                    state.get("planner_invocations") or []
                ),
                "node_trace": _trace(
                    state,
                    node="planner",
                    started=started,
                    status="direct_skipped",
                    summary={
                        "planner_llm_invocations": 0,
                        "reason": "direct orchestration",
                    },
                ),
            }
        result = await dependencies.planner.plan(
            PlannerRequest(
                request_id=state["request_id"],
                run_id=state["run_id"],
                user_message=state["user_message"],
                history_messages=list(state.get("history_messages") or []),
                agent_messages=list(state.get("agent_messages") or []),
                context_summary=str(state.get("context_summary") or ""),
                route_context=dict(state.get("route_context") or {}),
                review_feedback=str(state.get("review_feedback") or ""),
                previous_plan=(
                    dict(state.get("current_decision") or {})
                    if is_plan_repair
                    else {}
                ),
                plan_attempt_in_round=plan_attempt,
                plan_repair_count=int(state.get("plan_repair_count") or 0),
                completed_execution_rounds=execution_round,
                target_execution_round=next_execution_round,
                replan_count=int(state.get("replan_count") or 0),
                last_execution_observation=dict(
                    state.get("last_execution_observation") or {}
                ),
                allowed_tool_names=_optional_tool_filter(state.get("allowed_tool_names")),
                allowed_tool_groups=_optional_tool_filter(state.get("allowed_tool_groups")),
                execution_policy=state.get("execution_policy", "auto"),
                agent_round=next_execution_round,
                remaining_tool_calls=remaining,
                repeated_error_count=int(state.get("repeated_error_count") or 0),
            )
        )
        decision = result.decision
        audits = [
            *(state.get("planner_invocations") or []),
            {
                "round_index": next_execution_round,
                "invocation_index": invocation_index,
                "completed_execution_rounds": execution_round,
                "target_execution_round": next_execution_round,
                "creates_execution_round": decision.action == "call_tools",
                "plan_repair": is_plan_repair,
                "plan_attempt_in_round": plan_attempt,
                "action": decision.action,
                "tool_names": [call.tool_name for call in decision.tool_calls],
                "model": result.model,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
                "attempts": result.attempts,
                "protocol_repaired": result.protocol_repaired,
                "raw_tool_call_names": result.raw_tool_call_names,
                "error": result.error,
                "plan_signature": plan_semantic_signature(decision),
                "previous_plan_signature": (
                    plan_semantic_signature(state["current_decision"])
                    if is_plan_repair and state.get("current_decision")
                    else None
                ),
            },
        ]
        return {
            "planner_round": next_execution_round,
            "planner_invocation_count": invocation_index,
            "plan_attempt_in_round": plan_attempt,
            "plan_repair_count": (
                int(state.get("plan_repair_count") or 0)
                if is_plan_repair
                else 0
            ),
            "current_decision": decision.model_dump(mode="json"),
            "current_assistant_message": result.assistant_message,
            "planner_invocations": audits,
            "review_feedback": "",
            "loop_status": (
                "completed"
                if decision.action == "respond"
                else "clarification_required"
                if decision.action == "clarify"
                else "fallback"
                if decision.action == "fallback"
                else "running"
            ),
            "loop_finish_reason": (
                "planner_finished"
                if decision.action == "respond"
                else "planner_requested_clarification"
                if decision.action == "clarify"
                else "planner_requested_fallback"
                if decision.action == "fallback"
                else ""
            ),
            "node_trace": _trace(
                state,
                node="planner",
                started=started,
                summary={
                    "execution_round": next_execution_round,
                    "completed_execution_rounds": execution_round,
                    "target_execution_round": next_execution_round,
                    "planner_invocation": invocation_index,
                    "plan_attempt": plan_attempt,
                    "plan_repair": is_plan_repair,
                    "action": decision.action,
                    "tool_names": [call.tool_name for call in decision.tool_calls],
                },
            ),
        }

    return planner


def route_after_planner(state: ProductionFinanceGraphState) -> str:
    if state.get("error"):
        return "failure"
    decision = PlannerDecision.model_validate(state.get("current_decision") or {})
    return "review" if decision.action == "call_tools" else "assemble"


def build_plan_review_node(dependencies: ProductionGraphDependencies):
    async def plan_review(state: ProductionFinanceGraphState) -> dict[str, Any]:
        started = _start(state, "plan_review")
        decision = PlannerDecision.model_validate(state["current_decision"])
        # Native tool calling marks every multi-call wave for review for
        # backward compatibility. In the semantic production route, parallel
        # low-risk read-only tools are already permission/schema checked and
        # do not need an extra LLM review solely because there are several.
        reviewer_policy = getattr(dependencies.reviewer, "policy", None)
        reviewer_registry = getattr(reviewer_policy, "registry", None)
        if (
            reviewer_registry is not None
            and (state.get("route_context") or {}).get("semantic_route")
            and len(decision.tool_calls) > 1
            and all(
                (spec := reviewer_registry.get(call.tool_name)) is not None
                and spec.risk_level != "high"
                and not spec.side_effect
                for call in decision.tool_calls
            )
        ):
            decision = decision.model_copy(update={"needs_review": False})
        should_review = dependencies.reviewer.should_review(
            decision=decision,
            route_context=dict(state.get("route_context") or {}),
            repeated_error_count=int(state.get("repeated_error_count") or 0),
        )
        if not should_review:
            return {
                "current_review": {"verdict": "approve", "feedback": ""},
                "node_trace": _trace(
                    state,
                    node="plan_review",
                    started=started,
                    summary={
                        "verdict": "approve",
                        "llm_invoked": False,
                        "target_execution_round": int(
                            state.get("execution_round") or 0
                        ) + 1,
                        "plan_attempt_in_round": int(
                            state.get("plan_attempt_in_round") or 1
                        ),
                    },
                ),
            }

        result = await dependencies.reviewer.review(
            PlanReviewRequest(
                request_id=state["request_id"],
                run_id=state["run_id"],
                user_message=state["user_message"],
                decision=decision,
                context_summary=str(state.get("context_summary") or ""),
                route_context=dict(state.get("route_context") or {}),
                agent_messages=list(state.get("agent_messages") or []),
                repeated_error_count=int(state.get("repeated_error_count") or 0),
                plan_revision_count=int(state.get("plan_revision_count") or 0),
            )
        )
        review = result.decision
        audits = [
            *(state.get("review_invocations") or []),
            {
                "round_index": int(state.get("planner_round") or 1),
                "planner_invocation_index": int(
                    state.get("planner_invocation_count") or 1
                ),
                "target_execution_round": int(
                    state.get("execution_round") or 0
                ) + 1,
                "plan_attempt_in_round": int(
                    state.get("plan_attempt_in_round") or 1
                ),
                "verdict": review.verdict,
                "feedback": review.feedback,
                "model": result.model,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
                "attempts": result.attempts,
                "protocol_repaired": result.protocol_repaired,
                "semantic_consistency_checked": (
                    result.semantic_consistency_checked
                ),
                "verdict_normalized": result.verdict_normalized,
                "plan_repair_count": (
                    int(state.get("plan_repair_count") or 0)
                    + (1 if review.verdict == "revise" else 0)
                ),
                "error": result.error,
            },
        ]
        update: dict[str, Any] = {
            "current_review": review.model_dump(mode="json"),
            "review_invocations": audits,
        }
        if review.verdict == "revise":
            repairs_in_round = int(state.get("plan_repair_count") or 0) + 1
            if (
                repairs_in_round
                > dependencies.limits.max_plan_repairs_per_execution_round
            ):
                update.update(
                    loop_status="fallback",
                    loop_finish_reason="max_plan_repairs_in_execution_round_exceeded",
                    plan_repair_count=repairs_in_round,
                )
            else:
                update.update(
                    plan_repair_count=repairs_in_round,
                    review_feedback=review.feedback,
                )
        elif review.verdict == "clarify":
            update.update(
                current_decision=PlannerDecision(
                    action="clarify",
                    clarification_question=review.feedback,
                    decision_reason="Reviewer 要求补充必要信息。",
                    plan_version=int(state.get("planner_round") or 1),
                ).model_dump(mode="json"),
                loop_status="clarification_required",
                loop_finish_reason="reviewer_requested_clarification",
            )
        elif review.verdict == "reject":
            update.update(
                loop_status="fallback",
                loop_finish_reason="reviewer_rejected_plan",
            )
        update["node_trace"] = _trace(
            state,
            node="plan_review",
            started=started,
            summary={
                "verdict": review.verdict,
                "llm_invoked": True,
                "target_execution_round": int(
                    state.get("execution_round") or 0
                ) + 1,
                "plan_attempt_in_round": int(
                    state.get("plan_attempt_in_round") or 1
                ),
                "plan_repair_count": int(
                    update.get(
                        "plan_repair_count",
                        state.get("plan_repair_count") or 0,
                    )
                ),
                "completed_execution_rounds": int(
                    state.get("execution_round") or 0
                ),
            },
        )
        return update

    return plan_review


def route_after_review(state: ProductionFinanceGraphState) -> str:
    if state.get("error"):
        return "failure"
    if state.get("loop_status") in {"fallback", "clarification_required"}:
        return "assemble"
    verdict = str((state.get("current_review") or {}).get("verdict") or "approve")
    return "execute" if verdict == "approve" else "replan" if verdict == "revise" else "assemble"


def build_tool_executor_node(dependencies: ProductionGraphDependencies):
    async def tool_executor(state: ProductionFinanceGraphState) -> dict[str, Any]:
        started = _start(state, "tool_executor")
        decision = PlannerDecision.model_validate(state["current_decision"])
        remaining = max(
            0,
            min(
                int(state.get("remaining_tool_calls") or dependencies.limits.max_total_tool_calls),
                dependencies.limits.max_total_tool_calls,
            )
            - int(state.get("total_tool_calls") or 0),
        )
        context = _execution_context(state, remaining=remaining)
        cache = {
            signature: ToolResult.model_validate(result)
            for signature, result in (state.get("successful_tool_results") or {}).items()
        }
        assistant = dict(state.get("current_assistant_message") or {})
        if not assistant:
            raise ValueError("Planner 要求调用工具但未返回 assistant tool message。")
        agent_messages = [
            *(state.get("agent_messages") or []),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant.get("tool_calls", []),
            },
        ]
        result = await execute_planner_tool_calls(
            decision=decision,
            executor=dependencies.executor,
            context=context,
            successful_results_by_signature=cache,
            round_index=int(state.get("planner_round") or 1),
        )
        decision_calls = {call.tool_call_id: call for call in decision.tool_calls}
        for tool_result in result.tool_results:
            if not tool_result.success:
                continue
            call = decision_calls.get(tool_result.tool_call_id)
            if call is None:
                continue
            signature = build_tool_reuse_signature(
                executor=dependencies.executor,
                tool_call=call,
                context=context,
            )
            if signature:
                cache.setdefault(signature, tool_result)
        cumulative_results = [
            *(state.get("tool_results") or []),
            *(item.model_dump(mode="json") for item in result.tool_results),
        ]
        return {
            "agent_messages": [*agent_messages, *result.feedback_messages],
            "tool_results": cumulative_results,
            "tool_traces": [
                *(state.get("tool_traces") or []),
                *(item.model_dump(mode="json") for item in result.tool_traces),
            ],
            "reused_tool_calls": [
                *(state.get("reused_tool_calls") or []),
                *(item.model_dump(mode="json") for item in result.reused_tool_calls),
            ],
            "successful_tool_results": {
                key: value.model_dump(mode="json") for key, value in cache.items()
            },
            "current_tool_results": [
                item.model_dump(mode="json") for item in result.tool_results
            ],
            "total_tool_calls": int(state.get("total_tool_calls") or 0)
            + result.executed_call_count,
            "reused_tool_call_count": int(state.get("reused_tool_call_count") or 0)
            + result.reused_call_count,
            "node_trace": _trace(
                state,
                node="tool_executor",
                started=started,
                summary={
                    "execution_round": int(state.get("execution_round") or 0) + 1,
                    "executed": result.executed_call_count,
                    "reused": result.reused_call_count,
                    "succeeded": result.successful_call_count,
                    "failed": result.failed_call_count,
                },
            ),
        }

    return tool_executor


def build_observation_validator_node(dependencies: ProductionGraphDependencies):
    async def observation_validator(state: ProductionFinanceGraphState) -> dict[str, Any]:
        started = _start(state, "observation_validator")
        results = [
            ToolResult.model_validate(item)
            for item in (state.get("current_tool_results") or [])
        ]
        error_counts = dict(state.get("error_counts") or {})
        new_error = False
        repeated = int(state.get("repeated_error_count") or 0)
        for result in results:
            if result.success or result.error is None:
                continue
            signature = build_error_signature(tool_name=result.tool_name, error=result.error)
            new_error = new_error or signature not in error_counts
            error_counts[signature] = error_counts.get(signature, 0) + 1
            repeated = max(repeated, error_counts[signature])
        success_count = sum(item.success for item in results)
        no_progress = int(state.get("consecutive_no_progress_rounds") or 0)
        no_progress = 0 if success_count or new_error else no_progress + 1
        update: dict[str, Any] = {
            "error_counts": error_counts,
            "repeated_error_count": repeated,
            "consecutive_no_progress_rounds": no_progress,
            "loop_status": "running",
            "loop_finish_reason": "",
        }
        # v3.1 checkpoint compatibility only. New runs always carry
        # execution_round and are governed exclusively by the completed
        # Execute -> Observe -> Validate count in capability_validator.
        if (
            "execution_round" not in state
            and int(state.get("planner_round") or 0) >= dependencies.limits.max_agent_rounds
            and success_count
        ):
            update.update(
                loop_status="completed",
                loop_finish_reason="max_agent_rounds_completed_with_verified_results",
                execution_round=dependencies.limits.max_agent_rounds - 1,
                current_decision=PlannerDecision(
                    action="respond",
                    decision_reason="max_agent_rounds_completed_with_verified_results",
                ).model_dump(mode="json"),
            )
        if repeated >= dependencies.limits.max_same_error_count:
            update.update(
                loop_status="fallback",
                loop_finish_reason="repeated_tool_error_limit",
            )
        elif no_progress >= dependencies.limits.max_consecutive_no_progress_rounds:
            update.update(
                loop_status="fallback",
                loop_finish_reason="no_progress_loop_detected",
            )
        update["node_trace"] = _trace(
            state,
            node="observation_validator",
            started=started,
            status="completed" if update["loop_status"] == "running" else "bounded",
            summary={
                "success_count": success_count,
                "failed_count": len(results) - success_count,
                "repeated_error_count": repeated,
                "target_execution_round": int(
                    state.get("execution_round") or 0
                ) + 1,
                "execution_round_committed": False,
            },
        )
        return update

    return observation_validator


def route_after_observation(state: ProductionFinanceGraphState) -> str:
    return "validate"


def build_capability_validator_node(dependencies: ProductionGraphDependencies):
    """Summarize one completed execution cycle before any replan decision."""

    async def capability_validator(
        state: ProductionFinanceGraphState,
    ) -> dict[str, Any]:
        started = _start(state, "result_validator")
        # Commit the round only here, after Execute -> Observe -> Validate.
        execution_round = int(state.get("execution_round") or 0) + 1
        current_results = [
            ToolResult.model_validate(item)
            for item in (state.get("current_tool_results") or [])
        ]
        completed_tools = [
            result.tool_name for result in current_results if result.success
        ]
        verified_results = [
            {
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "output": result.output,
            }
            for result in current_results
            if result.success
        ]
        failed_tools = [
            {
                "tool_name": result.tool_name,
                "error_code": result.error.code if result.error else None,
                "model_repairable": bool(
                    result.error and result.error.model_repairable
                ),
            }
            for result in current_results
            if not result.success
        ]
        route_context = dict(state.get("route_context") or {})
        retrieval = dict(route_context.get("retrieval_outcome") or {})
        semantic_route = dict(route_context.get("semantic_route") or {})
        required_capabilities = list(
            semantic_route.get("required_capabilities") or []
        )
        successful_total = sum(
            bool(item.get("success"))
            for item in (state.get("tool_results") or [])
        )
        retrieval_status = str(
            retrieval.get("status") or "not_observed"
        )
        coverage_by_task = {
            str(item.get("requirement_id") or ""): item
            for item in (retrieval.get("requirement_coverage") or [])
        }

        _ALLOWED_EVIDENCE_STATUSES = {
            "direct_support",
            "partial_support",
            "background_support",
        }
        _FAILED_EVIDENCE_STATUSES = {
            "assessment_protocol_failed",
            "service_failed",
        }

        def _task_evidence_ids(task: dict[str, Any]) -> list[str]:
            task_id = str(task.get("id") or "")
            requirements = list(
                task.get("evidence_requirements") or []
            )
            return [
                f"{task_id}:{index}"
                for index in range(1, len(requirements) + 1)
            ]

        def _task_retrieval_ok(task: dict[str, Any]) -> bool:
            evidence_ids = _task_evidence_ids(task)
            if evidence_ids:
                entries = [
                    coverage_by_task.get(requirement_id)
                    for requirement_id in evidence_ids
                ]
                if not any(entries):
                    # 要求逐条检索但没有子证据 observation → not_observed
                    return False
                return all(
                    bool(entry)
                    and entry.get("status")
                    in _ALLOWED_EVIDENCE_STATUSES
                    for entry in entries
                )
            item = coverage_by_task.get(
                str(task.get("id") or "")
            )
            if item is None:
                return retrieval_status in {
                    "completed_with_evidence",
                    "completed_with_partial_evidence",
                    "completed_no_evidence",
                }
            return bool(
                item.get("status") in _ALLOWED_EVIDENCE_STATUSES
            )

        def _task_retrieval_failed(task: dict[str, Any]) -> bool:
            evidence_ids = _task_evidence_ids(task)
            if evidence_ids:
                entries = [
                    coverage_by_task.get(requirement_id)
                    for requirement_id in evidence_ids
                ]
                return bool(entries) and any(
                    bool(entry)
                    and entry.get("status")
                    in _FAILED_EVIDENCE_STATUSES
                    for entry in entries
                )
            item = coverage_by_task.get(
                str(task.get("id") or "")
            )
            if item is None:
                return False
            return bool(
                item.get("status") in _FAILED_EVIDENCE_STATUSES
            )

        def _task_has_partial(task: dict[str, Any]) -> bool:
            evidence_ids = _task_evidence_ids(task)
            if evidence_ids:
                return any(
                    (
                        coverage_by_task.get(requirement_id)
                        or {}
                    ).get("status")
                    == "partial_support"
                    for requirement_id in evidence_ids
                )
            return (
                (
                    coverage_by_task.get(
                        str(task.get("id") or "")
                    )
                    or {}
                ).get("status")
                == "partial_support"
            ) or retrieval_status == "completed_with_partial_evidence"

        capability_status: dict[str, str] = {}
        remaining_work_hints: list[str] = []
        actionable_work_hints: list[str] = []
        task_status: dict[str, dict[str, Any]] = {}
        successful_tool_names = {
            str(item.get("tool_name") or "")
            for item in (state.get("tool_results") or [])
            if bool(item.get("success"))
        }
        required_tasks = list(semantic_route.get("task_requirements") or [])
        for task in required_tasks:
            if not bool(task.get("required", True)):
                continue
            task_id = str(task.get("id") or "required_task")
            evidence_tools = {
                str(name)
                for name in (task.get("evidence_tool_names") or [])
                if str(name)
            }
            requires_citations = bool(task.get("requires_citations"))
            task_kind = str(task.get("task_kind") or "reasoning")
            missing_tools = sorted(evidence_tools - successful_tool_names)
            retrieval_ready = _task_retrieval_ok(task)
            retrieval_failed = _task_retrieval_failed(task)
            if missing_tools:
                status = "missing_tool_evidence"
            elif retrieval_failed:
                status = "retrieval_protocol_failed"
            elif task_kind == "retrieval" and requires_citations and not retrieval_ready:
                status = "missing_retrieval_evidence"
            elif evidence_tools or requires_citations:
                status = "completed"
            else:
                status = "delivery_pending" if task_kind in {"synthesis", "validation"} else "completed"
            task_status[task_id] = {
                "status": status,
                "evidence_tool_names": sorted(evidence_tools),
                "completed_tool_names": sorted(
                    evidence_tools & successful_tool_names
                ),
                "missing_tool_names": missing_tools,
                "requires_citations": requires_citations,
                "task_kind": task_kind,
                "contract_phase": (
                    "delivery" if task_kind in {"synthesis", "validation"}
                    else "execution"
                ),
            }
            remaining_work_hints.extend(
                f"required_task:{task_id}:missing_tool:{name}"
                for name in missing_tools
            )
            actionable_work_hints.extend(
                f"required_task:{task_id}:missing_tool:{name}"
                for name in missing_tools
            )
            if status == "retrieval_protocol_failed":
                remaining_work_hints.append(
                    f"required_task:{task_id}:retrieval_protocol_failed"
                )
            elif status == "missing_retrieval_evidence":
                remaining_work_hints.append(
                    f"required_task:{task_id}:missing_retrieval_evidence"
                )
        for capability in required_capabilities:
            if capability in {"knowledge_retrieval", "citation_validation"}:
                eligible_tasks = [
                    task
                    for task in required_tasks
                    if bool(task.get("required", True))
                    and (
                        capability
                        in (task.get("capabilities") or [])
                        if capability == "knowledge_retrieval"
                        else bool(task.get("requires_citations"))
                    )
                ]
                if not eligible_tasks:
                    status = "not_observed"
                else:
                    satisfied_ids = [
                        str(task.get("id") or "")
                        for task in eligible_tasks
                        if _task_retrieval_ok(
                            task
                        )
                    ]
                    failed_ids = [
                        str(task.get("id") or "")
                        for task in eligible_tasks
                        if _task_retrieval_failed(
                            task
                        )
                    ]
                    if failed_ids and satisfied_ids:
                        status = "partial"
                    elif failed_ids and not satisfied_ids:
                        status = "failed_technical"
                    elif satisfied_ids and any(
                        _task_has_partial(task)
                        for task in eligible_tasks
                    ):
                        status = "partial_evidence"
                    elif (
                        satisfied_ids
                        and retrieval_status
                        == "completed_with_partial_evidence"
                    ):
                        status = "partial_evidence"
                    elif satisfied_ids:
                        status = "verified"
                    else:
                        status = "not_observed"
                capability_status[capability] = status
                # partial_evidence / evidence_insufficient are normal
                # completed outcomes (the RAG ran and the answer must disclose
                # the support level).  Only technical failure or missing
                # execution is remaining work.
                if status in {"failed_technical", "not_observed"}:
                    remaining_work_hints.append(
                        f"{capability}:{status}"
                    )
                elif status == "partial":
                    remaining_work_hints.append(
                        f"{capability}:partial_protocol_failure"
                    )
            elif capability == "financial_calculation":
                financial_tasks = [
                    task
                    for task in required_tasks
                    if bool(task.get("required", True))
                    and capability in (task.get("capabilities") or [])
                ]
                incomplete_financial_tasks = [
                    str(task.get("id") or "required_task")
                    for task in financial_tasks
                    if task_status.get(
                        str(task.get("id") or "required_task"), {}
                    ).get("status")
                    not in {"completed", "synthesis_required"}
                ]
                capability_status[capability] = (
                    "verified"
                    if financial_tasks and not incomplete_financial_tasks
                    else "partial"
                    if successful_total
                    else "no_verified_result"
                )
                if incomplete_financial_tasks:
                    remaining_work_hints.extend(
                        f"financial_calculation:incomplete_task:{task_id}"
                        for task_id in incomplete_financial_tasks
                    )
                elif not successful_total and financial_tasks:
                    remaining_work_hints.append(
                        "financial_calculation:no_verified_result"
                    )
            else:
                capability_status[capability] = "requires_final_synthesis"
        for requirement_id in (
            retrieval.get("missing_retrieval_requirements") or []
        ):
            remaining_work_hints.append(
                f"knowledge_retrieval:missing_requirement:{requirement_id}"
            )
        for item in failed_tools:
            hint = (
                f"retry_or_replace_tool:{item['tool_name']}:"
                f"{item['error_code']}"
            )
            remaining_work_hints.append(hint)
            if item["model_repairable"]:
                actionable_work_hints.append(hint)
        observation = {
            "execution_round": execution_round,
            "completed_tools": completed_tools,
            "verified_results": verified_results,
            "failed_tools": failed_tools,
            "retrieval_outcome": retrieval,
            "required_tasks": required_tasks,
            "task_status": task_status,
            "required_capabilities": required_capabilities,
            "capability_status": capability_status,
            "remaining_work_hints": remaining_work_hints,
            "actionable_work_hints": actionable_work_hints,
            "recovery_decision": {},
            "total_successful_tool_results": successful_total,
            "remaining_tool_call_budget": max(
                0,
                int(state.get("remaining_tool_calls") or 0)
                - int(state.get("total_tool_calls") or 0),
            ),
        }
        history = [
            *(state.get("execution_round_history") or []),
            observation,
        ]
        loop_status = str(state.get("loop_status") or "running")
        update: dict[str, Any] = {
            "execution_round": execution_round,
            "last_execution_observation": observation,
            "execution_round_history": history,
        }
        has_verified_results = successful_total > 0
        needs_replan = bool(actionable_work_hints)
        remaining_budget = int(observation["remaining_tool_call_budget"])

        if loop_status == "running" and not needs_replan:
            fully_satisfied = not remaining_work_hints
            update.update(
                loop_status="completed",
                loop_finish_reason=(
                    "result_validation_completed"
                    if fully_satisfied
                    else "result_validation_partial"
                ),
                current_decision=PlannerDecision(
                    action="respond",
                    decision_reason=(
                        "本轮 Execute → Observe → Result Validate 已完成。"
                        + (
                            "所需运行时能力已有可验证结果。"
                            if fully_satisfied
                            else "存在当前执行链无法继续修复的未满足要求；"
                            "仅基于已验证结果回答并明确披露缺口。"
                        )
                    ),
                    confidence="high" if fully_satisfied else "medium",
                    plan_version=max(1, execution_round),
                ).model_dump(mode="json"),
            )
        elif loop_status == "running" and (
            execution_round >= dependencies.limits.max_agent_rounds
            or remaining_budget <= 0
        ):
            update.update(
                loop_status="completed" if has_verified_results else "fallback",
                loop_finish_reason=(
                    "max_agent_rounds_completed_with_verified_results"
                    if execution_round >= dependencies.limits.max_agent_rounds
                    and has_verified_results
                    else "max_agent_rounds_exceeded_without_results"
                    if execution_round >= dependencies.limits.max_agent_rounds
                    else "tool_budget_completed_with_verified_results"
                    if has_verified_results
                    else "tool_budget_exhausted_without_results"
                ),
            )
            if has_verified_results:
                update["current_decision"] = PlannerDecision(
                    action="respond",
                    decision_reason=(
                        "执行预算已到边界；使用现有已验证结果回答，"
                        "并披露未满足项。"
                    ),
                    confidence="medium",
                    plan_version=max(1, execution_round),
                ).model_dump(mode="json")
        elif loop_status == "running" and needs_replan:
            update["replan_count"] = int(state.get("replan_count") or 0) + 1
        recovery_status = (
            "completed" if not remaining_work_hints
            else "incomplete_retryable" if needs_replan
            else "incomplete_nonretryable"
        )
        recovery_action = (
            "replan" if needs_replan and execution_round < dependencies.limits.max_agent_rounds
            else "assemble" if recovery_status == "completed"
            else "assemble_partial"
        )
        observation["recovery_decision"] = {
            "status": recovery_status,
            "retry_scope": "planner" if needs_replan else "none",
            "next_action": recovery_action,
            "reason_codes": list(dict.fromkeys(actionable_work_hints or remaining_work_hints)),
        }
        update["node_trace"] = _trace(
            state,
            node="result_validator",
            started=started,
            status=(
                "completed"
                if update.get("loop_status", loop_status) == "completed"
                else "bounded"
                if update.get("loop_status", loop_status) == "fallback"
                else "completed"
            ),
            summary={
                "execution_round": execution_round,
                "completed_tools": completed_tools,
                "failed_tool_count": len(failed_tools),
                "retrieval_status": retrieval_status,
                "remaining_work_hint_count": len(remaining_work_hints),
                "completed_execution_rounds": execution_round,
                "next_action": recovery_action,
                "recovery_status": recovery_status,
            },
        )
        return update

    return capability_validator


def route_after_capability_validation(
    state: ProductionFinanceGraphState,
) -> str:
    return "replan" if state.get("loop_status") == "running" else "assemble"


def build_agent_result_node(dependencies: ProductionGraphDependencies):
    async def assemble(state: ProductionFinanceGraphState) -> dict[str, Any]:
        started = _start(state, "agent_result_assembler")
        decision = PlannerDecision.model_validate(state["current_decision"])
        status = str(state.get("loop_status") or "fallback")
        result = AgentLoopResult(
            status=status,
            final_decision=decision,
            agent_messages=list(state.get("agent_messages") or []),
            tool_results=list(state.get("tool_results") or []),
            tool_traces=list(state.get("tool_traces") or []),
            planner_invocations=list(state.get("planner_invocations") or []),
            review_invocations=list(state.get("review_invocations") or []),
            # Planner invocations and repairs never count as execution rounds.
            agent_rounds=int(state.get("execution_round") or 0),
            execution_round=int(state.get("execution_round") or 0),
            completed_execution_rounds=int(
                state.get("execution_round") or 0
            ),
            target_execution_round=max(
                1,
                int(state.get("execution_round") or 0)
                + (1 if state.get("loop_status") == "running" else 0),
            ),
            planner_invocation_count=int(
                state.get("planner_invocation_count") or 0
            ),
            plan_attempt_in_round=int(
                state.get("plan_attempt_in_round") or 0
            ),
            plan_repair_count=int(
                state.get("plan_repair_count") or 0
            ),
            replan_count=int(state.get("replan_count") or 0),
            execution_round_history=list(
                state.get("execution_round_history") or []
            ),
            total_tool_calls=int(state.get("total_tool_calls") or 0),
            reused_tool_calls=list(state.get("reused_tool_calls") or []),
            reused_tool_call_count=int(state.get("reused_tool_call_count") or 0),
            consecutive_no_progress_rounds=int(
                state.get("consecutive_no_progress_rounds") or 0
            ),
            max_consecutive_no_progress_rounds=int(
                state.get("consecutive_no_progress_rounds") or 0
            ),
            repeated_error_count=int(state.get("repeated_error_count") or 0),
            plan_revision_count=int(state.get("plan_revision_count") or 0),
            clarification_question=decision.clarification_question,
            finish_reason=str(state.get("loop_finish_reason") or "agent_finished"),
        )
        return {
            "agent_loop_result": result.model_dump(mode="json"),
            "status": result.status,
            "finish_reason": result.finish_reason,
            "node_trace": _trace(
                state,
                node="agent_result_assembler",
                started=started,
                summary={
                    "status": result.status,
                    "completed_execution_rounds": result.agent_rounds,
                    "planner_invocation_count": result.planner_invocation_count,
                    "replan_count": result.replan_count,
                    "tool_result_count": len(result.tool_results),
                },
            ),
        }

    return assemble


def _usage_totals(state: ProductionFinanceGraphState) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    invocations = [
        *(state.get("planner_invocations") or []),
        *(state.get("review_invocations") or []),
        *(state.get("model_invocations") or []),
    ]
    for invocation in invocations:
        usage = invocation.get("usage") or {}
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return {key: value for key, value in totals.items() if value}


def build_answer_synthesis_node(dependencies: ProductionGraphDependencies):
    async def answer_synthesis(state: ProductionFinanceGraphState) -> dict[str, Any]:
        started = _start(state, "answer_synthesis")
        loop_result = AgentLoopResult.model_validate(state["agent_loop_result"])
        if loop_result.status == "clarification_required":
            answer = loop_result.clarification_question or "请补充完成分析所需的信息。"
            return {
                "status": "clarification_required",
                "final_answer": answer,
                "finish_reason": "clarification_required",
                "guard_action": "done",
                "node_trace": _trace(
                    state,
                    node="answer_synthesis",
                    started=started,
                    status="skipped",
                    summary={"reason": "clarification"},
                ),
            }
        if loop_result.status != "completed":
            return {
                "status": "fallback",
                "final_answer": _safe_fallback_answer(loop_result),
                "finish_reason": f"loop_{loop_result.finish_reason}",
                "guard_action": "done",
                "node_trace": _trace(
                    state,
                    node="answer_synthesis",
                    started=started,
                    status="skipped",
                    summary={"reason": "loop_not_completed"},
                ),
            }
        invocation = await dependencies.synthesizer.synthesize(
            SynthesisRequest(
                request_id=state["request_id"],
                run_id=state["run_id"],
                user_message=state["user_message"],
                loop_result=loop_result,
                context_summary=str(state.get("context_summary") or ""),
                citations=list(state.get("citations") or []),
                rewrite_instructions=str(state.get("rewrite_instructions") or ""),
                source_authority=(
                    source_authority_from_route_context(
                        state.get("route_context")
                    )
                ),
                delivery_contract=(
                    _delivery_contract_from_route_context(
                        dict(state.get("route_context") or {})
                    )
                ),
                requirement_observations=list(
                    (
                        dict(
                            (state.get("route_context") or {}).get(
                                "retrieval_outcome"
                            )
                            or {}
                        ).get("requirement_coverage")
                        or []
                    )
                ),
            )
        )
        audit = {
            "stage": "synthesis",
            "model": invocation.model,
            "finish_reason": invocation.finish_reason,
            "usage": invocation.usage,
            "attempts": invocation.attempts,
            "protocol_repaired": invocation.protocol_repaired,
            "error": invocation.error,
        }
        invocations = [*(state.get("model_invocations") or []), audit]
        usage_by_node = dict(state.get("usage_by_node") or {})
        usage_by_node["synthesis"] = invocation.usage
        if invocation.result is None:
            return {
                "status": "fallback",
                "final_answer": _safe_fallback_answer(loop_result),
                "finish_reason": "synthesis_failed",
                "guard_action": "done",
                "model_invocations": invocations,
                "usage_by_node": usage_by_node,
                "usage": _usage_totals({**state, "model_invocations": invocations}),
                "node_trace": _trace(
                    state,
                    node="answer_synthesis",
                    started=started,
                    status="failed",
                    summary={"model": invocation.model},
                ),
            }
        synthesis = _append_disclaimer(invocation.result)
        return {
            "synthesis_result": synthesis.model_dump(mode="json"),
            "status": "running",
            "guard_action": "guard",
            "model_invocations": invocations,
            "usage_by_node": usage_by_node,
            "node_trace": _trace(
                state,
                node="answer_synthesis",
                started=started,
                summary={
                    "model": invocation.model,
                    "rewrite": int(state.get("output_rewrite_count") or 0),
                },
            ),
        }

    return answer_synthesis


def route_after_synthesis(state: ProductionFinanceGraphState) -> str:
    return "guard" if state.get("guard_action") == "guard" else "done"


def build_output_guard_node(dependencies: ProductionGraphDependencies):
    async def output_guard(state: ProductionFinanceGraphState) -> dict[str, Any]:
        started = _start(state, "output_guard")
        loop_result = AgentLoopResult.model_validate(state["agent_loop_result"])
        synthesis = SynthesisResult.model_validate(state["synthesis_result"])
        semantic_route_data = dict(
            dict(state.get("route_context") or {}).get(
                "semantic_route"
            )
            or {}
        )
        invocation = await dependencies.output_guard.guard(
            OutputGuardRequest(
                request_id=state["request_id"],
                run_id=state["run_id"],
                user_message=state["user_message"],
                loop_result=loop_result,
                synthesis=synthesis,
                citations=list(state.get("citations") or []),
                context_summary=str(state.get("context_summary") or ""),
                source_authority=(
                    source_authority_from_route_context(
                        state.get("route_context")
                    )
                ),
                requirement_observations=list(
                    (
                        dict(
                            (state.get("route_context") or {}).get(
                                "retrieval_outcome"
                            )
                            or {}
                        ).get("requirement_coverage")
                        or []
                    )
                ),
                completion_contract={
                    "last_execution_observation": dict(
                        state.get("last_execution_observation") or {}
                    ),
                    "execution_round_history": list(
                        state.get("execution_round_history") or []
                    ),
                    "delivery_contract": (
                        _delivery_contract_from_route_context(
                            dict(state.get("route_context") or {})
                        )
                    ),
                    "result_reference_context": {
                        "resolved_handles": [
                            str(item.get("handle") or "")
                            for item in (
                                dict(
                                    state.get("route_context") or {}
                                ).get(
                                    "resolved_result_artifacts"
                                )
                                or []
                            )
                        ],
                        "has_claims": any(
                            bool(item.get("claims") or [])
                            for item in (
                                dict(
                                    state.get("route_context") or {}
                                ).get(
                                    "resolved_result_artifacts"
                                )
                                or []
                            )
                        ),
                        "has_citations": any(
                            bool(item.get("citations") or [])
                            for item in (
                                dict(
                                    state.get("route_context") or {}
                                ).get(
                                    "resolved_result_artifacts"
                                )
                                or []
                            )
                        ),
                        "has_mutation_intent": bool(
                            semantic_route_data.get(
                                "state_update_only"
                            )
                            or semantic_route_data.get(
                                "fact_updates"
                            )
                            or semantic_route_data.get(
                                "constraint_updates"
                            )
                        ),
                        "committed_fact_fields": [
                            str(fact.get("field") or "")
                            for fact in (
                                dict(
                                    dict(
                                        state.get("route_context") or {}
                                    ).get(
                                        "effective_task_contract"
                                    )
                                    or {}
                                ).get("canonical_facts")
                                or []
                            )
                        ],
                        "current_turn_mutation_fields": list(
                            dict.fromkeys(
                                [
                                    str(
                                        patch.get("field")
                                        or ""
                                    )
                                    for patch in (
                                        (
                                            semantic_route_data.get(
                                                "fact_updates"
                                            )
                                            or []
                                        )
                                        + (
                                            semantic_route_data.get(
                                                "extracted_facts"
                                            )
                                            or []
                                        )
                                    )
                                    if str(
                                        patch.get("field")
                                        or ""
                                    )
                                    ]
                                )
                            ),
                        "technical_failures": list(
                            dict(
                                state.get("route_context") or {}
                            ).get("technical_failures")
                            or []
                        ),
                        "requirement_observations": list(
                            (
                                dict(
                                    state.get("route_context") or {}
                                ).get("retrieval_outcome")
                                or {}
                            ).get("requirement_coverage")
                            or []
                        ),
                    },
                    "canonical_fact_fields": [
                        str(fact.get("field") or "")
                        for fact in (
                            dict(
                                state.get("route_context") or {}
                            ).get(
                                "effective_task_contract", {}
                            )
                            or {}
                        ).get("canonical_facts")
                        or []
                    ],
                    "known_derivation_ids": [
                        str(calc.get("handle") or "")
                        for artifact in (
                            dict(
                                state.get("route_context") or {}
                            ).get("resolved_result_artifacts")
                            or []
                        )
                        for calc in (
                            artifact.get("calculations") or []
                        )
                    ],
                    "known_sub_artifact_ids": [
                        f"{str(artifact.get('handle') or '')}."
                        f"{str(sub) or ''}"
                        for artifact in (
                            dict(
                                state.get("route_context") or {}
                            ).get("resolved_result_artifacts")
                            or []
                        )
                        for sub in (
                            artifact.get(
                                "sub_artifact_handles"
                            )
                            or []
                        )
                    ],
                },
            )
        )
        guard = invocation.result
        audit = {
            "stage": "output_guard",
            "model": invocation.model,
            "finish_reason": invocation.finish_reason,
            "usage": invocation.usage,
            "attempts": invocation.attempts,
            "protocol_repaired": invocation.protocol_repaired,
            "error": invocation.error,
        }
        invocations = [*(state.get("model_invocations") or []), audit]
        usage_by_node = dict(state.get("usage_by_node") or {})
        usage_by_node["output_guard"] = invocation.usage
        update: dict[str, Any] = {
            "output_guard_result": guard.model_dump(mode="json"),
            "model_invocations": invocations,
            "usage_by_node": usage_by_node,
            "usage": _usage_totals({**state, "model_invocations": invocations}),
            "guard_action": "done",
        }
        if guard.verdict == "pass":
            update.update(
                status="completed",
                final_answer=synthesis.answer,
                finish_reason="output_guard_passed",
            )
        elif guard.verdict == "rewrite":
            rewrites = int(state.get("output_rewrite_count") or 0)
            flags = sorted(str(flag) for flag in (guard.risk_flags or []))
            remaining = sorted(
                str(item)
                for item in (
                    (state.get("last_execution_observation") or {}).get(
                        "remaining_work_hints"
                    )
                    or []
                )
            )
            signature_payload = {"flags": flags, "remaining": remaining}
            fingerprint = hashlib.sha256(
                json.dumps(signature_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            prior_fingerprints = list(
                state.get("guard_violation_fingerprints") or []
            )
            non_blocking_markers = (
                "evidence", "citation", "requirement", "conflict",
                "provenance", "completeness",
            )
            non_blocking = bool(flags or remaining) and all(
                any(marker in item.lower() for marker in non_blocking_markers)
                for item in [*flags, *remaining]
            )
            if fingerprint in prior_fingerprints and rewrites >= 1 and non_blocking:
                update.update(
                    status="completed",
                    final_answer=synthesis.answer,
                    finish_reason="output_guard_non_converging_upstream_constraint",
                    guard_action="done",
                    guard_violation_fingerprints=prior_fingerprints,
                )
            elif rewrites < dependencies.limits.max_output_rewrites:
                update.update(
                    guard_action="rewrite",
                    output_rewrite_count=rewrites + 1,
                    guard_violation_fingerprints=[
                        *prior_fingerprints,
                        fingerprint,
                    ],
                    rewrite_instructions=(
                        guard.rewrite_instructions or "根据输出检查结果重写回答。"
                    ),
                )
            else:
                _SOURCE_AUTHORITY_FLAGS = {
                    "source_authority_general_knowledge_used",
                    "source_authority_ungrounded_explanation",
                    "source_authority_ungrounded_answer",
                    "source_authority_result_reference_not_cited",
                }
                if set(flags) & _SOURCE_AUTHORITY_FLAGS:
                    update.update(
                        status="completed",
                        final_answer=(
                            "在当前来源约束下（禁止通用模型知识与经验法则），"
                            "本轮没有可引用的文档证据或已验证工具结果，"
                            "无法给出进一步的定性解释；"
                            "如需继续，请提供文档依据或允许使用的来源。"
                        ),
                        finish_reason=(
                            "source_authority_blocked_with_disclosure"
                        ),
                        guard_action="done",
                    )
                else:
                    update.update(
                        status=(
                            "fallback"
                            if not non_blocking
                            else "completed"
                        ),
                        final_answer=(
                            synthesis.answer
                            if non_blocking
                            else _safe_fallback_answer(loop_result)
                        ),
                        finish_reason=(
                            "max_output_rewrites_non_blocking_limit"
                            if non_blocking
                            else "max_output_rewrites_blocking_violation"
                        ),
                    )
        else:
            risk_flags = set(guard.risk_flags or [])
            if "guard_protocol_failure" in risk_flags:
                # LLMOutputGuard already performed its single schema repair.
                # A malformed guard response is infrastructure degradation,
                # not evidence that the answer content is unsafe. Preserve the
                # synthesis that passed deterministic provenance checks and do
                # not spend execution rounds or repeat the whole guard call.
                update.update(
                    status="completed",
                    final_answer=synthesis.answer,
                    finish_reason="output_guard_protocol_degraded",
                    guard_action="done",
                )
                transient = False
            else:
                transient = "guard_service_unavailable" in risk_flags
            retries = int(state.get("guard_retry_count") or 0)
            if "guard_protocol_failure" in risk_flags:
                pass
            elif transient and retries < 1:
                update.update(guard_action="retry", guard_retry_count=retries + 1)
            else:
                update.update(
                    status="fallback",
                    final_answer=_safe_fallback_answer(loop_result),
                    finish_reason="output_guard_fallback",
                )
        update["final_response_result"] = {
            "status": update.get("status", state.get("status")),
            "answer": update.get("final_answer", state.get("final_answer", "")),
            "finish_reason": update.get("finish_reason", state.get("finish_reason", "")),
            "synthesis": synthesis.model_dump(mode="json"),
            "guard": guard.model_dump(mode="json"),
            "model_invocations": invocations,
            "output_rewrites": update.get(
                "output_rewrite_count", state.get("output_rewrite_count", 0)
            ),
            "usage": update["usage"],
            "usage_by_stage": usage_by_node,
        }
        update["node_trace"] = _trace(
            state,
            node="output_guard",
            started=started,
            status="completed" if update["guard_action"] == "done" else "retrying",
            summary={"verdict": guard.verdict, "action": update["guard_action"]},
        )
        return update

    return output_guard


def route_after_output_guard(state: ProductionFinanceGraphState) -> str:
    action = str(state.get("guard_action") or "done")
    return action if action in {"rewrite", "retry"} else "done"


def record_trace_node(state: ProductionFinanceGraphState) -> dict[str, Any]:
    started = _start(state, "trace_finalizer")
    return {
        "node_trace": _trace(
            state,
            node="trace_finalizer",
            started=started,
            summary={
                "status": state.get("status"),
                "finish_reason": state.get("finish_reason"),
            },
        )
    }
