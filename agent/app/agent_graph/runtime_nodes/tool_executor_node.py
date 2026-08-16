from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent_graph.llm_task_planner import (
    build_tool_feedback_message,
)
from app.agent_graph.schemas.loop_schema import (
    ToolReuseAudit,
)
from app.agent_graph.schemas.planner_schema import (
    PlannerDecision,
    ToolCallRequest,
    resolve_typed_references,
)
from app.agent_graph.schemas.tool_schema import (
    ToolErrorInfo,
    ToolResult,
    ToolTraceEntry,
)
from app.tools.tool_executor import (
    ProductionToolExecutor,
    ToolExecutionContext,
)


class ToolExecutorNodeResult(BaseModel):
    """
    工具执行节点的结构化结果。

    feedback_messages 会在下一轮 Planner 调用时，
    作为 role=tool 消息回传给 DeepSeek。
    """

    model_config = ConfigDict(extra="forbid")

    tool_results: list[ToolResult] = Field(
        default_factory=list
    )

    tool_traces: list[ToolTraceEntry] = Field(
        default_factory=list
    )

    feedback_messages: list[dict[str, Any]] = Field(
        default_factory=list
    )

    reused_tool_calls: list[ToolReuseAudit] = Field(
        default_factory=list
    )

    executed_call_count: int = Field(default=0, ge=0)

    reused_call_count: int = Field(default=0, ge=0)

    successful_call_count: int = Field(default=0, ge=0)

    failed_call_count: int = Field(default=0, ge=0)


def build_tool_reuse_signature(
    *,
    executor: ProductionToolExecutor,
    tool_call: ToolCallRequest,
    context: ToolExecutionContext,
) -> str | None:
    """
    兼容旧测试替身。

    正式 ProductionToolExecutor 提供 build_reuse_signature；
    旧 Fake Executor 没有该方法时，视为不可复用，
    继续走原有真实执行路径。
    """

    signature_builder = getattr(
        executor,
        "build_reuse_signature",
        None,
    )

    if not callable(signature_builder):
        return None

    return signature_builder(
        tool_call,
        context=context,
    )


def _build_reused_feedback_message(
    *,
    requested_call: ToolCallRequest,
    original_result: ToolResult,
) -> dict[str, Any]:
    """
    为 Planner 本轮新产生的 tool_call_id 构造合法 role=tool 反馈。

    原始成功结果仍保留原 tool_call_id；
    这里只复制输出并明确标记结果来自复用。
    """

    feedback_result = original_result.model_copy(
        update={
            "tool_call_id": requested_call.tool_call_id,
        }
    )

    message = build_tool_feedback_message(feedback_result)

    raw_content = message.get("content")

    if isinstance(raw_content, str):
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    payload["reused"] = True
    payload["reused_from_tool_call_id"] = (
        original_result.tool_call_id
    )

    message["content"] = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return message


async def execute_planner_tool_calls(
    *,
    decision: PlannerDecision,
    executor: ProductionToolExecutor,
    context: ToolExecutionContext,
    successful_results_by_signature: Mapping[
        str,
        ToolResult,
    ] | None = None,
    round_index: int = 1,
) -> ToolExecutorNodeResult:
    """
    执行 PlannerDecision 中的业务工具调用。

    已经在本次 Agent 运行中成功完成、参数等价且允许复用的
    工具调用不会再次执行；节点会为当前新的 tool_call_id
    返回一条带复用标记的 role=tool 消息。

    该节点不重新理解用户自然语言，
    也不修改 Planner 生成的工具参数。
    """

    if decision.action != "call_tools":
        raise ValueError(
            "只有 action=call_tools 的决定才能进入工具执行节点。"
        )

    if not decision.tool_calls:
        raise ValueError(
            "工具执行节点没有收到任何工具调用。"
        )

    if any(call.depends_on for call in decision.tool_calls):
        pending = {call.effective_step_id: call for call in decision.tool_calls}
        outputs: dict[str, Any] = {}
        all_results: list[ToolResult] = []
        all_traces: list[ToolTraceEntry] = []
        all_feedback: list[dict[str, Any]] = []
        all_reuse: list[ToolReuseAudit] = []
        failed_steps: set[str] = set()
        executed = reused = succeeded = failed = 0
        while pending:
            blocked = [
                call
                for call in pending.values()
                if any(dependency in failed_steps for dependency in call.depends_on)
            ]
            for call in blocked:
                dependency_error = ToolResult(
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    success=False,
                    error=ToolErrorInfo(
                        code="DEPENDENCY_UNAVAILABLE",
                        message="前置工具步骤失败，当前步骤未执行。",
                        model_repairable=True,
                        infrastructure_retryable=False,
                        details={
                            "step_id": call.effective_step_id,
                            "failed_dependencies": [
                                dependency
                                for dependency in call.depends_on
                                if dependency in failed_steps
                            ],
                        },
                    ),
                )
                all_results.append(dependency_error)
                all_feedback.append(build_tool_feedback_message(dependency_error))
                all_traces.append(
                    ToolTraceEntry(
                        tool_call_id=call.tool_call_id,
                        tool_name=call.tool_name,
                        status="rejected",
                        error_code="DEPENDENCY_UNAVAILABLE",
                    )
                )
                failed_steps.add(call.effective_step_id)
                pending.pop(call.effective_step_id, None)
                failed += 1
            if not pending:
                break
            ready = [
                call for call in pending.values()
                if all(dependency in outputs for dependency in call.depends_on)
            ]
            if not ready:
                raise RuntimeError("tool dependency graph cannot make progress")
            resolved_calls = [
                call.model_copy(
                    update={
                        "arguments": resolve_typed_references(call.arguments, outputs),
                        "depends_on": [],
                    }
                )
                for call in ready
            ]
            wave = await execute_planner_tool_calls(
                decision=PlannerDecision(
                    action="call_tools",
                    tool_calls=resolved_calls,
                    decision_reason="execute a validated dependency wave",
                    confidence=decision.confidence,
                    plan_version=decision.plan_version,
                ),
                executor=executor,
                context=replace(
                    context,
                    remaining_tool_calls=max(
                        0,
                        context.remaining_tool_calls - executed,
                    ),
                ),
                successful_results_by_signature=successful_results_by_signature,
                round_index=round_index,
            )
            by_id = {item.tool_call_id: item for item in wave.tool_results}
            for original in ready:
                result = by_id.get(original.tool_call_id)
                if result is not None and result.success:
                    outputs[original.effective_step_id] = result.output
                else:
                    failed_steps.add(original.effective_step_id)
                pending.pop(original.effective_step_id, None)
            all_results.extend(wave.tool_results)
            all_traces.extend(wave.tool_traces)
            all_feedback.extend(wave.feedback_messages)
            all_reuse.extend(wave.reused_tool_calls)
            executed += wave.executed_call_count
            reused += wave.reused_call_count
            succeeded += wave.successful_call_count
            failed += wave.failed_call_count
        return ToolExecutorNodeResult(
            tool_results=all_results,
            tool_traces=all_traces,
            feedback_messages=all_feedback,
            reused_tool_calls=all_reuse,
            executed_call_count=executed,
            reused_call_count=reused,
            successful_call_count=succeeded,
            failed_call_count=failed,
        )

    reuse_cache = dict(
        successful_results_by_signature or {}
    )

    calls_to_execute: list[ToolCallRequest] = []
    reused_by_call_id: dict[
        str,
        tuple[str, ToolResult],
    ] = {}

    for tool_call in decision.tool_calls:
        signature = build_tool_reuse_signature(
            executor=executor,
            tool_call=tool_call,
            context=context,
        )

        previous_result = (
            reuse_cache.get(signature)
            if signature is not None
            else None
        )

        if (
            signature is not None
            and previous_result is not None
            and previous_result.success
        ):
            reused_by_call_id[tool_call.tool_call_id] = (
                signature,
                previous_result,
            )
            continue

        calls_to_execute.append(tool_call)

    outcomes = (
        await executor.execute_many(
            calls_to_execute,
            context=context,
        )
        if calls_to_execute
        else []
    )

    tool_results = [
        outcome.result
        for outcome in outcomes
    ]

    tool_traces = [
        outcome.trace
        for outcome in outcomes
    ]

    executed_result_by_call_id = {
        result.tool_call_id: result
        for result in tool_results
    }

    feedback_messages: list[dict[str, Any]] = []
    reuse_audits: list[ToolReuseAudit] = []

    for tool_call in decision.tool_calls:
        reused_entry = reused_by_call_id.get(
            tool_call.tool_call_id
        )

        if reused_entry is not None:
            signature, original_result = reused_entry

            feedback_messages.append(
                _build_reused_feedback_message(
                    requested_call=tool_call,
                    original_result=original_result,
                )
            )

            reuse_audits.append(
                ToolReuseAudit(
                    requested_tool_call_id=(
                        tool_call.tool_call_id
                    ),
                    reused_from_tool_call_id=(
                        original_result.tool_call_id
                    ),
                    tool_name=tool_call.tool_name,
                    signature=signature,
                    round_index=round_index,
                )
            )
            continue

        executed_result = executed_result_by_call_id.get(
            tool_call.tool_call_id
        )

        if executed_result is None:
            raise RuntimeError(
                "工具执行节点没有得到对应工具调用的执行结果。"
            )

        feedback_messages.append(
            build_tool_feedback_message(executed_result)
        )

    executed_successful_count = sum(
        1
        for tool_result in tool_results
        if tool_result.success
    )

    failed_call_count = (
        len(tool_results) - executed_successful_count
    )

    reused_call_count = len(reuse_audits)

    return ToolExecutorNodeResult(
        tool_results=tool_results,
        tool_traces=tool_traces,
        feedback_messages=feedback_messages,
        reused_tool_calls=reuse_audits,
        executed_call_count=len(calls_to_execute),
        reused_call_count=reused_call_count,
        successful_call_count=(
            executed_successful_count
            + reused_call_count
        ),
        failed_call_count=failed_call_count,
    )
