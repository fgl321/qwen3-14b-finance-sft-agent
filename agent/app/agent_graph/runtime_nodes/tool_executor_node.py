from __future__ import annotations

import json
from collections.abc import Mapping
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
)
from app.agent_graph.schemas.tool_schema import (
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
