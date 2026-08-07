from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.agent_graph.final_response_pipeline import (
    FinalResponseRequest,
)
from app.agent_graph.llm_task_planner import (
    PlannerRequest,
)
from app.agent_graph.production_dependencies import (
    ProductionGraphDependencies,
)
from app.agent_graph.production_state import (
    ProductionFinanceGraphState,
)
from app.agent_graph.release_contract import (
    STAGE_4_2_8_VERSION,
)
from app.agent_graph.runtime.agent_errors import (
    exception_to_agent_error,
    log_event,
)
from app.agent_graph.schemas.loop_schema import (
    AgentLoopResult,
)
from app.agent_graph.schemas.planner_schema import (
    normalize_execution_policy,
)
from app.core.logging import get_logger
from app.tools.tool_executor import (
    ToolExecutionContext,
)


logger = get_logger(__name__)


_GRAPH_VERSION = STAGE_4_2_8_VERSION


def _optional_tool_filter(
    values: Any,
) -> frozenset[str] | None:
    """
    将 API/State 中的工具过滤列表转换为运行时过滤条件。

    None、空列表和空集合都表示“不按该维度额外限制”。
    非空值会被清洗并转换为不可变集合。
    """

    if values is None:
        return None

    normalized = frozenset(
        str(item).strip()
        for item in values
        if str(item).strip()
    )

    return normalized or None


def prepare_production_run_node(
    state: ProductionFinanceGraphState,
) -> dict[str, Any]:
    """
    初始化当前轮次。

    即使同一个 thread_id 已经存在旧 Checkpoint，
    也必须清空上一轮的临时执行结果。
    """

    user_message = str(
        state.get("user_message") or ""
    ).strip()

    user_id = str(
        state.get("user_id") or ""
    ).strip()

    thread_id = str(
        state.get("thread_id") or ""
    ).strip()

    if not user_message:
        raise ValueError(
            "state.user_message 不能为空。"
        )

    if not user_id:
        raise ValueError(
            "state.user_id 不能为空。"
        )

    if not thread_id:
        raise ValueError(
            "state.thread_id 不能为空。"
        )

    request_id = str(
        state.get("request_id")
        or f"prod-request-{uuid4()}"
    )

    run_id = str(
        state.get("run_id")
        or f"prod-run-{uuid4()}"
    )

    return {
        "request_id": request_id,
        "run_id": run_id,
        "user_message": user_message,
        "user_id": user_id,
        "thread_id": thread_id,
        "tenant_id": (
            str(
                state.get("tenant_id")
                or "default"
            )
        ),
        "knowledge_base_id": (
            str(
                state.get("knowledge_base_id")
                or "kb_finance_basic"
            )
        ),
        "history_messages": list(
            state.get("history_messages") or []
        ),
        "context_summary": str(
            state.get("context_summary") or ""
        ),
        "route_context": dict(
            state.get("route_context") or {}
        ),
        "allowed_tool_names": list(
            state.get("allowed_tool_names") or []
        ),
        "allowed_tool_groups": list(
            state.get("allowed_tool_groups")
            or ["financial_calculation"]
        ),
        "execution_policy": (
            normalize_execution_policy(
                state.get("execution_policy")
            )
        ),
        "remaining_tool_calls": int(
            state.get("remaining_tool_calls")
            or 12
        ),
        "allow_side_effects": bool(
            state.get("allow_side_effects", False)
        ),

        # 清空上一轮临时状态
        "agent_loop_result": None,
        "final_response_result": None,
        "status": "running",
        "final_answer": "",
        "finish_reason": "",
        "usage": {},
        "error": None,
        "graph_version": _GRAPH_VERSION,
    }


def build_agent_loop_node(
    dependencies: ProductionGraphDependencies,
):
    async def agent_loop_node(
        state: ProductionFinanceGraphState,
    ) -> dict[str, Any]:
        request_id = state["request_id"]
        run_id = state["run_id"]

        logger.info(
            "production_graph_agent_loop_started",
            request_id=request_id,
            run_id=run_id,
            thread_id=state["thread_id"],
        )

        try:
            result = await dependencies.agent_loop.run(
                request=PlannerRequest(
                    request_id=request_id,
                    run_id=run_id,
                    user_message=(
                        state["user_message"]
                    ),
                    history_messages=list(
                        state.get(
                            "history_messages"
                        )
                        or []
                    ),
                    context_summary=str(
                        state.get(
                            "context_summary"
                        )
                        or ""
                    ),
                    route_context=dict(
                        state.get(
                            "route_context"
                        )
                        or {}
                    ),
                    allowed_tool_names=(
                        _optional_tool_filter(
                            state.get(
                                "allowed_tool_names"
                            )
                        )
                    ),
                    allowed_tool_groups=(
                        _optional_tool_filter(
                            state.get(
                                "allowed_tool_groups"
                            )
                        )
                    ),
                    execution_policy=(
                        normalize_execution_policy(
                            state.get(
                                "execution_policy"
                            )
                        )
                    ),
                    remaining_tool_calls=int(
                        state.get(
                            "remaining_tool_calls"
                        )
                        or 12
                    ),
                ),
                execution_context=(
                    ToolExecutionContext(
                        request_id=request_id,
                        run_id=run_id,
                        tenant_id=state.get(
                            "tenant_id",
                            "default",
                        ),
                        user_id=state["user_id"],
                        role="user",
                        allowed_tool_names=(
                            _optional_tool_filter(
                                state.get(
                                    "allowed_tool_names"
                                )
                            )
                        ),
                        allowed_tool_groups=(
                            _optional_tool_filter(
                                state.get(
                                    "allowed_tool_groups"
                                )
                            )
                        ),
                        allow_side_effects=bool(
                            state.get(
                                "allow_side_effects",
                                False,
                            )
                        ),
                        remaining_tool_calls=int(
                            state.get(
                                "remaining_tool_calls"
                            )
                            or 12
                        ),
                    )
                ),
            )

        except Exception as exc:
            error = exception_to_agent_error(
                exc,
                stage="agent_loop",
                request_id=request_id,
                run_id=run_id,
            )

            log_event(
                logger,
                "error",
                "production_graph_agent_loop_failed",
                request_id=request_id,
                run_id=run_id,
                error_id=error.error_id,
                error_code=error.code,
                error_type=type(exc).__name__,
            )

            return {
                "status": "fallback",
                "finish_reason": (
                    "agent_loop_node_failed"
                ),
                "error": error.model_dump(
                    mode="json"
                ),
            }

        return {
            "agent_loop_result": (
                result.model_dump(mode="json")
            ),
            "status": result.status,
            "finish_reason": result.finish_reason,
            "error": None,
        }

    return agent_loop_node


def route_after_agent_loop(
    state: ProductionFinanceGraphState,
) -> str:
    if state.get("error"):
        return "failure"

    if not state.get("agent_loop_result"):
        return "failure"

    return "final_response"


def build_final_response_node(
    dependencies: ProductionGraphDependencies,
):
    async def final_response_node(
        state: ProductionFinanceGraphState,
    ) -> dict[str, Any]:
        request_id = state["request_id"]
        run_id = state["run_id"]

        try:
            loop_result = (
                AgentLoopResult.model_validate(
                    state["agent_loop_result"]
                )
            )

            result = (
                await dependencies
                .final_response_pipeline
                .run(
                    FinalResponseRequest(
                        request_id=request_id,
                        run_id=run_id,
                        user_message=(
                            state["user_message"]
                        ),
                        loop_result=loop_result,
                        context_summary=str(
                            state.get(
                                "context_summary"
                            )
                            or ""
                        ),
                        citations=[],
                    )
                )
            )

        except Exception as exc:
            error = exception_to_agent_error(
                exc,
                stage="final_response",
                request_id=request_id,
                run_id=run_id,
            )

            log_event(
                logger,
                "error",
                "production_graph_final_response_failed",
                request_id=request_id,
                run_id=run_id,
                error_id=error.error_id,
                error_code=error.code,
                error_type=type(exc).__name__,
            )

            return {
                "status": "fallback",
                "final_answer": (
                    "系统已经完成部分处理，"
                    "但最终回答生成失败。"
                    "请稍后重新提交问题。"
                ),
                "finish_reason": (
                    "final_response_node_failed"
                ),
                "error": error.model_dump(
                    mode="json"
                ),
            }

        logger.info(
            "production_graph_finished",
            request_id=request_id,
            run_id=run_id,
            status=result.status,
            finish_reason=result.finish_reason,
        )

        return {
            "final_response_result": (
                result.model_dump(mode="json")
            ),
            "status": result.status,
            "final_answer": result.answer,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
            "error": None,
        }

    return final_response_node


def failure_response_node(
    state: ProductionFinanceGraphState,
) -> dict[str, Any]:
    """
    节点异常时的确定性安全回退。

    不把内部异常信息直接返回给用户。
    """

    return {
        "status": "fallback",
        "final_answer": (
            "系统暂时无法安全完成本次处理，"
            "请稍后重新提交问题。"
        ),
        "finish_reason": (
            state.get("finish_reason")
            or "production_graph_failed"
        ),
    }
