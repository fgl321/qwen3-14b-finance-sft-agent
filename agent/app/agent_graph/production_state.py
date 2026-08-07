from __future__ import annotations

from typing import Any, TypedDict

from app.agent_graph.schemas.planner_schema import (
    ExecutionPolicy,
)


class ProductionFinanceGraphState(
    TypedDict,
    total=False,
):
    """
    Stage 4.2 生产主图状态。

    Checkpointer 中只存放 JSON 可序列化的数据，
    不直接保存 LLM 客户端、工具执行器或 Pydantic 对象。
    """

    # 请求身份
    request_id: str
    run_id: str

    user_id: str
    thread_id: str
    tenant_id: str
    knowledge_base_id: str

    # 用户输入
    user_message: str
    history_messages: list[dict[str, Any]]

    # 上下文与路由
    context_summary: str
    route_context: dict[str, Any]

    allowed_tool_names: list[str]
    allowed_tool_groups: list[str]
    execution_policy: ExecutionPolicy

    # 运行预算
    remaining_tool_calls: int
    allow_side_effects: bool

    # Agent 工具循环
    agent_loop_result: dict[str, Any] | None

    # 最终回答流水线
    final_response_result: dict[str, Any] | None

    # 对外输出
    status: str
    final_answer: str
    finish_reason: str

    usage: dict[str, Any]

    # 统一错误结构；只保存安全摘要，不保存原始异常堆栈
    error: dict[str, Any] | None

    # 图协议版本，便于后续迁移
    graph_version: str
