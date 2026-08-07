from typing import Any, NotRequired, TypedDict


class FinanceAgentGraphState(TypedDict):
    """
    LangGraph 状态对象。

    Stage 3.2：
    1. 继续复用旧 FinanceAgent。
    2. 增加回答质量门控。
    3. 当知识库没命中但问题属于通用金融知识时，
       允许走通用金融兜底。

    Stage 4.1：
    1. 在旧 FinanceAgent 前增加问题路由。
    2. 支持一个问题同时需要多种能力。
    3. 根据能力结果编译执行计划。
    4. 保留 Stage 3 的质量门控和兜底机制。
    """

    # ============================================================
    # 输入字段
    # ============================================================
    user_message: str
    user_id: str
    thread_id: str

    # ============================================================
    # 可选输入字段
    # ============================================================
    request_id: NotRequired[str]
    tenant_id: NotRequired[str]
    knowledge_base_id: NotRequired[str]
    history_messages: NotRequired[list[dict[str, Any]]]

    # ============================================================
    # Stage 4.1 问题路由字段
    # ============================================================

    # 当前问题需要的能力，例如：
    # [
    #     "knowledge_retrieval",
    #     "financial_calculation",
    #     "complex_reasoning",
    # ]
    question_capabilities: NotRequired[list[str]]

    # 最终使用的路由器，例如：
    # hard_rule
    # llm_semantic_router
    # llm_fallback
    question_router: NotRequired[str]

    # 路由置信度：
    # high / medium / low
    question_router_confidence: NotRequired[str]

    # 路由器给出的判断原因
    question_router_reason: NotRequired[str]

    # 路由过程中是否使用了保守回退
    question_router_used_fallback: NotRequired[bool]

    # 硬规则命中的规则名称
    question_router_matched_rules: NotRequired[list[str]]

    # 保存完整的原始路由结果，方便日志记录和问题排查
    question_route_detail: NotRequired[dict[str, Any]]

    # 根据能力编译出的执行计划，例如：
    # ["general_finance_answer"]
    # 或：
    # ["finance_agent"]
    execution_plan: NotRequired[list[str]]

    # ============================================================
    # 旧 FinanceAgent 输出字段
    # ============================================================
    final_answer: NotRequired[str]
    agent_result: NotRequired[dict[str, Any]]
    executed_tools: NotRequired[list[dict[str, Any]]]
    usage: NotRequired[dict[str, Any]]
    finish_reason: NotRequired[str]
    message_count: NotRequired[int]
    safety_check: NotRequired[dict[str, Any]]

    # ============================================================
    # LangGraph 质量门控字段
    # ============================================================
    quality_gate: NotRequired[dict[str, Any]]
    needs_general_finance_fallback: NotRequired[bool]
    fallback_used: NotRequired[bool]
    fallback_reason: NotRequired[str]

    # ============================================================
    # 错误字段
    # ============================================================
    error: NotRequired[str]
