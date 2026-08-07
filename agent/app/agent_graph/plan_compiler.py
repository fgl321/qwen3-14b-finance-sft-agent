from collections.abc import Iterable
from typing import Any

from app.agent_graph.question_router import QuestionCapability


GENERAL_FINANCE_ANSWER_STEP = "general_finance_answer"
FINANCE_AGENT_STEP = "finance_agent"


def _get_capability_value(capability: Any) -> str:
    """
    将能力值统一转换为字符串。

    支持两种输入：

    1. QuestionCapability 枚举：
       QuestionCapability.GENERAL_EXPLANATION

    2. 普通字符串：
       "general_explanation"
    """
    if isinstance(capability, QuestionCapability):
        return capability.value

    return str(capability).strip()


def compile_execution_plan(
    capabilities: Iterable[str | QuestionCapability],
) -> list[str]:
    """
    将问题路由器返回的多能力结果编译成第一版执行计划。

    第一版采用保守接入策略：

    只有单纯的通用概念解释：
        general_explanation
        → general_finance_answer

    只要包含以下任意一种能力：
        knowledge_retrieval
        financial_calculation
        memory_read
        complex_reasoning
        → finance_agent

    原因：
    旧 FinanceAgent 已经能够处理知识库、计算、记忆和复杂规划，
    第一版先复用成熟链路，不急着拆成多个独立节点。
    """
    normalized_capabilities = [
        _get_capability_value(capability)
        for capability in capabilities
    ]

    normalized_capabilities = [
        capability
        for capability in normalized_capabilities
        if capability
    ]

    if not normalized_capabilities:
        raise ValueError("capabilities 不能为空")

    allowed_capabilities = {
        capability.value
        for capability in QuestionCapability
    }

    unknown_capabilities = [
        capability
        for capability in normalized_capabilities
        if capability not in allowed_capabilities
    ]

    if unknown_capabilities:
        raise ValueError(
            "发现未知能力："
            f"{unknown_capabilities}"
        )

    unique_capabilities = set(normalized_capabilities)

    if unique_capabilities == {
        QuestionCapability.GENERAL_EXPLANATION.value
    }:
        return [GENERAL_FINANCE_ANSWER_STEP]

    return [FINANCE_AGENT_STEP]
