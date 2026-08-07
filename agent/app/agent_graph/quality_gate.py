from typing import Any

from app.agent_graph.state import FinanceAgentGraphState


KB_SPECIFIC_KEYWORDS = (
    "知识库",
    "文档",
    "资料",
    "上传",
    "附件",
    "原文",
    "报告",
    "论文",
    "根据材料",
    "根据文档",
    "根据知识库",
    "引用",
    "出处",
    "citation",
    "citations",
)

NO_EVIDENCE_ANSWER_PATTERNS = (
    "当前知识库中没有找到足够依据",
    "知识库中没有找到足够依据",
    "不能基于知识库给出确定回答",
    "没有检索到相关证据",
    "缺少可用于回答该问题的知识库证据",
)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def user_is_explicitly_asking_for_knowledge_base(user_message: str) -> bool:
    """
    判断用户是否明确要求“基于知识库/文档/上传资料”回答。

    如果用户明确要求基于资料回答，那么知识库没命中时不能兜底乱答。
    """
    normalized = (user_message or "").strip().lower()
    return any(keyword.lower() in normalized for keyword in KB_SPECIFIC_KEYWORDS)


def final_answer_looks_like_no_evidence_answer(final_answer: str) -> bool:
    """
    判断最终回答是否是典型的“知识库证据不足”回答。
    """
    answer = (final_answer or "").strip()
    return any(pattern in answer for pattern in NO_EVIDENCE_ANSWER_PATTERNS)


def find_no_evidence_rag_tool_result(
    executed_tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    从工具调用记录中找到“知识库检索没命中 / 证据不足”的结果。
    """
    for tool_call in executed_tools or []:
        tool_name = tool_call.get("tool_name")

        if tool_name != "search_knowledge_base":
            continue

        result = tool_call.get("result") or {}

        retrieved_count = _as_int(result.get("retrieved_count"), default=0)

        evidence_assessment = result.get("evidence_assessment") or {}
        sufficient = evidence_assessment.get("sufficient")

        if retrieved_count == 0:
            return {
                "reason": "search_knowledge_base retrieved_count=0",
                "tool_call": tool_call,
            }

        if sufficient is False:
            return {
                "reason": "search_knowledge_base evidence sufficient=false",
                "tool_call": tool_call,
            }

    return None


def build_quality_gate_result(
    state: FinanceAgentGraphState,
) -> dict[str, Any]:
    """
    LangGraph 回答质量门控。

    目前只做一个非常明确的生产问题：
    - 旧 Agent 调用了知识库检索；
    - 但是知识库没有命中；
    - 最终回答变成“当前知识库没有依据”；
    - 用户又没有明确要求“必须基于知识库/文档”。

    这种情况下，允许进入通用金融解释兜底节点。
    """
    user_message = state.get("user_message", "")
    final_answer = state.get("final_answer", "")
    executed_tools = state.get("executed_tools", [])

    no_evidence_tool_result = find_no_evidence_rag_tool_result(executed_tools)
    is_kb_specific = user_is_explicitly_asking_for_knowledge_base(user_message)
    answer_is_no_evidence = final_answer_looks_like_no_evidence_answer(final_answer)

    needs_fallback = bool(
        no_evidence_tool_result
        and answer_is_no_evidence
        and not is_kb_specific
    )

    if needs_fallback:
        reason = (
            "旧 Agent 调用了知识库检索，但没有检索到证据；"
            "用户问题不是必须基于知识库回答的问题，因此进入通用金融解释兜底。"
        )
    elif is_kb_specific and no_evidence_tool_result:
        reason = (
            "用户明确要求基于知识库/文档回答，知识库无证据时不做通用兜底。"
        )
    else:
        reason = "当前回答不需要兜底。"

    return {
        "needs_general_finance_fallback": needs_fallback,
        "reason": reason,
        "is_kb_specific": is_kb_specific,
        "answer_is_no_evidence": answer_is_no_evidence,
        "has_no_evidence_rag_tool_result": no_evidence_tool_result is not None,
    }
