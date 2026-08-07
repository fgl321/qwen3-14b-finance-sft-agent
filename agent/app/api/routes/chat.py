from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Request

from app.agent.finance_agent import FinanceAgent
from app.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ExecutedToolPayload,
    RagCitationPayload,
    RagToolPayload,
)
from app.core.config import Settings
from app.core.logging import get_logger
from app.llm.deepseek_client import DeepSeekClient
from app.memory.memory_policy import LongTermMemoryPolicy
from app.memory.short_term_memory import ShortTermMemoryService
from app.rag.rag_audit import RagCitationAuditor
from app.rag.rag_quality_audit import RagQualityAuditor
from app.rag.rag_service import RagAnswerService
from app.tools.tool_audit import ToolCallAuditor


logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request_body: ChatRequest,
    request: Request,
) -> ChatResponse:
    settings: Settings = request.app.state.settings
    llm_client: DeepSeekClient = request.app.state.deepseek
    rag_service: RagAnswerService = request.app.state.rag_service
    short_memory: ShortTermMemoryService = request.app.state.short_memory

    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())

    tenant_id = request_body.tenant_id or "default"
    thread_id = request_body.thread_id or request_id or "default_thread"

    history_messages: list[dict[str, str]] = []

    if settings.short_memory_enabled:
        history_messages = short_memory.get_recent_messages(
            user_id=request_body.user_id,
            thread_id=thread_id,
            limit=settings.short_memory_max_messages,
        )

    memory_policy = LongTermMemoryPolicy()

    long_memory_result = memory_policy.process_user_message(
        user_message=request_body.message,
        user_id=request_body.user_id,
        tenant_id=tenant_id,
        thread_id=thread_id,
    )

    is_memory_record_request = _is_memory_record_request(request_body.message)

    if (
        is_memory_record_request
        and long_memory_result.saved_count > 0
        and long_memory_result.error is None
    ):
        answer = _build_memory_saved_answer(long_memory_result.saved_facts)

        if settings.short_memory_enabled:
            short_memory.append_turn(
                user_id=request_body.user_id,
                thread_id=thread_id,
                user_message=request_body.message,
                assistant_message=answer,
            )

        executed_tools: list[dict[str, Any]] = []

        rag_payload = RagToolPayload(
            used=False,
            sufficient=None,
            retrieved_count=None,
            citations=[],
        )

        rag_payload_dict = rag_payload.model_dump()

        rag_audit = RagCitationAuditor().audit(
            answer=answer,
            executed_tools=executed_tools,
        ).model_dump()

        rag_quality_audit = RagQualityAuditor().audit(
            answer=answer,
            executed_tools=executed_tools,
            rag_payload=rag_payload_dict,
        ).model_dump()

        tool_audit = ToolCallAuditor().audit(
            executed_tools=executed_tools,
        ).model_dump()

        response = ChatResponse(
            request_id=request_id,
            answer=answer,
            finish_reason="long_memory_saved",
            message_count=1,
            executed_tools=[],
            rag=rag_payload,
            safety_check=_build_direct_memory_safety_check(request_body.message),
            usage={
                "short_memory": {
                    "enabled": settings.short_memory_enabled,
                    "thread_id": thread_id,
                    "history_message_count": len(history_messages),
                    "max_messages": settings.short_memory_max_messages,
                },
                "long_memory": {
                    "enabled": True,
                    "tenant_id": tenant_id,
                    "user_id": request_body.user_id,
                    "thread_id": thread_id,
                    "loaded": long_memory_result.loaded,
                    "saved_count": long_memory_result.saved_count,
                    "saved_facts": long_memory_result.saved_facts,
                    "error": long_memory_result.error,
                    "direct_return": True,
                },
                "rag_audit": rag_audit,
                "rag_quality_audit": rag_quality_audit,
                "tool_audit": tool_audit,
            },
        )

        logger.info(
            "chat_request_finished_by_long_memory_direct_return",
            request_id=request_id,
            user_id=request_body.user_id,
            thread_id=thread_id,
            tenant_id=tenant_id,
            long_memory_saved_count=long_memory_result.saved_count,
            rag_audit_issue=rag_audit.get("issue"),
            rag_quality_level=rag_quality_audit.get("quality_level"),
        )

        return response

    agent_user_message = _build_agent_user_message(
        user_message=request_body.message,
        long_memory_context=long_memory_result.prompt_context,
    )

    agent = FinanceAgent(
        llm_client=llm_client,
        settings=settings,
        rag_service=rag_service,
    )

    result = await agent.run(
        user_message=agent_user_message,
        user_id=request_body.user_id,
        thread_id=thread_id,
        request_id=request_id,
        tenant_id=tenant_id,
        knowledge_base_id=request_body.knowledge_base_id,
        history_messages=history_messages,
    )

    if settings.short_memory_enabled:
        short_memory.append_turn(
            user_id=request_body.user_id,
            thread_id=thread_id,
            user_message=request_body.message,
            assistant_message=result.answer,
        )

    rag_payload = _extract_rag_payload(result.executed_tools)
    rag_payload_dict = rag_payload.model_dump()

    rag_audit = RagCitationAuditor().audit(
        answer=result.answer,
        executed_tools=result.executed_tools,
    ).model_dump()

    rag_quality_audit = RagQualityAuditor().audit(
        answer=result.answer,
        executed_tools=result.executed_tools,
        rag_payload=rag_payload_dict,
    ).model_dump()

    tool_audit = ToolCallAuditor().audit(
        executed_tools=result.executed_tools,
    ).model_dump()

    response = ChatResponse(
        request_id=result.request_id,
        answer=result.answer,
        finish_reason=result.finish_reason,
        message_count=result.message_count,
        executed_tools=[
            _build_executed_tool_payload(item)
            for item in result.executed_tools
        ],
        rag=rag_payload,
        safety_check=result.safety_check,
        usage={
            **(result.usage or {}),
            "short_memory": {
                "enabled": settings.short_memory_enabled,
                "thread_id": thread_id,
                "history_message_count": len(history_messages),
                "max_messages": settings.short_memory_max_messages,
            },
            "long_memory": {
                "enabled": True,
                "tenant_id": tenant_id,
                "user_id": request_body.user_id,
                "thread_id": thread_id,
                "loaded": long_memory_result.loaded,
                "saved_count": long_memory_result.saved_count,
                "saved_facts": long_memory_result.saved_facts,
                "error": long_memory_result.error,
                "direct_return": False,
            },
            "rag_audit": rag_audit,
            "rag_quality_audit": rag_quality_audit,
            "tool_audit": tool_audit,
        },
    )

    logger.info(
        "chat_request_finished",
        request_id=result.request_id,
        user_id=request_body.user_id,
        thread_id=thread_id,
        tenant_id=tenant_id,
        knowledge_base_id=request_body.knowledge_base_id,
        finish_reason=result.finish_reason,
        tool_count=len(result.executed_tools),
        rag_used=response.rag.used,
        rag_sufficient=response.rag.sufficient,
        short_memory_enabled=settings.short_memory_enabled,
        history_message_count=len(history_messages),
        long_memory_loaded=long_memory_result.loaded,
        long_memory_saved_count=long_memory_result.saved_count,
        rag_audit_issue=rag_audit.get("issue"),
        rag_citation_consistent=rag_audit.get("citation_consistent"),
        rag_quality_level=rag_quality_audit.get("quality_level"),
        tool_failed_count=tool_audit.get("failed_tool_calls"),
    )

    return response


def _is_memory_record_request(user_message: str) -> bool:
    if not user_message:
        return False

    text = (
        user_message.replace("，", ",")
        .replace("。", ".")
        .replace("：", ":")
        .replace("；", ";")
        .replace(" ", "")
        .replace("\n", "")
    )

    record_keywords = [
        "记录",
        "记一下",
        "记住",
        "保存",
        "存一下",
        "补充一下",
        "更新一下",
        "我的家庭信息",
        "家庭信息如下",
        "个人信息如下",
        "我的情况是",
        "我的资料是",
    ]

    return any(keyword in text for keyword in record_keywords)


def _build_memory_saved_answer(saved_facts: list[dict[str, Any]]) -> str:
    lines = ["好的，我已记录以下长期信息："]

    for item in saved_facts:
        label = _fact_label(
            fact_type=item.get("fact_type"),
            fact_key=item.get("fact_key"),
        )

        value_text = _fact_value_text(
            fact_type=item.get("fact_type"),
            fact_key=item.get("fact_key"),
            fact_value=item.get("fact_value") or {},
        )

        lines.append(f"- {label}：{value_text}")

    lines.append("")
    lines.append("后续即使更换新的 thread_id，我也可以基于这些长期记忆继续为你做家庭财务规划。")

    return "\n".join(lines)


def _fact_label(
    *,
    fact_type: str | None,
    fact_key: str | None,
) -> str:
    label_map = {
        ("family_finance", "annual_necessary_expense"): "家庭年度必要支出",
        ("family_finance", "monthly_necessary_expense"): "家庭月度必要支出",
        ("family_finance", "mortgage_balance"): "房贷余额",
        ("family_finance", "available_assets"): "已有可用资产",
        ("family_finance", "annual_income"): "年度收入",
        ("family_finance", "monthly_income"): "月度收入",
        ("insurance", "husband_life_insurance"): "丈夫寿险保额",
        ("insurance", "wife_life_insurance"): "妻子寿险保额",
        ("insurance", "existing_life_insurance"): "已有寿险保额",
        ("preference", "risk_preference"): "风险偏好",
        ("family_profile", "husband_age"): "丈夫年龄",
        ("family_profile", "wife_age"): "妻子年龄",
        ("family_profile", "child_age"): "孩子年龄",
    }

    return label_map.get(
        (fact_type, fact_key),
        f"{fact_type}.{fact_key}",
    )


def _fact_value_text(
    *,
    fact_type: str | None,
    fact_key: str | None,
    fact_value: dict[str, Any],
) -> str:
    if fact_type == "preference" and fact_key == "risk_preference":
        value = fact_value.get("value")

        risk_map = {
            "conservative": "稳健 / 保守型",
            "balanced": "平衡型",
            "aggressive": "进取型",
        }

        return risk_map.get(value, str(value))

    if fact_type == "family_profile":
        age = fact_value.get("age")

        if age is not None:
            return f"{age}岁"

    amount = fact_value.get("amount")
    currency = fact_value.get("currency", "CNY")

    if amount is not None:
        return _format_money(amount, currency)

    original_text = fact_value.get("original_text")

    if original_text:
        return str(original_text)

    return str(fact_value)


def _format_money(amount: Any, currency: str = "CNY") -> str:
    try:
        number = float(amount)
    except Exception:
        return str(amount)

    if currency == "CNY":
        if number == 0:
            return "0元"

        if number % 10000 == 0:
            return f"{int(number / 10000)}万元"

        return f"{int(number)}元"

    return f"{number:g} {currency}"


def _build_direct_memory_safety_check(user_message: str) -> dict[str, Any]:
    return {
        "safe": True,
        "decision": "allow",
        "risk_level": "low",
        "findings": [],
        "explanation": "该回答仅确认已记录用户提供的家庭财务信息，不涉及投资建议、产品推荐或收益承诺。",
        "user_message": user_message,
        "stage": "output",
        "rewritten": False,
        "guard_usage": None,
        "guard_model": "local_rule",
        "guard_finish_reason": "direct_memory_saved",
    }


def _build_agent_user_message(
    *,
    user_message: str,
    long_memory_context: str,
) -> str:
    if not long_memory_context:
        return user_message

    return f"""
你是一个金融规划助手。回答用户当前问题时，可以参考下面的长期记忆。

【长期记忆】
{long_memory_context}

【使用规则】
1. 长期记忆只作为上下文，不要生硬复述全部记忆。
2. 如果用户问到已记录的信息，应直接使用长期记忆回答。
3. 如果用户提供了新信息，应以新信息为准。
4. 不要把“长期记忆”这几个字暴露得太机械，正常自然回答即可。

【用户当前问题】
{user_message}
""".strip()


def _build_executed_tool_payload(
    payload: dict[str, Any],
) -> ExecutedToolPayload:
    return ExecutedToolPayload(
        tool_name=payload.get("tool_name"),
        ok=payload.get("ok"),
        arguments=payload.get("arguments") or {},
        result=payload.get("result"),
        usage=payload.get("usage"),
    )


def _extract_rag_payload(
    executed_tools: list[dict[str, Any]],
) -> RagToolPayload:
    for tool_payload in executed_tools:
        if tool_payload.get("tool_name") != "search_knowledge_base":
            continue

        result = tool_payload.get("result") or {}
        evidence_assessment = result.get("evidence_assessment") or {}
        raw_citations = result.get("citations") or []

        citations = [
            RagCitationPayload.model_validate(item)
            for item in raw_citations
        ]

        return RagToolPayload(
            used=True,
            sufficient=evidence_assessment.get("sufficient"),
            retrieved_count=result.get("retrieved_count"),
            citations=citations,
        )

    return RagToolPayload(
        used=False,
        sufficient=None,
        retrieved_count=None,
        citations=[],
    )
