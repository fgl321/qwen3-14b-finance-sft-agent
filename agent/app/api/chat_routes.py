from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.agent.finance_agent import FinanceAgent
from app.api.chat_schema import (
    ChatRequest,
    ChatResponse,
    ExecutedToolPayload,
    RagCitationPayload,
    RagToolPayload,
)
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.llm.deepseek_client import DeepSeekClient


logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["chat"],
)


def get_llm_client(
    settings: Settings = Depends(get_settings),
) -> DeepSeekClient:
    return DeepSeekClient(settings)


async def get_finance_agent(
    settings: Settings = Depends(get_settings),
    llm_client: DeepSeekClient = Depends(get_llm_client),
) -> FinanceAgent:
    return FinanceAgent(
        llm_client=llm_client,
        settings=settings,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    agent: FinanceAgent = Depends(get_finance_agent),
) -> ChatResponse:
    try:
        result = await agent.run(
            user_message=request.message,
            user_id=request.user_id,
            thread_id=request.thread_id,
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
        )

        response = ChatResponse(
            request_id=result.request_id,
            answer=result.answer,
            finish_reason=result.finish_reason,
            message_count=result.message_count,
            executed_tools=[
                _build_executed_tool_payload(item)
                for item in result.executed_tools
            ],
            rag=_extract_rag_payload(result.executed_tools),
            safety_check=result.safety_check,
            usage=result.usage,
        )

        logger.info(
            "chat_request_finished",
            request_id=result.request_id,
            user_id=request.user_id,
            thread_id=request.thread_id,
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            finish_reason=result.finish_reason,
            tool_count=len(result.executed_tools),
            rag_used=response.rag.used,
            rag_sufficient=response.rag.sufficient,
        )

        return response

    except Exception as exc:
        logger.exception(
            "chat_request_failed",
            user_id=request.user_id,
            thread_id=request.thread_id,
            tenant_id=request.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "聊天接口执行失败。",
                "error": str(exc),
            },
        ) from exc

    finally:
        llm_client = getattr(agent, "llm_client", None)
        if llm_client is not None:
            await llm_client.close()


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
