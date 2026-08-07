from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent_graph.service import get_finance_agent_graph_service


router = APIRouter(tags=["chat-graph"])


class ChatGraphRequest(BaseModel):
    """
    LangGraph Chat 测试请求。

    这里同时兼容 message 和 user_message：
    - message：更像前端接口字段
    - user_message：更像内部 Agent 字段

    二者传一个即可。
    """

    message: str | None = Field(default=None)
    user_message: str | None = Field(default=None)

    user_id: str = Field(default="graph_api_user_001")
    thread_id: str = Field(default="graph_api_thread_001")
    tenant_id: str = Field(default="tenant_001")
    knowledge_base_id: str = Field(default="kb_finance_basic")
    request_id: str | None = Field(default=None)

    history_messages: list[dict[str, Any]] = Field(default_factory=list)


class ChatGraphResponse(BaseModel):
    """
    LangGraph Chat 测试响应。

    注意：
    这里保留 answer 和 final_answer 两个字段：
    - answer：方便前端直接展示
    - final_answer：方便调试 LangGraph 状态
    """

    request_id: str
    answer: str
    final_answer: str

    fallback_used: bool = False
    finish_reason: str | None = None

    question_capabilities: list[str] = Field(default_factory=list)
    question_router: str | None = None
    question_router_confidence: str | None = None
    question_router_reason: str | None = None
    question_router_used_fallback: bool = False
    question_router_matched_rules: list[str] = Field(default_factory=list)
    execution_plan: list[str] = Field(default_factory=list)
    question_route_detail: dict[str, Any] | None = None

    quality_gate: dict[str, Any] | None = None
    executed_tools: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


@router.post("/api/chat/graph", response_model=ChatGraphResponse)
async def chat_with_graph(request: ChatGraphRequest) -> ChatGraphResponse:
    """
    Stage 3.4 LangGraph 测试入口。

    重要：
    这个接口不会替换旧 /api/chat。
    它只是为了单独验证 LangGraph 编排链路是否可以通过 HTTP 正常访问。
    """
    user_message = (request.user_message or request.message or "").strip()

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="message 或 user_message 不能为空",
        )

    service = get_finance_agent_graph_service()

    result = await service.run(
        request_id=request.request_id or f"graph-api-{uuid4()}",
        user_message=user_message,
        user_id=request.user_id,
        thread_id=request.thread_id,
        tenant_id=request.tenant_id,
        knowledge_base_id=request.knowledge_base_id,
        history_messages=request.history_messages,
    )

    if result.get("error"):
        raise HTTPException(
            status_code=500,
            detail=result["error"],
        )

    final_answer = result.get("final_answer") or ""

    return ChatGraphResponse(
        request_id=result.get("request_id") or "",
        answer=final_answer,
        final_answer=final_answer,
        fallback_used=bool(result.get("fallback_used", False)),
        finish_reason=result.get("finish_reason"),

        question_capabilities=result.get("question_capabilities") or [],
        question_router=result.get("question_router"),
        question_router_confidence=result.get("question_router_confidence"),
        question_router_reason=result.get("question_router_reason"),
        question_router_used_fallback=bool(
            result.get("question_router_used_fallback", False)
        ),
        question_router_matched_rules=(
                result.get("question_router_matched_rules") or []
        ),
        execution_plan=result.get("execution_plan") or [],
        question_route_detail=result.get("question_route_detail"),

        quality_gate=result.get("quality_gate"),
        executed_tools=result.get("executed_tools") or [],
        usage=result.get("usage") or {},
        error=None,
    )
